"""
Full Waveform Inversion (FWI) – METHOD 2: Column penalty loss
=============================================================
Stessa struttura di fwi_method2.py con l'aggiunta di uno scheduling
dinamico di alpha durante le iterazioni (curriculum learning / loss annealing).

    total_loss = alpha(t) * data_loss + (1 - alpha(t)) * col_loss

Tre strategie disponibili via --alpha_schedule:
  fixed      → alpha costante (comportamento originale)
  linear     → alpha cresce linearmente da alpha_start a alpha_end
  sigmoid    → transizione sigmoidale (più netta, centrata a --alpha_pivot)
  cosine     → annealing cosinusoidale (transizione morbida)

Image and Sound Processing Lab - Politecnico di Milano
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pinn_utils import bandpass_shots, forward, set_gpu, Siren

os.nice(10)
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Alpha scheduling
# ---------------------------------------------------------------------------

def get_alpha(
    epoch: int,
    total_epochs: int,
    schedule: str,
    alpha_start: float,
    alpha_end: float,
    pivot: float = 0.5,
    steepness: float = 10.0,
) -> float:
    """
    Compute the current alpha value based on the chosen schedule.

    Args:
        epoch         : current epoch (0-indexed).
        total_epochs  : total number of FWI iterations.
        schedule      : one of 'fixed', 'linear', 'sigmoid', 'cosine'.
        alpha_start   : initial alpha value (low → col_loss dominates early).
        alpha_end     : final alpha value   (high → data_loss dominates late).
        pivot         : fractional epoch [0,1] where the sigmoid is centred
                        (only used in 'sigmoid' mode).
        steepness     : controls how sharp the sigmoid transition is
                        (only used in 'sigmoid' mode).

    Returns:
        Scalar alpha value in [alpha_start, alpha_end].
    """
    t = epoch / max(total_epochs - 1, 1)   # normalised time in [0, 1]

    if schedule == "fixed":
        return alpha_start   # alpha_start is used as the fixed value

    elif schedule == "linear":
        return alpha_start + (alpha_end - alpha_start) * t

    elif schedule == "sigmoid":
        # Shifted sigmoid: 0 → alpha_start, 1 → alpha_end
        x = steepness * (t - pivot)
        sig = 1.0 / (1.0 + np.exp(-x))
        return alpha_start + (alpha_end - alpha_start) * sig

    elif schedule == "cosine":
        # Cosine annealing from alpha_start to alpha_end
        cos_val = (1.0 - np.cos(np.pi * t)) / 2.0
        return alpha_start + (alpha_end - alpha_start) * cos_val

    else:
        raise ValueError(
            f"Unknown alpha_schedule: '{schedule}'. "
            f"Choose from: fixed, linear, sigmoid, cosine."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_shots(policy: str, epoch: int, n_total: int, n_per_epoch: int) -> np.ndarray:
    if policy == "random":
        return np.random.choice(n_total, n_per_epoch, replace=False)
    elif policy == "sequential":
        return np.arange(epoch * n_per_epoch, (epoch + 1) * n_per_epoch) % n_total
    elif policy == "spaced":
        step = n_total // n_per_epoch
        return np.arange(epoch % step, n_total, step)
    else:
        raise ValueError(f"Unknown shot selection policy: '{policy}'.")


def _select_known_columns(
    n_cols_total: int,
    n_known: int,
    mode: str,
    seed: int = 42,
) -> np.ndarray:
    if n_known <= 0 or n_known > n_cols_total:
        raise ValueError(f"n_known_cols must be in [1, {n_cols_total}], got {n_known}.")
    if mode == "spaced":
        indices = np.round(np.linspace(0, n_cols_total - 1, n_known)).astype(int)
        return np.unique(indices)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n_cols_total, size=n_known, replace=False))
    else:
        raise ValueError(f"Unknown col_selection_mode: '{mode}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    npzfile  = np.load(args.obs_data_path, allow_pickle=True)
    d_obs_np = npzfile["d_obs_list"]
    dt_ms    = float(npzfile["dt"])
    dt_s     = dt_ms / 1000.0
    spacing  = npzfile["spacing"]
    dh       = float(spacing[0])
    wave_np  = npzfile["wave"]
    domain_pad = tuple(npzfile["domain_pad"].tolist())
    domain     = tuple(npzfile["domain"].tolist())
    nbl        = int(npzfile["nbl"])
    pmlc_np    = npzfile["pmlc"]

    fwi_iterations        = args.fwi_iterations
    shots_per_epoch       = args.shots_per_epoch
    shot_selection_policy = args.shot_selection_policy
    use_all_shots         = args.all_shots

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    rec_coordinates = npzfile["rec_coordinates"]
    src_coordinates = npzfile["src_coordinates"]
    sources  = (src_coordinates // spacing).astype(int) + [nbl, nbl]
    recz     = int(nbl + float(rec_coordinates[0, 1]) / float(spacing[0]))

    n_total_shots    = len(sources)
    num_mini_batches = n_total_shots // shots_per_epoch if use_all_shots else 1
    epochs_per_plot  = 1 if use_all_shots else 20

    # Known columns
    Nx = domain[1]
    if args.n_known_cols is not None:
        n_known = args.n_known_cols
        mode    = args.col_selection_mode
        known_col_idx = _select_known_columns(Nx, n_known, mode, seed=args.col_seed)
    else:
        col_spacing   = args.known_col_spacing
        known_col_idx = np.arange(0, Nx, col_spacing)
        n_known       = len(known_col_idx)
        mode          = "spaced"

    true_vp      = np.load(args.true_vp_path)["vp"].T
    known_cols_t = torch.from_numpy(
                       true_vp[:, known_col_idx].astype(np.float32)
                   ).to(dev)
    col_idx_t    = torch.tensor(known_col_idx, dtype=torch.long, device=dev)
    col_scale    = known_cols_t.abs().mean().clamp(min=1e-8)

    # Alpha schedule setup
    schedule    = args.alpha_schedule
    alpha_start = args.alpha_start
    alpha_end   = args.alpha_end
    # For 'fixed' mode: alpha_start is the constant value (mirrors old --alpha)

    print(f"\n[Method 2 - Column penalty loss + dynamic alpha]")
    print(f"Alpha schedule     : {schedule}")
    if schedule == "fixed":
        print(f"  alpha (fixed)    : {alpha_start:.2f}")
    else:
        print(f"  alpha_start      : {alpha_start:.2f}  (col_loss dominates early)")
        print(f"  alpha_end        : {alpha_end:.2f}  (data_loss dominates late)")
        if schedule == "sigmoid":
            print(f"  pivot            : {args.alpha_pivot:.2f} of total epochs")
            print(f"  steepness        : {args.alpha_steepness:.1f}")
    print(f"Known col mode     : {mode}  ({n_known} / {Nx} cols)")
    print(f"Total shots        : {n_total_shots}")
    print(f"Shots per epoch    : {shots_per_epoch}")
    print(f"FWI iterations     : {fwi_iterations}\n")

    # Preview the alpha schedule
    if args.plot:
        preview_epochs = np.arange(fwi_iterations)
        preview_alphas = [
            get_alpha(e, fwi_iterations, schedule, alpha_start, alpha_end,
                      args.alpha_pivot, args.alpha_steepness)
            for e in preview_epochs
        ]
        plt.figure(figsize=(8, 3))
        plt.plot(preview_epochs, preview_alphas, lw=2, color="steelblue", label="alpha(t)")
        plt.plot(preview_epochs, [1 - a for a in preview_alphas],
                 lw=2, color="darkorange", ls="--", label="1-alpha(t)")
        plt.xlabel("Epoch")
        plt.ylabel("Weight")
        plt.title(f"Alpha schedule: {schedule}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Multi-scale frequency schedule
    if args.multiscale:
        stage_len = fwi_iterations // 3
        freq_schedule = [
            (0,             stage_len,         1.0,  5.0),
            (stage_len,     2 * stage_len,     1.0, 10.0),
            (2 * stage_len, fwi_iterations,    1.0,  None),
        ]
    else:
        freq_schedule = [(0, fwi_iterations, None, None)]

    # SIREN
    imvel = Siren(
        in_features=2, out_features=1, hidden_features=128, hidden_layers=4,
        outermost_linear=True, domain_shape=domain,
        first_omega_0=10.0, hidden_omega_0=10.0,
        pretrained=os.path.abspath(args.siren_path),
    ).to(dev)

    l2loss = torch.nn.MSELoss()
    opt    = torch.optim.AdamW(imvel.parameters(), lr=1e-4, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6
    )

    obs_t   = torch.from_numpy(d_obs_np).float().to(dev)
    wave_t  = torch.from_numpy(wave_np.astype(np.float32)).to(dev)
    coords  = imvel.coords.to(dev)
    pmlc_t  = torch.from_numpy(pmlc_np.astype(np.float32)).to(dev)

    LOSS, LOSS_ALL           = [], []
    RAW_DATA_LOSS, RAW_COL_LOSS = [], []
    ALPHA_HISTORY            = []   # track how alpha evolved

    best_loss, best_epoch = float("inf"), 0
    obs_scale = obs_t.abs().max().clamp(min=1e-8)

    fwd_kwargs = dict(
        wave=wave_t, src_list=np.array(sources), domain=domain_pad,
        dt=dt_s, h=dh, dev=dev, recz=recz, b=pmlc_t, pmln=nbl,
    )

    # FWI loop
    for epoch in tqdm(range(fwi_iterations), desc="FWI [M2-dynAlpha]"):
        opt.zero_grad()

        # Compute alpha for this epoch
        alpha = get_alpha(
            epoch, fwi_iterations, schedule,
            alpha_start, alpha_end,
            args.alpha_pivot, args.alpha_steepness,
        )
        ALPHA_HISTORY.append(alpha)

        active_f_low, active_f_high = None, None
        if args.multiscale:
            for (s, e, fl, fh) in freq_schedule:
                if s <= epoch < e:
                    active_f_low, active_f_high = fl, fh
                    break

        INNER_LOSS, INNER_DATA_LOSS, INNER_COL_LOSS = [], [], []

        for inner_epoch in range(num_mini_batches):
            src_idx = _select_shots(shot_selection_policy,
                                    epoch * num_mini_batches + inner_epoch,
                                    n_total_shots, shots_per_epoch)
            fwd_kwargs["src_list"] = np.array(sources)[src_idx]

            vp, _ = imvel(coords)
            vp = vp * 1.0 + 3.0
            vp = torch.clamp(vp, min=1.5, max=4.5)

            col_loss = l2loss(vp[:, col_idx_t] / col_scale, known_cols_t / col_scale)

            vp_pad = torch.nn.functional.pad(
                vp.unsqueeze(0).unsqueeze(0), (nbl, nbl, nbl, nbl), mode="replicate",
            )[0, 0]

            syn = forward(c=vp_pad, **fwd_kwargs)
            if torch.isnan(syn).any():
                raise RuntimeError(f"NaN in synthetics at epoch {epoch}.")

            obs_batch = obs_t[src_idx]
            data_loss = l2loss(syn / obs_scale, obs_batch / obs_scale)

            if active_f_high is not None:
                syn_filt = bandpass_shots(syn.detach().cpu().numpy(), active_f_low, active_f_high, dt_s)
                obs_filt = bandpass_shots(obs_batch.detach().cpu().numpy(), active_f_low, active_f_high, dt_s)
                syn_f    = torch.from_numpy(syn_filt).to(dev)
                obs_f    = torch.from_numpy(obs_filt).to(dev)
                scale_f  = obs_f.abs().max().clamp(min=1e-8)
                data_loss = l2loss(syn_f / scale_f, obs_f / scale_f)

            loss = (alpha * data_loss + (1.0 - alpha) * col_loss) / num_mini_batches

            INNER_DATA_LOSS.append(data_loss.item())
            INNER_COL_LOSS.append(col_loss.item())
            INNER_LOSS.append(loss.item())
            LOSS_ALL.append(loss.item())
            loss.backward()

        torch.nn.utils.clip_grad_norm_(imvel.parameters(), max_norm=1.0)
        opt.step()
        sched.step(epoch)

        epoch_loss     = float(np.mean(INNER_LOSS))
        epoch_data_raw = float(np.mean(INNER_DATA_LOSS))
        epoch_col_raw  = float(np.mean(INNER_COL_LOSS))

        LOSS.append(epoch_loss)
        RAW_DATA_LOSS.append(epoch_data_raw)
        RAW_COL_LOSS.append(epoch_col_raw)

        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            if not args.debug:
                torch.save(imvel.state_dict(), os.path.join(out_dir, "fwi_best_model.pth"))

        if epoch % epochs_per_plot == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:5d} | "
                f"alpha={alpha:.3f} | "
                f"Loss: {epoch_loss:.6e} | "
                f"data_loss (raw): {epoch_data_raw:.6e} | "
                f"col_loss  (raw): {epoch_col_raw:.6e} | "
                f"LR: {lr_now:.2e} | Best @ {best_epoch}"
            )

            inverted = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]
            fig, axes = plt.subplots(4, 1, figsize=(10, 16))

            # Panel 1: inverted model
            axes[0].imshow(inverted, cmap="RdBu_r", aspect="auto", vmin=1.0, vmax=4.5)
            axes[0].set_title(
                f"Inverted Vp — epoch {epoch}  "
                f"[M2, {schedule}, alpha={alpha:.3f}, n_known={n_known}]"
            )
            axes[0].set_xlabel("x [samples]"); axes[0].set_ylabel("z [samples]")
            for ci in known_col_idx:
                axes[0].axvline(x=ci, color="k", linewidth=0.4, alpha=0.5)

            # Panel 2: total loss
            axes[1].semilogy(LOSS, color="steelblue", lw=1.5, label="total loss")
            axes[1].axvline(best_epoch, color="red", ls="--", lw=1,
                            label=f"best ({best_epoch})")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Normalised MSE")
            axes[1].legend(); axes[1].set_title("Total loss")

            # Panel 3: raw loss terms
            axes[2].semilogy(RAW_DATA_LOSS, color="darkorange", lw=1.5,
                             label="data_loss (raw)")
            axes[2].semilogy(RAW_COL_LOSS, color="forestgreen", lw=1.5,
                             label="col_loss (raw)")
            axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Raw MSE")
            axes[2].legend(); axes[2].set_title("Raw loss terms (pre-weighting)")

            # Panel 4: alpha schedule evolution
            axes[3].plot(ALPHA_HISTORY, color="steelblue", lw=2, label="alpha(t)")
            axes[3].plot([1 - a for a in ALPHA_HISTORY],
                         color="darkorange", lw=2, ls="--", label="1-alpha(t)")
            axes[3].axvline(epoch, color="gray", ls=":", lw=1)
            axes[3].set_xlabel("Epoch"); axes[3].set_ylabel("Weight")
            axes[3].set_ylim(-0.05, 1.05)
            axes[3].legend(); axes[3].set_title("Alpha schedule")

            plt.tight_layout()
            if not args.debug:
                plt.savefig(f"{png_dir}/epoch{epoch:05d}.png", dpi=150, bbox_inches="tight")
            if args.plot:
                plt.show()
            plt.close(fig)

    # Save
    final_vp = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]
    if not args.debug:
        np.savez(
            os.path.join(out_dir, "fwi_results.npz"),
            vp=final_vp,
            LOSS=np.array(LOSS),
            LOSS_ALL=np.array(LOSS_ALL),
            RAW_DATA_LOSS=np.array(RAW_DATA_LOSS),
            RAW_COL_LOSS=np.array(RAW_COL_LOSS),
            ALPHA_HISTORY=np.array(ALPHA_HISTORY),
            known_col_idx=known_col_idx,
            allow_pickle=True,
        )
        torch.save(imvel.state_dict(), os.path.join(out_dir, "fwi_final_model.pth"))
        print(f"\nResults saved to: {out_dir}")
        print(f"Best model (epoch {best_epoch}, loss {best_loss:.6e}) -> fwi_best_model.pth")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FWI Method 2 — column penalty loss with dynamic alpha scheduling"
    )

    parser.add_argument("--obs_data_path",  type=str, default="./data/shots/marmousi_paper_sp15.npz")
    parser.add_argument("--siren_path",     type=str, default="./data/siren/marmousi_paper_sm10_sp15.pth")
    parser.add_argument("--out_dir",        type=str, default="./data/output/fwi_method2_alpha_regolabile_25_linear")
    parser.add_argument("--fwi_iterations", type=int, default=10000)
    parser.add_argument("--shots_per_epoch",type=int, default=5)
    parser.add_argument("--shot_selection_policy", type=str, default="random",
                        choices=["random", "sequential", "spaced"])
    parser.add_argument("--all_shots",   action="store_true")
    parser.add_argument("--multiscale",  action="store_true")
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--debug",       action="store_true")

    # Column selection
    parser.add_argument("--n_known_cols",       type=int,   default=None)
    parser.add_argument("--col_selection_mode", type=str,   default="spaced",
                        choices=["spaced", "random"])
    parser.add_argument("--col_seed",           type=int,   default=42)
    parser.add_argument("--known_col_spacing",  type=int,   default=50)   # legacy
    parser.add_argument("--true_vp_path",       type=str,
                        default="./data/v_models/marmousi_paper_sp15.npz")

    # Alpha scheduling
    parser.add_argument("--alpha_schedule", type=str, default="fixed",
                        choices=["fixed", "linear", "sigmoid", "cosine"],
                        help=(
                            "fixed   → constant alpha (original behaviour)\n"
                            "linear  → alpha grows linearly from start to end\n"
                            "sigmoid → sharp transition centred at --alpha_pivot\n"
                            "cosine  → smooth cosine annealing"
                        ))
    parser.add_argument("--alpha_start", type=float, default=0.5,
                        help="Initial alpha. In 'fixed' mode this is the only value used.")
    parser.add_argument("--alpha_end",   type=float, default=0.9,
                        help="Final alpha (ignored in 'fixed' mode).")
    parser.add_argument("--alpha_pivot", type=float, default=0.2,
                        help="Fraction of total epochs where sigmoid is centred [0,1]. "
                             "E.g. 0.2 means the transition happens at 20%% of training.")
    parser.add_argument("--alpha_steepness", type=float, default=10.0,
                        help="Sigmoid steepness. Higher = sharper transition.")

    args = parser.parse_args()
    main(args)