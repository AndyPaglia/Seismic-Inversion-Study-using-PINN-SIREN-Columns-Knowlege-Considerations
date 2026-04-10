"""
Full Waveform Inversion (FWI) – METHOD 2: Column penalty loss
=============================================================
Identico a fwi.py con una sola aggiunta alla loss: viene calcolato un MSE
tra le colonne "note" prodotte dalla SIREN e i valori veri, e sommato alla
data-misfit loss con peso (1 - alpha) per fare un giusto bilanciamento tra i due termini.:

    total_loss = alpha * data_loss + (1 - alpha) * col_loss

Entrambi i termini sono normalizzati sulla propria scala così da essere
direttamente confrontabili indipendentemente dal valore di alpha.

Selezione colonne note via --n_known_cols + --col_selection_mode:
  --n_known_cols 50  --col_selection_mode spaced   → 50 colonne a spacing fisso
  --n_known_cols 50  --col_selection_mode random   → 50 colonne estratte casualmente

(Compatibilità legacy: --known_col_spacing è ancora accettato se si preferisce
 specificare direttamente lo spacing invece del numero di colonne.)

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
        raise ValueError(f"Unknown shot selection policy: '{policy}'. "
                         f"Choose from: random, sequential, spaced.")


def _select_known_columns(
    n_cols_total: int,
    n_known: int,
    mode: str,
    seed: int = 42,
) -> np.ndarray:
    """
    Return the indices of the 'known' columns.

    Args:
        n_cols_total : total number of columns in the physical domain (Nx).
        n_known      : how many columns to mark as known.
        mode         : 'spaced'  → evenly spaced (deterministic),
                       'random'  → randomly sampled without replacement.
        seed         : RNG seed used only in 'random' mode.

    Returns:
        1-D array of column indices, sorted in ascending order.
    """
    if n_known <= 0 or n_known > n_cols_total:
        raise ValueError(
            f"n_known_cols must be in [1, {n_cols_total}], got {n_known}."
        )

    if mode == "spaced":
        # linspace gives n_known evenly-spaced points over [0, n_cols_total-1]
        indices = np.round(np.linspace(0, n_cols_total - 1, n_known)).astype(int)
        indices = np.unique(indices)          # safety: remove accidental duplicates
    elif mode == "random":
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_cols_total, size=n_known, replace=False))
    else:
        raise ValueError(
            f"Unknown col_selection_mode: '{mode}'. Choose from: spaced, random."
        )

    return indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load observed data and metadata
    # ------------------------------------------------------------------
    npzfile = np.load(args.obs_data_path, allow_pickle=True)
    d_obs_np = npzfile["d_obs_list"]

    dt_ms       = float(npzfile["dt"])
    dt_s        = dt_ms / 1000.0
    spacing     = npzfile["spacing"]
    dh          = float(spacing[0])
    wave_np     = npzfile["wave"]
    domain_pad  = tuple(npzfile["domain_pad"].tolist())
    domain      = tuple(npzfile["domain"].tolist())
    nbl         = int(npzfile["nbl"])
    pmlc_np     = npzfile["pmlc"]

    fwi_iterations        = args.fwi_iterations
    shots_per_epoch       = args.shots_per_epoch
    shot_selection_policy = args.shot_selection_policy
    use_all_shots         = args.all_shots

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    rec_coordinates = npzfile["rec_coordinates"]
    src_coordinates = npzfile["src_coordinates"]
    sources  = (src_coordinates // spacing).astype(int) + [nbl, nbl]
    recz     = int(nbl + float(rec_coordinates[0, 1]) / float(spacing[0]))

    n_total_shots    = len(sources)
    num_mini_batches = n_total_shots // shots_per_epoch if use_all_shots else 1
    epochs_per_plot  = 1 if use_all_shots else 20

    # ------------------------------------------------------------------
    # [M2] Resolve known-column indices
    #
    # Priority:
    #   1. --n_known_cols  (new interface — preferred)
    #   2. --known_col_spacing  (legacy interface — kept for compatibility)
    # ------------------------------------------------------------------
    Nx = domain[1]   # number of columns in the physical domain

    if args.n_known_cols is not None:
        # New interface: user specifies number of columns + selection mode
        n_known = args.n_known_cols
        mode    = args.col_selection_mode
        known_col_idx = _select_known_columns(Nx, n_known, mode, seed=args.col_seed)
        # Compute effective spacing for display purposes only
        effective_spacing = Nx / n_known
        print(f"Known col mode     : {mode}  ({n_known} cols out of {Nx})")
        print(f"Effective spacing  : ~{effective_spacing:.1f} cols")
    else:
        # Legacy interface: user specifies spacing directly
        col_spacing   = args.known_col_spacing
        known_col_idx = np.arange(0, Nx, col_spacing)
        n_known       = len(known_col_idx)
        mode          = "spaced"
        print(f"Known col mode     : spaced (legacy --known_col_spacing={col_spacing})")
        print(f"Known cols         : {n_known} out of {Nx}")

    # ------------------------------------------------------------------
    # Load true velocity and extract known columns
    # ------------------------------------------------------------------
    true_vp       = np.load(args.true_vp_path)["vp"].T          # (Nz, Nx)
    known_cols_t  = torch.from_numpy(
                        true_vp[:, known_col_idx].astype(np.float32)
                    ).to(dev)                                     # (Nz, n_known)
    col_idx_t     = torch.tensor(known_col_idx, dtype=torch.long, device=dev)
    col_scale     = known_cols_t.abs().mean().clamp(min=1e-8)
    alpha         = args.alpha

    print(f"\n[Method 2 - Column penalty loss]")
    print(f"Total shots        : {n_total_shots}")
    print(f"Shots per epoch    : {shots_per_epoch}")
    print(f"FWI iterations     : {fwi_iterations}")
    print(f"Shot policy        : {shot_selection_policy}")
    print(f"alpha              : {alpha:.2f}  (data={alpha:.2f}, col={1-alpha:.2f})")
    print(f"Known col indices  : {known_col_idx[:5]} ... {known_col_idx[-5:]}  (total {n_known})\n")

    # ------------------------------------------------------------------
    # Multi-scale frequency schedule
    # ------------------------------------------------------------------
    if args.multiscale:
        stage_len = fwi_iterations // 3
        freq_schedule = [
            (0,             stage_len,         1.0,  5.0),
            (stage_len,     2 * stage_len,     1.0, 10.0),
            (2 * stage_len, fwi_iterations,    1.0,  None),
        ]
        print("Multi-scale mode ON:")
        for s, e, fl, fh in freq_schedule:
            print(f"  epochs {s:4d}-{e:4d} -> f <= {f'{fh} Hz' if fh else 'full band'}")
    else:
        freq_schedule = [(0, fwi_iterations, None, None)]

    # ------------------------------------------------------------------
    # Diagnostic plot of observed data
    # ------------------------------------------------------------------
    if args.plot:
        plt.figure()
        plt.imshow(d_obs_np[0], cmap="RdBu_r", aspect="auto", clim=[-0.5, 0.5])
        plt.title("Observed data - shot 0")
        plt.colorbar()
        plt.show()

    # ------------------------------------------------------------------
    # SIREN model
    # ------------------------------------------------------------------
    imvel = Siren(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=4,
        outermost_linear=True,
        domain_shape=domain,
        first_omega_0=10.0,
        hidden_omega_0=10.0,
        pretrained=os.path.abspath(args.siren_path),
    ).to(dev)

    # ------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # ------------------------------------------------------------------
    l2loss = torch.nn.MSELoss()
    opt    = torch.optim.AdamW(imvel.parameters(), lr=1e-4, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # Move static tensors to device
    # ------------------------------------------------------------------
    obs_t    = torch.from_numpy(d_obs_np).float().to(dev)
    wave_t   = torch.from_numpy(wave_np.astype(np.float32)).to(dev)
    coords   = imvel.coords.to(dev)
    pmlc_t   = torch.from_numpy(pmlc_np.astype(np.float32)).to(dev)

    LOSS, LOSS_ALL = [], []
    # Separate tracking for raw (pre-weighting) loss terms
    RAW_DATA_LOSS, RAW_COL_LOSS = [], []

    best_loss, best_epoch = float("inf"), 0
    obs_scale = obs_t.abs().max().clamp(min=1e-8)

    fwd_kwargs = dict(
        wave=wave_t, src_list=np.array(sources), domain=domain_pad,
        dt=dt_s, h=dh, dev=dev, recz=recz, b=pmlc_t, pmln=nbl,
    )

    # ------------------------------------------------------------------
    # FWI loop
    # ------------------------------------------------------------------
    for epoch in tqdm(range(fwi_iterations), desc="FWI [M2]"):
        opt.zero_grad()

        active_f_low, active_f_high = None, None
        if args.multiscale:
            for (s, e, fl, fh) in freq_schedule:
                if s <= epoch < e:
                    active_f_low, active_f_high = fl, fh
                    break

        INNER_LOSS      = []
        INNER_DATA_LOSS = []   # raw data loss values (before alpha weighting)
        INNER_COL_LOSS  = []   # raw col  loss values (before (1-alpha) weighting)

        for inner_epoch in range(num_mini_batches):
            src_idx = _select_shots(shot_selection_policy,
                                    epoch * num_mini_batches + inner_epoch,
                                    n_total_shots, shots_per_epoch)
            fwd_kwargs["src_list"] = np.array(sources)[src_idx]

            # Build velocity model from SIREN
            vp, _ = imvel(coords)
            vp = vp * 1.0 + 3.0
            vp = torch.clamp(vp, min=1.5, max=4.5)

            # [M2] Column penalty
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
                syn_f, obs_f = torch.from_numpy(syn_filt).to(dev), torch.from_numpy(obs_filt).to(dev)
                data_loss = l2loss(syn_f / obs_f.abs().max().clamp(min=1e-8),
                                   obs_f / obs_f.abs().max().clamp(min=1e-8))

            # [M2] Combined loss: alpha * data + (1-alpha) * col
            loss = (alpha * data_loss + (1.0 - alpha) * col_loss) / num_mini_batches

            # Track raw (unweighted) values for inspection
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
            # Print both the combined loss and the raw individual terms
            print(
                f"  Epoch {epoch:5d} | "
                f"Loss: {epoch_loss:.6e} | "
                f"data_loss (raw): {epoch_data_raw:.6e} | "
                f"col_loss  (raw): {epoch_col_raw:.6e} | "
                f"LR: {lr_now:.2e} | "
                f"Best @ {best_epoch}"
            )

            inverted = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            axes[0].imshow(inverted, cmap="RdBu_r", aspect="auto", vmin=1.0, vmax=4.5)
            axes[0].set_title(
                f"Inverted Vp - epoch {epoch}  "
                f"[M2, n_known={n_known}, mode={mode}, alpha={alpha}]"
            )
            axes[0].set_xlabel("x [samples]")
            axes[0].set_ylabel("z [samples]")
            for ci in known_col_idx:
                axes[0].axvline(x=ci, color="k", linewidth=0.4, alpha=0.5)

            axes[1].semilogy(LOSS, color="steelblue", lw=1.5, label="total loss (weighted)")
            axes[1].axvline(best_epoch, color="red", ls="--", lw=1, label=f"best ({best_epoch})")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Normalised MSE")
            axes[1].legend()
            axes[1].set_title("Total loss")

            axes[2].semilogy(RAW_DATA_LOSS, color="darkorange", lw=1.5,
                             label=f"data_loss (raw, ×α={alpha:.2f})")
            axes[2].semilogy(RAW_COL_LOSS,  color="forestgreen", lw=1.5,
                             label=f"col_loss  (raw, ×(1-α)={1-alpha:.2f})")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Raw MSE (before weighting)")
            axes[2].legend()
            axes[2].set_title("Raw loss terms (pre-weighting) — useful for tuning alpha")

            plt.tight_layout()
            if not args.debug:
                plt.savefig(f"{png_dir}/epoch{epoch:05d}.png", dpi=150, bbox_inches="tight")
            if args.plot:
                plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    final_vp = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]
    if not args.debug:
        np.savez(
            os.path.join(out_dir, "fwi_results.npz"),
            vp=final_vp,
            LOSS=np.array(LOSS),
            LOSS_ALL=np.array(LOSS_ALL),
            RAW_DATA_LOSS=np.array(RAW_DATA_LOSS),
            RAW_COL_LOSS=np.array(RAW_COL_LOSS),
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
    parser = argparse.ArgumentParser(description="FWI Method 2 - column penalty loss")

    parser.add_argument("--obs_data_path",  type=str, default="./data/shots/marmousi_paper_sp15.npz")
    parser.add_argument("--siren_path",     type=str, default="./data/siren/marmousi_paper_sm10_sp15.pth")
    parser.add_argument("--out_dir",        type=str, default="./data/output/fwi_method2_10_random")
    parser.add_argument("--fwi_iterations", type=int, default=10000)
    parser.add_argument("--shots_per_epoch",type=int, default=5)
    parser.add_argument("--shot_selection_policy", type=str, default="random",
                        choices=["random", "sequential", "spaced"])
    parser.add_argument("--all_shots",   action="store_true")
    parser.add_argument("--multiscale",  action="store_true")
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--debug",       action="store_true")

    # [M2] Column selection — new interface (preferred)
    parser.add_argument("--n_known_cols", type=int, default=None,
                        help="Number of known columns to use as constraint. "
                             "If set, --known_col_spacing is ignored.")
    parser.add_argument("--col_selection_mode", type=str, default="spaced",
                        choices=["spaced", "random"],
                        help="How to pick the known columns: "
                             "'spaced' = evenly distributed, "
                             "'random' = random subset (reproducible via --col_seed).")
    parser.add_argument("--col_seed", type=int, default=42,
                        help="RNG seed for random column selection (ignored in spaced mode).")

    # [M2] Column selection — legacy interface
    parser.add_argument("--known_col_spacing", type=int, default=50,
                        help="(Legacy) one known column every N columns. "
                             "Ignored when --n_known_cols is set.")

    parser.add_argument("--true_vp_path", type=str,
                        default="./data/v_models/marmousi_paper_sp15.npz",
                        help="True velocity model from which to extract known columns.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for data-misfit loss: total = alpha*data + (1-alpha)*col.")

    args = parser.parse_args()
    main(args)
