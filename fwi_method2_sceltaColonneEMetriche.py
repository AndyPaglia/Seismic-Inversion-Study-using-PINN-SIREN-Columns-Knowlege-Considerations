"""
Full Waveform Inversion (FWI) – METHOD 2: Column penalty loss
=============================================================
Stessa struttura di fwi_method2.py con l'aggiunta di uno scheduling
dinamico di alpha durante le iterazioni (curriculum learning / loss annealing).

    total_loss = alpha(t) * data_loss + (1 - alpha(t)) * col_loss

Metriche tracciate e salvate:
  - Loss vs epoch (total, data_loss raw, col_loss raw)
  - RMS error e MSE vs modello vero (ogni epochs_per_plot epoch)
  - MAE vs modello vero (ogni epochs_per_plot epoch)
  - Differenza assoluta 2D (mappa) salvata come PNG
  - Tempo totale di esecuzione
  - Epoch di convergenza (prima epoch in cui loss < soglia)
  - Tutto esportato anche in metrics.txt

Image and Sound Processing Lab - Politecnico di Milano
"""

import argparse
import os
import time

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
    t = epoch / max(total_epochs - 1, 1)
    if schedule == "fixed":
        return alpha_start
    elif schedule == "linear":
        return alpha_start + (alpha_end - alpha_start) * t
    elif schedule == "sigmoid":
        x = steepness * (t - pivot)
        sig = 1.0 / (1.0 + np.exp(-x))
        return alpha_start + (alpha_end - alpha_start) * sig
    elif schedule == "cosine":
        cos_val = (1.0 - np.cos(np.pi * t)) / 2.0
        return alpha_start + (alpha_end - alpha_start) * cos_val
    else:
        raise ValueError(f"Unknown alpha_schedule: '{schedule}'.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(vp_pred: np.ndarray, vp_true: np.ndarray):
    """
    Compute pixel-wise comparison metrics between predicted and true velocity models.

    Args:
        vp_pred : inverted model (Nz, Nx), already stripped of PML.
        vp_true : true model    (Nz, Nx).

    Returns:
        dict with keys: mse, rmse, mae, abs_diff (2D array)
    """
    diff      = vp_pred - vp_true
    abs_diff  = np.abs(diff)
    mse       = float(np.mean(diff ** 2))
    rmse      = float(np.sqrt(mse))
    mae       = float(np.mean(abs_diff))
    return dict(mse=mse, rmse=rmse, mae=mae, abs_diff=abs_diff)


def save_metrics_txt(path: str, metrics_history: list, total_time_s: float,
                     convergence_epoch: int, convergence_thresh: float,
                     args: argparse.Namespace) -> None:
    """
    Write a human-readable summary of all tracked metrics to a .txt file.
    """
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FWI Method 2 — Metrics Summary\n")
        f.write("=" * 70 + "\n\n")

        # Run configuration
        f.write("[Run configuration]\n")
        f.write(f"  alpha_schedule   : {args.alpha_schedule}\n")
        f.write(f"  alpha_start      : {args.alpha_start}\n")
        f.write(f"  alpha_end        : {args.alpha_end}\n")
        if args.alpha_schedule == "sigmoid":
            f.write(f"  alpha_pivot      : {args.alpha_pivot}\n")
            f.write(f"  alpha_steepness  : {args.alpha_steepness}\n")
        f.write(f"  n_known_cols     : {args.n_known_cols}\n")
        f.write(f"  col_selection    : {args.col_selection_mode}\n")
        f.write(f"  fwi_iterations   : {args.fwi_iterations}\n")
        f.write(f"  shots_per_epoch  : {args.shots_per_epoch}\n\n")

        # Timing
        h = int(total_time_s // 3600)
        m = int((total_time_s % 3600) // 60)
        s = total_time_s % 60
        f.write("[Timing]\n")
        f.write(f"  Total execution time : {h:02d}h {m:02d}m {s:05.2f}s\n")
        f.write(f"  Avg time per epoch   : {total_time_s / args.fwi_iterations * 1000:.2f} ms\n\n")

        # Convergence
        f.write("[Convergence]\n")
        f.write(f"  Convergence threshold : {convergence_thresh:.2e}\n")
        if convergence_epoch >= 0:
            f.write(f"  First epoch below threshold : {convergence_epoch}\n")
        else:
            f.write(f"  Never reached threshold within {args.fwi_iterations} epochs\n")
        f.write("\n")

        # Per-checkpoint metrics
        f.write("[Per-checkpoint metrics vs true model]\n")
        f.write(f"  {'Epoch':>8}  {'MSE':>12}  {'RMSE':>12}  {'MAE':>12}  "
                f"{'data_loss_raw':>14}  {'col_loss_raw':>13}  {'alpha':>7}\n")
        f.write("  " + "-" * 85 + "\n")
        for m in metrics_history:
            f.write(
                f"  {m['epoch']:>8d}  "
                f"{m['mse']:>12.6e}  "
                f"{m['rmse']:>12.6e}  "
                f"{m['mae']:>12.6e}  "
                f"{m['data_loss_raw']:>14.6e}  "
                f"{m['col_loss_raw']:>13.6e}  "
                f"{m['alpha']:>7.4f}\n"
            )
        f.write("\n")

        # Best metrics
        if metrics_history:
            best_rmse = min(metrics_history, key=lambda x: x["rmse"])
            best_mae  = min(metrics_history, key=lambda x: x["mae"])
            f.write("[Best checkpoint metrics]\n")
            f.write(f"  Best RMSE : {best_rmse['rmse']:.6e}  at epoch {best_rmse['epoch']}\n")
            f.write(f"  Best MAE  : {best_mae['mae']:.6e}   at epoch {best_mae['epoch']}\n")

    print(f"  → Metrics saved to: {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_shots(policy, epoch, n_total, n_per_epoch):
    if policy == "random":
        return np.random.choice(n_total, n_per_epoch, replace=False)
    elif policy == "sequential":
        return np.arange(epoch * n_per_epoch, (epoch + 1) * n_per_epoch) % n_total
    elif policy == "spaced":
        step = n_total // n_per_epoch
        return np.arange(epoch % step, n_total, step)
    else:
        raise ValueError(f"Unknown shot selection policy: '{policy}'.")


def _select_known_columns(n_cols_total, n_known, mode, seed=42,
                          col_range_start=0, col_range_end=None,
                          col_range_start_2=None, col_range_end_2=None):
    if col_range_end is None:
        col_range_end = n_cols_total - 1
    if col_range_start_2 is not None and col_range_end_2 is not None:
        pool = np.unique(np.concatenate([
            np.arange(col_range_start, col_range_end + 1),
            np.arange(col_range_start_2, col_range_end_2 + 1),
        ]))
    else:
        pool = np.arange(col_range_start, col_range_end + 1)
    if n_known > len(pool):
        raise ValueError(f"n_known_cols ({n_known}) > pool size ({len(pool)})")
    if mode == "spaced":
        idx = np.round(np.linspace(0, len(pool) - 1, n_known)).astype(int)
        return np.unique(pool[idx])
    elif mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(pool, size=n_known, replace=False))
    else:
        raise ValueError(f"Unknown col_selection_mode: '{mode}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    t_start = time.time()   # ← start timer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    npzfile    = np.load(args.obs_data_path, allow_pickle=True)
    d_obs_np   = npzfile["d_obs_list"]
    dt_ms      = float(npzfile["dt"])
    dt_s       = dt_ms / 1000.0
    spacing    = npzfile["spacing"]
    dh         = float(spacing[0])
    wave_np    = npzfile["wave"]
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
    diff_dir = os.path.join(out_dir, "abs_diff")   # ← cartella per mappe 2D
    os.makedirs(diff_dir, exist_ok=True)

    rec_coordinates = npzfile["rec_coordinates"]
    src_coordinates = npzfile["src_coordinates"]
    sources = (src_coordinates // spacing).astype(int) + [nbl, nbl]
    recz    = int(nbl + float(rec_coordinates[0, 1]) / float(spacing[0]))

    n_total_shots    = len(sources)
    num_mini_batches = n_total_shots // shots_per_epoch if use_all_shots else 1
    epochs_per_plot  = 1 if use_all_shots else 20

    # True model (needed for metrics)
    true_vp_np = np.load(args.true_vp_path)["vp"].T   # (Nz, Nx) physical

    # Known columns
    Nx = domain[1]
    if args.n_known_cols is not None:
        n_known = args.n_known_cols
        mode    = args.col_selection_mode
        known_col_idx = _select_known_columns(
            Nx, n_known, mode, seed=args.col_seed,
            col_range_start=args.col_range_start,
            col_range_end=args.col_range_end,
            col_range_start_2=args.col_range_start_2,
            col_range_end_2=args.col_range_end_2,
        )
    else:
        col_spacing   = args.known_col_spacing
        known_col_idx = np.arange(0, Nx, col_spacing)
        n_known       = len(known_col_idx)
        mode          = "spaced"

    known_cols_t = torch.from_numpy(
        true_vp_np[:, known_col_idx].astype(np.float32)
    ).to(dev)
    col_idx_t = torch.tensor(known_col_idx, dtype=torch.long, device=dev)
    col_scale = known_cols_t.abs().mean().clamp(min=1e-8)

    schedule    = args.alpha_schedule
    alpha_start = args.alpha_start
    alpha_end   = args.alpha_end

    print(f"\n[Method 2 — Column penalty loss + dynamic alpha + metrics]")
    print(f"Alpha schedule   : {schedule}")
    if schedule == "fixed":
        print(f"  alpha (fixed)  : {alpha_start:.2f}")
    else:
        print(f"  alpha_start    : {alpha_start:.2f}")
        print(f"  alpha_end      : {alpha_end:.2f}")
        if schedule == "sigmoid":
            print(f"  pivot          : {args.alpha_pivot:.2f} × total_epochs")
            print(f"  steepness      : {args.alpha_steepness:.1f}")
    print(f"Known cols       : {n_known} / {Nx}  (mode={mode})")
    print(f"Total shots      : {n_total_shots}")
    print(f"Shots/epoch      : {shots_per_epoch}")
    print(f"FWI iterations   : {fwi_iterations}\n")

    # Alpha schedule preview
    if args.plot:
        preview_alphas = [
            get_alpha(e, fwi_iterations, schedule, alpha_start, alpha_end,
                      args.alpha_pivot, args.alpha_steepness)
            for e in range(fwi_iterations)
        ]
        plt.figure(figsize=(8, 3))
        plt.plot(preview_alphas, lw=2, color="steelblue", label="alpha(t)")
        plt.plot([1 - a for a in preview_alphas], lw=2, color="darkorange",
                 ls="--", label="1-alpha(t)")
        plt.xlabel("Epoch"); plt.ylabel("Weight")
        plt.title(f"Alpha schedule preview: {schedule}")
        plt.legend(); plt.tight_layout(); plt.show()

    # Multi-scale
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

    obs_t  = torch.from_numpy(d_obs_np).float().to(dev)
    wave_t = torch.from_numpy(wave_np.astype(np.float32)).to(dev)
    coords = imvel.coords.to(dev)
    pmlc_t = torch.from_numpy(pmlc_np.astype(np.float32)).to(dev)

    # History arrays
    LOSS, LOSS_ALL              = [], []
    RAW_DATA_LOSS, RAW_COL_LOSS = [], []
    ALPHA_HISTORY               = []
    metrics_history             = []   # list of dicts, one per checkpoint

    best_loss, best_epoch       = float("inf"), 0
    obs_scale = obs_t.abs().max().clamp(min=1e-8)

    # Convergence detection
    conv_thresh = args.convergence_threshold
    convergence_epoch = -1   # -1 means "never reached"

    fwd_kwargs = dict(
        wave=wave_t, src_list=np.array(sources), domain=domain_pad,
        dt=dt_s, h=dh, dev=dev, recz=recz, b=pmlc_t, pmln=nbl,
    )

    # -----------------------------------------------------------------------
    # FWI loop
    # -----------------------------------------------------------------------
    for epoch in tqdm(range(fwi_iterations), desc="FWI [M2]"):
        opt.zero_grad()

        alpha = get_alpha(epoch, fwi_iterations, schedule,
                          alpha_start, alpha_end,
                          args.alpha_pivot, args.alpha_steepness)
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

            col_loss = l2loss(vp[:, col_idx_t] / col_scale,
                              known_cols_t / col_scale)

            vp_pad = torch.nn.functional.pad(
                vp.unsqueeze(0).unsqueeze(0), (nbl, nbl, nbl, nbl),
                mode="replicate",
            )[0, 0]

            syn = forward(c=vp_pad, **fwd_kwargs)
            if torch.isnan(syn).any():
                raise RuntimeError(f"NaN in synthetics at epoch {epoch}.")

            obs_batch = obs_t[src_idx]
            data_loss = l2loss(syn / obs_scale, obs_batch / obs_scale)

            if active_f_high is not None:
                syn_filt = bandpass_shots(syn.detach().cpu().numpy(),
                                          active_f_low, active_f_high, dt_s)
                obs_filt = bandpass_shots(obs_batch.detach().cpu().numpy(),
                                          active_f_low, active_f_high, dt_s)
                syn_f   = torch.from_numpy(syn_filt).to(dev)
                obs_f   = torch.from_numpy(obs_filt).to(dev)
                scale_f = obs_f.abs().max().clamp(min=1e-8)
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

        # Convergence check
        if convergence_epoch < 0 and epoch_loss < conv_thresh:
            convergence_epoch = epoch
            print(f"\n  ✓ Converged at epoch {epoch}  (loss={epoch_loss:.4e} < {conv_thresh:.4e})")

        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            if not args.debug:
                torch.save(imvel.state_dict(),
                           os.path.join(out_dir, "fwi_best_model.pth"))

        # -------------------------------------------------------------------
        # Checkpoint: metrics + plots
        # -------------------------------------------------------------------
        if epoch % epochs_per_plot == 0:
            lr_now   = opt.param_groups[0]["lr"]
            inverted = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]

            # Compute metrics vs true model
            m = compute_metrics(inverted, true_vp_np)
            m.update(epoch=epoch, data_loss_raw=epoch_data_raw,
                     col_loss_raw=epoch_col_raw, alpha=alpha,
                     total_loss=epoch_loss)
            metrics_history.append(m)

            print(
                f"  Epoch {epoch:5d} | alpha={alpha:.3f} | "
                f"Loss={epoch_loss:.4e} | "
                f"data(raw)={epoch_data_raw:.4e} | "
                f"col(raw)={epoch_col_raw:.4e} | "
                f"RMSE={m['rmse']:.4e} | MAE={m['mae']:.4e} | "
                f"LR={lr_now:.2e} | Best@{best_epoch}"
            )

            if not args.debug:
                # ── Absolute difference map ──────────────────────────────
                fig_diff, ax_diff = plt.subplots(figsize=(10, 4))
                im = ax_diff.imshow(
                    m["abs_diff"], cmap="hot_r", aspect="auto",
                    vmin=0, vmax=np.percentile(m["abs_diff"], 98),
                )
                ax_diff.set_title(
                    f"|Inverted − True| — epoch {epoch}  "
                    f"(RMSE={m['rmse']:.4e}, MAE={m['mae']:.4e})"
                )
                ax_diff.set_xlabel("x [samples]")
                ax_diff.set_ylabel("z [samples]")
                plt.colorbar(im, ax=ax_diff, label="|Δvp| [km/s]")
                plt.tight_layout()
                plt.savefig(f"{diff_dir}/diff_epoch{epoch:05d}.png",
                            dpi=150, bbox_inches="tight")
                if args.plot:
                    plt.show()
                plt.close(fig_diff)

            # ── Main 5-panel figure ──────────────────────────────────────
            fig, axes = plt.subplots(5, 1, figsize=(10, 20))

            # Panel 1: inverted model
            axes[0].imshow(inverted, cmap="RdBu_r", aspect="auto",
                           vmin=1.0, vmax=4.5)
            axes[0].set_title(
                f"Inverted Vp — epoch {epoch}  "
                f"[M2, {schedule}, alpha={alpha:.3f}, n_known={n_known}]"
            )
            axes[0].set_xlabel("x [samples]"); axes[0].set_ylabel("z [samples]")
            for ci in known_col_idx:
                axes[0].axvline(x=ci, color="k", linewidth=0.4, alpha=0.5)

            # Panel 2: total loss
            axes[1].semilogy(LOSS, color="steelblue", lw=1.5,
                             label="total loss")
            axes[1].axvline(best_epoch, color="red", ls="--", lw=1,
                            label=f"best ({best_epoch})")
            if convergence_epoch >= 0:
                axes[1].axvline(convergence_epoch, color="green", ls=":",
                                lw=1.5, label=f"converged ({convergence_epoch})")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Normalised MSE")
            axes[1].legend(); axes[1].set_title("Total loss")

            # Panel 3: raw loss terms
            axes[2].semilogy(RAW_DATA_LOSS, color="darkorange", lw=1.5,
                             label="data_loss (raw)")
            axes[2].semilogy(RAW_COL_LOSS, color="forestgreen", lw=1.5,
                             label="col_loss (raw)")
            axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Raw MSE")
            axes[2].legend()
            axes[2].set_title("Raw loss terms (pre-weighting)")

            # Panel 4: RMSE / MAE vs true model over checkpoints
            ck_epochs = [m["epoch"] for m in metrics_history]
            ck_rmse   = [m["rmse"]  for m in metrics_history]
            ck_mae    = [m["mae"]   for m in metrics_history]
            axes[3].semilogy(ck_epochs, ck_rmse, color="crimson", lw=1.5,
                             marker="o", ms=3, label="RMSE vs true")
            axes[3].semilogy(ck_epochs, ck_mae,  color="purple",  lw=1.5,
                             marker="s", ms=3, label="MAE vs true")
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("km/s")
            axes[3].legend()
            axes[3].set_title("RMSE & MAE vs true velocity model")

            # Panel 5: alpha schedule
            axes[4].plot(ALPHA_HISTORY, color="steelblue", lw=2,
                         label="alpha(t)")
            axes[4].plot([1 - a for a in ALPHA_HISTORY], color="darkorange",
                         lw=2, ls="--", label="1-alpha(t)")
            axes[4].axvline(epoch, color="gray", ls=":", lw=1)
            axes[4].set_xlabel("Epoch"); axes[4].set_ylabel("Weight")
            axes[4].set_ylim(-0.05, 1.05)
            axes[4].legend(); axes[4].set_title("Alpha schedule")

            plt.tight_layout()
            if not args.debug:
                plt.savefig(f"{png_dir}/epoch{epoch:05d}.png",
                            dpi=150, bbox_inches="tight")
            if args.plot:
                plt.show()
            plt.close(fig)

    # -----------------------------------------------------------------------
    # End of training
    # -----------------------------------------------------------------------
    total_time_s = time.time() - t_start
    h = int(total_time_s // 3600)
    m_t = int((total_time_s % 3600) // 60)
    s_t = total_time_s % 60
    print(f"\nTotal execution time: {h:02d}h {m_t:02d}m {s_t:05.2f}s")
    if convergence_epoch >= 0:
        print(f"Converged at epoch  : {convergence_epoch}")
    else:
        print(f"Did not reach convergence threshold ({conv_thresh:.2e})")

    final_vp = vp_pad.detach().cpu().numpy()[nbl:-nbl, nbl:-nbl]

    if not args.debug:
        # Save npz
        np.savez(
            os.path.join(out_dir, "fwi_results.npz"),
            vp=final_vp,
            LOSS=np.array(LOSS),
            LOSS_ALL=np.array(LOSS_ALL),
            RAW_DATA_LOSS=np.array(RAW_DATA_LOSS),
            RAW_COL_LOSS=np.array(RAW_COL_LOSS),
            ALPHA_HISTORY=np.array(ALPHA_HISTORY),
            known_col_idx=known_col_idx,
            # Metric arrays
            metric_epochs=np.array([m["epoch"] for m in metrics_history]),
            metric_mse   =np.array([m["mse"]   for m in metrics_history]),
            metric_rmse  =np.array([m["rmse"]  for m in metrics_history]),
            metric_mae   =np.array([m["mae"]   for m in metrics_history]),
            allow_pickle=True,
        )
        torch.save(imvel.state_dict(),
                   os.path.join(out_dir, "fwi_final_model.pth"))

        # Save metrics txt
        save_metrics_txt(
            path=os.path.join(out_dir, "metrics.txt"),
            metrics_history=metrics_history,
            total_time_s=total_time_s,
            convergence_epoch=convergence_epoch,
            convergence_thresh=conv_thresh,
            args=args,
        )

        # Final summary plot: RMSE + MAE over all checkpoints
        ck_epochs = [m["epoch"] for m in metrics_history]
        fig_sum, ax_sum = plt.subplots(figsize=(10, 4))
        ax_sum.semilogy(ck_epochs, [m["rmse"] for m in metrics_history],
                        color="crimson", lw=2, label="RMSE vs true")
        ax_sum.semilogy(ck_epochs, [m["mae"]  for m in metrics_history],
                        color="purple",  lw=2, ls="--", label="MAE vs true")
        ax_sum.set_xlabel("Epoch"); ax_sum.set_ylabel("km/s")
        ax_sum.set_title("Final RMSE & MAE summary")
        ax_sum.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_mae_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig_sum)

        print(f"Results saved to   : {out_dir}")
        print(f"Best model (epoch {best_epoch}, loss {best_loss:.6e}) "
              f"→ fwi_best_model.pth")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FWI Method 2 — column penalty + dynamic alpha + metrics"
    )

    parser.add_argument("--obs_data_path",  type=str,
                        default="./data/shots/marmousi_paper_sp15.npz")
    parser.add_argument("--siren_path",     type=str,
                        default="./data/siren/marmousi_paper_sm10_sp15.pth")
    parser.add_argument("--out_dir",        type=str,
                        default="./data/output/fwi_method2_alpha_regolabile_25_linear")
    parser.add_argument("--fwi_iterations", type=int, default=10000)
    parser.add_argument("--shots_per_epoch",type=int, default=5)
    parser.add_argument("--shot_selection_policy", type=str, default="random",
                        choices=["random", "sequential", "spaced"])
    parser.add_argument("--all_shots",   action="store_true")
    parser.add_argument("--multiscale",  action="store_true")
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--debug",       action="store_true")

    # Column selection
    parser.add_argument("--n_known_cols",       type=int,  default=None)
    parser.add_argument("--col_selection_mode", type=str,  default="spaced",
                        choices=["spaced", "random"])
    parser.add_argument("--col_seed",           type=int,  default=42)
    parser.add_argument("--known_col_spacing",  type=int,  default=50)
    parser.add_argument("--true_vp_path",       type=str,
                        default="./data/v_models/marmousi_paper_sp15.npz")
    parser.add_argument("--col_range_start",    type=int,  default=0)
    parser.add_argument("--col_range_end",      type=int,  default=None)
    parser.add_argument("--col_range_start_2",  type=int,  default=None)
    parser.add_argument("--col_range_end_2",    type=int,  default=None)

    # Alpha scheduling
    parser.add_argument("--alpha_schedule",    type=str,   default="fixed",
                        choices=["fixed", "linear", "sigmoid", "cosine"])
    parser.add_argument("--alpha_start",       type=float, default=0.5)
    parser.add_argument("--alpha_end",         type=float, default=0.9)
    parser.add_argument("--alpha_pivot",       type=float, default=0.2)
    parser.add_argument("--alpha_steepness",   type=float, default=10.0)

    # Convergence threshold for the total (weighted) loss
    parser.add_argument("--convergence_threshold", type=float, default=1e-4,
                        help="Loss value below which training is considered converged. "
                             "Used only for logging — does NOT stop training early.")

    args = parser.parse_args()
    main(args)
