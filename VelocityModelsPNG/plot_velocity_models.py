# From .npz file to .png file

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Files to convert in .png
files = [
    "marmousi_paper_sp15.npz",
    "overthrust_paper_sp15.npz",
    "bp2004_paper_sp15.npz",
]

# Definition of input and output directories
input_dir = "./data/v_models/"
output_dir = "./data/v_models_png/"

for fname in files:
    path = os.path.join(input_dir, fname)
    data = np.load(path)

    vp = data["vp"]          # shape (X, Z)
    dx, dz = data["spacing"] # spacing in meters

    nx, nz = vp.shape

    # Axes in meters
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz

    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(
        vp.T,                       # transpose: rows = z, columns = x
        cmap="RdBu_r",               # colormap red-blue
        extent=[x[0], x[-1], z[-1], z[0]],  # z axis inverted (depth towards the bottom)
        aspect="auto",
        origin="upper",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Velocity Model - {fname.replace('_paper_sp15.npz', '')}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Velocity (m/s)")

    out_name = fname.replace(".npz", ".png")
    out_path = os.path.join(output_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato: {out_path}")