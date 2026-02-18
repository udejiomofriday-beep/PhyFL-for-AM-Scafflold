"""
TPMS Dataset Generator
Generates voxelized TPMS datasets and metadata for research use.

Example:
    python tpms_generator_release.py --samples 1200

Outputs:
    tpms_dataset/
        voxels/
        stls/
        master_registry.csv
"""

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="TPMS dataset generator")
    parser.add_argument("--samples", type=int, default=1200, help="Number of samples to generate")
    args = parser.parse_args()

    # Override total_samples if CONFIG exists
    if "CONFIG" in globals():
        CONFIG["total_samples"] = args.samples

    # Ensure portable working directory
    os.makedirs("tpms_dataset", exist_ok=True)



#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         TPMS Scaffold Dataset Generator — CSME 2026                        ║
║  Privacy-Preserving Federated Learning for Biomedical Scaffold Optimization ║
║  Author : Friday Udeji, Dept. Mechanical Engineering, University of Manitoba ║
║  Usage  : python tpms_generator.py [--dry-run] [--n-workers N]              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generates 1,200 TPMS scaffold samples across 5 federation nodes with
engineered Non-IID bias, producing:
  • STL meshes  (128³ marching cubes, coordinates in mm)
  • Voxel arrays (64³ uint8 NumPy .npy, 0 = solid, 1 = void)
  • master_registry.csv
  • README.md
  • sample_preview.png

Dependencies (all open-source):
    numpy scipy scikit-image trimesh matplotlib
    Install: pip install numpy scipy scikit-image trimesh matplotlib
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import multiprocessing as mp

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive; safe for headless compute
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d
from skimage.measure import marching_cubes
# ─────────────────────────────── Configuration ────────────────────────────────
CONFIG: Dict = {
    # Dataset dimensions
    "n_samples": 1200,
    "n_nodes": 5,
    "n_cells": 3,          # 3 × 3 × 3 unit-cell domain per scaffold
    # Voxel / mesh resolution
    "voxel_res": 64,        # 64³ binary array → CNN input
    "stl_res": 128,         # 128³ marching cubes → FEA / COMSOL import
    "lookup_res": 64,       # resolution for C↔porosity lookup table
    # Design parameter ranges
    "porosity_min": 0.40,
    "porosity_max": 0.80,
    "cell_size_min": 1.0,  # mm
    "cell_size_max": 3.0,  # mm
    # Grid resolution over parameter space
    "n_porosity_steps": 20,  # 20 × 12 × 5 = 1,200 exactly
    "n_cellsize_steps": 12,
    # I/O
    "output_dir": "tpms_dataset",
    "random_seed": 42,
    "n_workers": max(1, mp.cpu_count() - 1),
}

# ── Five topologically distinct TPMS families ─────────────────────────────────
TPMS_TYPES: List[str] = [
    "Schwarz_P",
    "Gyroid",
    "Schwarz_D",
    "IWP",
    "Neovius",
]

# Biological / clinical node identities (for README and metadata)
NODE_PROFILES: Dict[int, str] = {
    1: "Gyroid-heavy (~55 % Gyroid). Models inter-institution topology bias.",
    2: "High-porosity focus (φ = 0.60–0.80). Simulates osteoporotic-bone institutions.",
    3: "Uniform mixed baseline — closest to IID reference node.",
    4: "Small cell-size focus (cs = 1.0–1.8 mm). Mimics pediatric scaffold centres.",
    5: "Schwarz_D + IWP heavy. Represents post-oncology/radiation cohort institutions.",
}

# ─────────────────────────── TPMS Implicit Functions ──────────────────────────


def tpms_function(
    tpms_type: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    k: float = 2 * np.pi,
) -> np.ndarray:
    """
    Evaluate the TPMS level-set function f(x, y, z) on a meshgrid.

    The scaffold solid region is defined as { (x,y,z) : f(x,y,z) ≥ C }.
    The void (pore) region is { f < C }.  Porosity = Vol(f < C) / Vol(Ω).

    Wavenumber k = 2π / cell_size_mm reproduces one full period per unit cell.

    Formulae (x̃ = k·X, ỹ = k·Y, z̃ = k·Z):
      Schwarz_P  : cos(x̃) + cos(ỹ) + cos(z̃)
      Gyroid     : sin(x̃)cos(ỹ) + sin(ỹ)cos(z̃) + sin(z̃)cos(x̃)
      Schwarz_D  : sin(x̃)sin(ỹ)sin(z̃) + sin(x̃)cos(ỹ)cos(z̃)
                   + cos(x̃)sin(ỹ)cos(z̃) + cos(x̃)cos(ỹ)sin(z̃)
      IWP        : 2[cos(x̃)cos(ỹ) + cos(ỹ)cos(z̃) + cos(z̃)cos(x̃)]
                   − [cos(2x̃) + cos(2ỹ) + cos(2z̃)]
      Neovius    : 3[cos(x̃) + cos(ỹ) + cos(z̃)] + 4·cos(x̃)·cos(ỹ)·cos(z̃)

    References:
      [1] Schwarz (1890); [2] Schoen (1970); [3] Neovius (1883)
      [4] Maskery et al., Addit. Manuf. 16, 2017.
    """
    x = k * X
    y = k * Y
    z = k * Z

    if tpms_type == "Schwarz_P":
        return np.cos(x) + np.cos(y) + np.cos(z)

    elif tpms_type == "Gyroid":
        return (
            np.sin(x) * np.cos(y)
            + np.sin(y) * np.cos(z)
            + np.sin(z) * np.cos(x)
        )

    elif tpms_type == "Schwarz_D":
        return (
            np.sin(x) * np.sin(y) * np.sin(z)
            + np.sin(x) * np.cos(y) * np.cos(z)
            + np.cos(x) * np.sin(y) * np.cos(z)
            + np.cos(x) * np.cos(y) * np.sin(z)
        )

    elif tpms_type == "IWP":
        return 2.0 * (
            np.cos(x) * np.cos(y)
            + np.cos(y) * np.cos(z)
            + np.cos(z) * np.cos(x)
        ) - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))

    elif tpms_type == "Neovius":
        return (
            3.0 * (np.cos(x) + np.cos(y) + np.cos(z))
            + 4.0 * np.cos(x) * np.cos(y) * np.cos(z)
        )

    else:
        raise ValueError(f"Unknown TPMS type: '{tpms_type}'")

# ─────────────────────── Porosity ↔ Iso-Level Lookup ──────────────────────────


def build_all_lookups(cfg: Dict) -> Dict[str, np.ndarray]:
    """
    Pre-compute sorted F-value distributions for each TPMS type on a single
    unit cell.  Because the function is periodic, the CDF of F values over
    one period equals the CDF over any integer number of periods.

    For a target porosity φ:
        C = np.percentile(F_sorted, φ × 100)
    equivalently:
        C = F_sorted[int(φ × N)]

    Returns a dict { tpms_type → F_sorted (1D array) }.
    """
    res = cfg["lookup_res"]
    # Evaluate over exactly one unit cell in normalised coords [0,1)
    coords = np.linspace(0, 1.0, res, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    lookups: Dict[str, np.ndarray] = {}
    for ttype in TPMS_TYPES:
        F = tpms_function(ttype, X, Y, Z, k=2 * np.pi)
        lookups[ttype] = np.sort(F.ravel())  # ascending sorted F values

    return lookups


def porosity_to_C(ttype: str, target_porosity: float, lookups: Dict) -> float:
    """
    Map a target void fraction φ ∈ [0,1] to the corresponding iso-level C
    using the pre-computed F-value CDF.

    Void region : { f < C }  →  P(F < C) = φ  →  C = quantile(F, φ)
    """
    F_sorted = lookups[ttype]
    idx = int(np.clip(target_porosity, 0.0, 1.0) * len(F_sorted))
    idx = min(idx, len(F_sorted) - 1)
    return float(F_sorted[idx])

# ──────────────────────────── Volume Generation ───────────────────────────────


def generate_volume(
    tpms_type: str,
    cell_size: float,
    C: float,
    res: int,
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate the TPMS function on an (n_cells · cell_size)³ domain at `res`³
    resolution.

    Returns
    -------
    F              : (res, res, res) float64 — raw level-set values
    voxel_binary   : (res, res, res) bool    — True = void (f < C)
    actual_porosity: float — measured void fraction from voxel array
    """
    domain = n_cells * cell_size                      # total edge length (mm)
    coords = np.linspace(0.0, domain, res, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    k = 2.0 * np.pi / cell_size                       # one period per unit cell
    F = tpms_function(tpms_type, X, Y, Z, k=k)
    voxel_binary = F < C                              # void where f < C
    actual_porosity = float(voxel_binary.mean())
    return F, voxel_binary, actual_porosity

# ───────────────────────────── STL Generation ─────────────────────────────────


def _write_binary_stl(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> None:
    """
    Write a binary STL file without external dependencies.

    Binary STL layout (IEEE 754, little-endian):
        80 bytes  — header (ASCII)
        4  bytes  — uint32 triangle count
        per triangle (50 bytes each):
          12 bytes — normal float32 × 3
          12 bytes — vertex 1 float32 × 3
          12 bytes — vertex 2 float32 × 3
          12 bytes — vertex 3 float32 × 3
          2  bytes — attribute byte count (= 0)
    """
    n_tri = len(faces)
    tri_verts = verts[faces]                              # (N,3,3) float64

    # Per-face normals: if marching_cubes vertex normals unavailable, compute
    if normals is not None and normals.shape[0] == verts.shape[0]:
        face_normals = normals[faces].mean(axis=1)
    else:
        v0 = tri_verts[:, 0, :]
        v1 = tri_verts[:, 1, :]
        v2 = tri_verts[:, 2, :]
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = np.cross(e1, e2)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        face_normals /= norms

    header = b"TPMS Scaffold Dataset CSME 2026 Friday Udeji U of Manitoba" + b" " * 22
    header = header[:80]

    # Pack into structured array: normal(3f) + v0(3f) + v1(3f) + v2(3f) + attr(H)
    dt = np.dtype([
        ("normal", np.float32, (3,)),
        ("v0",     np.float32, (3,)),
        ("v1",     np.float32, (3,)),
        ("v2",     np.float32, (3,)),
        ("attr",   np.uint16),
    ])
    data = np.zeros(n_tri, dtype=dt)
    data["normal"] = face_normals.astype(np.float32)
    data["v0"]     = tri_verts[:, 0, :].astype(np.float32)
    data["v1"]     = tri_verts[:, 1, :].astype(np.float32)
    data["v2"]     = tri_verts[:, 2, :].astype(np.float32)
    data["attr"]   = 0

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(np.uint32(n_tri).tobytes())
        fh.write(data.tobytes())


def generate_stl(
    F: np.ndarray,
    C: float,
    voxel_size_mm: float,
    stl_path: Path,
) -> bool:
    """
    Run marching cubes on F at iso-level C, convert voxel coordinates to mm,
    and write a binary STL using a pure NumPy writer (no trimesh required).

    Voxel-to-mm scaling:
        vertex_mm = vertex_vox × voxel_size_mm

    Returns True on success, False if marching cubes fails (no surface at C).
    """
    try:
        verts, faces, normals, _ = marching_cubes(F, level=C, allow_degenerate=False)
        verts_mm = verts * voxel_size_mm
        _write_binary_stl(stl_path, verts_mm, faces, normals)
        return True
    except (ValueError, RuntimeError):
        return False

# ──────────────────── Non-IID Federation Assignment ───────────────────────────


def assign_node_noniid(
    tpms_type: str,
    target_porosity: float,
    cell_size: float,
    rng: np.random.Generator,
) -> int:
    """
    Probabilistic node assignment encoding clinical/institutional heterogeneity.

    Decision tree (stochastic, intentionally non-deterministic to produce
    realistic overlap between nodes):

        Gyroid + roll < 0.55          → Node 1  (topology-biased)
        φ ≥ 0.60 + roll < 0.60       → Node 2  (high-porosity / osteoporotic)
        cs ≤ 1.8 + roll < 0.55       → Node 4  (small cell / pediatric)
        type ∈ {D, IWP} + roll < 0.55 → Node 5  (oncology cohort)
        otherwise                     → Node 3  (IID baseline)

    Returns node_id ∈ {1, 2, 3, 4, 5}.
    """
    roll = rng.random()

    if tpms_type == "Gyroid" and roll < 0.55:
        return 1

    if target_porosity >= 0.60 and roll < 0.60:
        return 2

    if cell_size <= 1.8 and roll < 0.55:
        return 4

    if tpms_type in ("Schwarz_D", "IWP") and roll < 0.55:
        return 5

    return 3  # uniform catch-all

# ──────────────────────── Parameter Grid Builder ──────────────────────────────


def build_parameter_grid(cfg: Dict, rng: np.random.Generator) -> List[Dict]:
    """
    Construct exactly 1,200 sample descriptors on a Cartesian grid:
        5 types × 20 porosity steps × 12 cell-size steps = 1,200

    Each descriptor is a plain dict that is pickleable for multiprocessing.
    """
    porosity_vals = np.linspace(
        cfg["porosity_min"], cfg["porosity_max"], cfg["n_porosity_steps"]
    )
    cell_vals = np.linspace(
        cfg["cell_size_min"], cfg["cell_size_max"], cfg["n_cellsize_steps"]
    )

    records: List[Dict] = []
    sample_id = 0

    for ttype in TPMS_TYPES:             # 5
        for phi in porosity_vals:         # 20
            for cs in cell_vals:          # 12
                node = assign_node_noniid(ttype, float(phi), float(cs), rng)
                records.append(
                    {
                        "sample_id": f"TPMS_{sample_id:04d}",
                        "tpms_type": ttype,
                        "target_porosity": round(float(phi), 4),
                        "unit_cell_size": round(float(cs), 4),
                        "node_id": node,
                    }
                )
                sample_id += 1

    assert len(records) == cfg["n_samples"], (
        f"Grid produced {len(records)} samples, expected {cfg['n_samples']}"
    )
    return records

# ───────────────────── Multiprocessing Worker State ───────────────────────────

# Module-level cache initialised in each worker process
_WORKER_LOOKUPS: Dict[str, np.ndarray] = {}
_WORKER_CFG: Dict = {}


def _worker_init(lookups: Dict, cfg: Dict) -> None:
    """Pool initialiser — runs once per worker process."""
    global _WORKER_LOOKUPS, _WORKER_CFG
    _WORKER_LOOKUPS = lookups
    _WORKER_CFG = cfg


# ─────────────────────────── Per-Sample Worker ────────────────────────────────


def process_sample(rec: Dict) -> Dict:
    """
    Worker function: generate one TPMS sample.

    Steps
    -----
    1. Resolve iso-level C from target porosity via lookup table.
    2. Generate 64³ voxel volume → save .npy.
    3. Generate 128³ volume → run marching cubes → save .stl.
    4. Return metadata dict for master_registry.csv.

    The STL vertex coordinates are in physical units (mm), matching
    COMSOL Multiphysics import conventions.
    The voxel array is uint8 { 0=solid, 1=void } for direct CNN ingestion.
    """
    cfg = _WORKER_CFG
    lookups = _WORKER_LOOKUPS

    sid = rec["sample_id"]
    ttype = rec["tpms_type"]
    target_phi = rec["target_porosity"]
    cs = rec["unit_cell_size"]
    node_id = rec["node_id"]
    n_cells = cfg["n_cells"]

    out_dir = Path(cfg["output_dir"])
    stl_path = out_dir / f"node_{node_id}" / "stl" / f"{sid}.stl"
    vox_path = out_dir / f"node_{node_id}" / "voxels" / f"{sid}.npy"

    # ── Step 1: iso-level ────────────────────────────────────────────────────
    C = porosity_to_C(ttype, target_phi, lookups)

    # ── Step 2: 64³ voxel ───────────────────────────────────────────────────
    vox_res = cfg["voxel_res"]
    _, voxel_binary, actual_phi = generate_volume(ttype, cs, C, vox_res, n_cells)
    # Save as uint8: 0 = solid strut, 1 = void pore channel
    np.save(str(vox_path), voxel_binary.astype(np.uint8))

    # ── Step 3: 128³ STL ────────────────────────────────────────────────────
    stl_res = cfg["stl_res"]
    voxel_size_mm = (n_cells * cs) / stl_res          # mm per voxel in STL grid
    F_stl, _, _ = generate_volume(ttype, cs, C, stl_res, n_cells)
    stl_ok = generate_stl(F_stl, C, voxel_size_mm, stl_path)

    return {
        "sample_id": sid,
        "node_id": node_id,
        "tpms_type": ttype,
        "target_porosity": target_phi,
        "actual_porosity": round(actual_phi, 4),
        "unit_cell_size": cs,
        "iso_level_C": round(C, 5),
        "stl_path": (
            str(stl_path.relative_to(out_dir)) if stl_ok else "GENERATION_FAILED"
        ),
        "voxel_path": str(vox_path.relative_to(out_dir)),
    }

# ─────────────────────────── Visualisation ───────────────────────────────────


def visualize_samples(
    registry_rows: List[Dict],
    cfg: Dict,
    n_show: int = 3,
) -> None:
    """
    Render the first `n_show` samples as 3-D surface plots from their 64³
    voxel arrays (marching cubes at level 0.5) and save a single PNG.

    Requires matplotlib with mpl_toolkits installed (standard).
    """
    out_dir = Path(cfg["output_dir"])
    fig = plt.figure(figsize=(5 * n_show, 5))
    shown = 0

    for row in registry_rows:
        if shown >= n_show:
            break
        vox_file = out_dir / row["voxel_path"]
        if not vox_file.exists():
            continue

        vol = np.load(str(vox_file)).astype(float)
        try:
            # The voxel array is void=1, solid=0.
            # Marching cubes at 0.5 traces the solid–void interface.
            verts, faces, _, _ = marching_cubes(vol, level=0.5)
        except (ValueError, RuntimeError):
            continue

        ax = fig.add_subplot(1, n_show, shown + 1, projection="3d")
        poly = Poly3DCollection(
            verts[faces],
            alpha=0.35,
            facecolor="#3a86ff",
            edgecolor="none",
        )
        ax.add_collection3d(poly)
        lim = vol.shape[0]
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_zlim(0, lim)
        ax.set_title(
            f"{row['tpms_type'].replace('_',' ')}\n"
            f"φ = {row['actual_porosity']:.2f}  |  "
            f"cs = {row['unit_cell_size']} mm",
            fontsize=9,
        )
        ax.set_axis_off()
        shown += 1

    fig.suptitle(
        "TPMS Scaffold Dataset — Sample Preview (CSME 2026)", fontsize=11
    )
    plt.tight_layout()
    save_path = out_dir / "sample_preview.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ]  Preview saved → {save_path}")

# ───────────────────────────── README Writer ──────────────────────────────────


def write_readme(
    cfg: Dict,
    registry_rows: List[Dict],
    out_dir: Path,
) -> None:
    """Generate a self-contained README.md describing dataset structure,
    TPMS formulae, node profiles, and usage examples."""

    node_counts = Counter(r["node_id"] for r in registry_rows)
    type_counts = Counter(r["tpms_type"] for r in registry_rows)

    node_table = "\n".join(
        f"| {k} | {node_counts.get(k, 0):>7} | {NODE_PROFILES[k]} |"
        for k in sorted(NODE_PROFILES)
    )

    type_table = "\n".join(
        f"| {t} | {type_counts.get(t, 0)} |" for t in TPMS_TYPES
    )

    readme_text = textwrap.dedent(
        f"""
        # TPMS Scaffold Dataset — CSME 2026
        ## Privacy-Preserving Federated Learning for Biomedical Scaffold Optimization

        **Generated :** {time.strftime('%Y-%m-%d %H:%M UTC')}  
        **Total samples :** {len(registry_rows)}  
        **Federation nodes :** {cfg['n_nodes']}  
        **Author :** Friday Udeji · Dept. Mechanical Engineering · University of Manitoba

        ---

        ## 1. Directory Structure

        ```
        tpms_dataset/
        ├── node_1/ ... node_5/        # 5 non-IID federation partitions
        │   ├── stl/                   # STL meshes ({cfg['stl_res']}³ MC, units mm)
        │   └── voxels/                # uint8 NumPy arrays, shape (64,64,64)
        ├── master_registry.csv        # Full metadata table (9 columns)
        ├── sample_preview.png         # Rendered surface of first 3 samples
        └── README.md                  # This file
        ```

        ---

        ## 2. TPMS Surface Families

        Each scaffold is generated from the level-set equation f(x,y,z) = C,
        where x̃ = (2π / cell_size)·X, and analogously for ỹ, z̃.

        | Family | Implicit Function |
        |--------|------------------|
        | Schwarz P  | cos(x̃) + cos(ỹ) + cos(z̃) |
        | Gyroid     | sin(x̃)cos(ỹ) + sin(ỹ)cos(z̃) + sin(z̃)cos(x̃) |
        | Schwarz D  | sin(x̃)sin(ỹ)sin(z̃) + sin(x̃)cos(ỹ)cos(z̃) + cos(x̃)sin(ỹ)cos(z̃) + cos(x̃)cos(ỹ)sin(z̃) |
        | IWP        | 2[cos(x̃)cos(ỹ)+cos(ỹ)cos(z̃)+cos(z̃)cos(x̃)] − [cos(2x̃)+cos(2ỹ)+cos(2z̃)] |
        | Neovius    | 3[cos(x̃)+cos(ỹ)+cos(z̃)] + 4·cos(x̃)·cos(ỹ)·cos(z̃) |

        **Solid region:** f(x,y,z) ≥ C · **Void (pore) region:** f(x,y,z) < C
        Domain: {cfg['n_cells']}×{cfg['n_cells']}×{cfg['n_cells']} unit cells per scaffold.

        ---

        ## 3. Parametric Ranges

        | Parameter          | Range          | Steps |
        |--------------------|----------------|-------|
        | Target porosity φ  | 40 % – 80 %    | {cfg['n_porosity_steps']}    |
        | Unit cell size     | 1.0 – 3.0 mm   | {cfg['n_cellsize_steps']}    |
        | TPMS families      | 5              | —     |
        | **Total samples**  | **{cfg['n_samples']}**   | —     |

        Topology breakdown:
        | Type | Samples |
        |------|---------|
        {type_table}

        ---

        ## 4. Federation Node Profiles (Non-IID)

        The five nodes simulate clinically realistic data heterogeneity.
        See [Hsieh et al., 2020] for a formal treatment of non-IID FL.

        | Node | Samples | Institutional Bias |
        |------|---------|--------------------|
        {node_table}

        ---

        ## 5. CSV Schema (`master_registry.csv`)

        | Column             | Type    | Description |
        |--------------------|---------|-------------|
        | `sample_id`        | string  | Unique ID (TPMS_XXXX) |
        | `node_id`          | int     | Federation node (1–5) |
        | `tpms_type`        | string  | Surface family |
        | `target_porosity`  | float   | Design-intent void fraction |
        | `actual_porosity`  | float   | Computed from 64³ voxels |
        | `unit_cell_size`   | float   | Repeating cell edge length (mm) |
        | `iso_level_C`      | float   | Marching-cubes iso-level used |
        | `stl_path`         | string  | Relative path to .stl |
        | `voxel_path`       | string  | Relative path to .npy |

        ---

        ## 6. Usage Examples

        ### Load voxels for deep learning
        ```python
        import numpy as np
        import pandas as pd

        df = pd.read_csv("tpms_dataset/master_registry.csv")
        node1 = df[df.node_id == 1]

        # Load one 64³ voxel volume
        vol = np.load("tpms_dataset/" + node1.iloc[0]["voxel_path"])
        # vol.shape = (64, 64, 64)  dtype = uint8  (0=solid, 1=void)
        print("Measured porosity:", vol.mean())
        ```

        ### Load STL for COMSOL FEA import
        ```python
        # Verify STL file (pure numpy — no trimesh required)
        import struct, numpy as np
        with open("tpms_dataset/node_1/stl/TPMS_0000.stl","rb") as f:
            header = f.read(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
        print(f"Triangles: {{n_tri}}")  # typically 50k–300k for 128³
        # For full FEA import use COMSOL's File > Import > STL/CAD
        ```

        ### Quick data-distribution check
        ```python
        import pandas as pd
        df = pd.read_csv("tpms_dataset/master_registry.csv")
        print(df.groupby("node_id")[["actual_porosity","unit_cell_size"]].describe())
        ```

        ---

        ## 7. Citation

        > Udeji F. (2026). "Privacy-Preserving Federated Learning for
        > Quality-Driven Optimization of Biomedical Scaffolds in Additive
        > Manufacturing." *Proc. CSME International Congress 2026*, Vancouver BC.

        ---

        ## 8. Reproducibility

        ```bash
        python tpms_generator.py   # random_seed={cfg['random_seed']} baked in
        ```
        All stochastic elements use `numpy.random.default_rng(seed={cfg['random_seed']})`.
        """
    ).strip()

    (out_dir / "README.md").write_text(readme_text, encoding="utf-8")
    print("[README] Written.")


# ────────────────────────── Dataset Statistics ────────────────────────────────


def print_dataset_stats(records: List[Dict], registry: List[Dict]) -> None:
    """Print a concise distribution table to stdout."""
    print("\n" + "=" * 60)
    print("FEDERATION NODE DISTRIBUTION")
    print("=" * 60)
    node_counts = Counter(r["node_id"] for r in records)
    type_by_node: Dict[int, Counter] = {n: Counter() for n in range(1, 6)}
    for r in records:
        type_by_node[r["node_id"]][r["tpms_type"]] += 1

    print(f"{'Node':<6} {'Total':<7} {'Dominant type':<16} {'Samples'}")
    print("-" * 55)
    for node in sorted(node_counts):
        dom_type = type_by_node[node].most_common(1)[0]
        print(
            f"  {node:<4} {node_counts[node]:<7} {dom_type[0]:<16} {dom_type[1]}"
        )
    print("=" * 60)

    if registry:
        actual_p = [r["actual_porosity"] for r in registry]
        print(
            f"\nPortosity: mean={np.mean(actual_p):.3f}  "
            f"std={np.std(actual_p):.3f}  "
            f"range=[{min(actual_p):.3f}, {max(actual_p):.3f}]"
        )
        failed = sum(1 for r in registry if "FAILED" in str(r["stl_path"]))
        print(f"STL failures: {failed} / {len(registry)}")

"""
Safe Cell 12 for Windows + Jupyter.
Paste this as a replacement for Cell 12 in tpms_generator.ipynb.
"""

import time, csv
import multiprocessing as mp
from pathlib import Path

# ── User settings ─────────────────────────────────────────────────────────────
DRY_RUN    = False          # flip to False to generate all files
OUTPUT_DIR = "tpms_dataset"
N_WORKERS  = max(1, mp.cpu_count() - 1)   # or set manually, e.g. 8
SAFE_MODE  = True           # True = single-threaded (100% reliable on Windows/Jupyter)
                             # False = multiprocessing Pool (faster but can stall)
# ─────────────────────────────────────────────────────────────────────────────

# Required for Windows multiprocessing — must be called before Pool
mp.freeze_support()

cfg = dict(CONFIG)
cfg.update({"output_dir": OUTPUT_DIR, "n_workers": N_WORKERS, "random_seed": 42})
t0      = time.time()
rng     = np.random.default_rng(cfg["random_seed"])
out_dir = Path(cfg["output_dir"])

print("=" * 60)
print("  TPMS Scaffold Dataset Generator — CSME 2026")
print("=" * 60)
for k in ("n_samples", "voxel_res", "stl_res", "n_workers"):
    print(f"  {k:<14}: {cfg[k]}")
print(f"  output_dir    : {out_dir.resolve()}")
print(f"  safe_mode     : {SAFE_MODE}")
print("=" * 60)

# 1 — Lookup tables
print("\n[1/5] Building porosity-to-iso-level lookup tables...")
lookups = build_all_lookups(cfg)
for t in TPMS_TYPES:
    print(f"      {t:<12}  C@0.40={porosity_to_C(t,0.40,lookups):+.3f}  "
          f"C@0.80={porosity_to_C(t,0.80,lookups):+.3f}")

# 2 — Parameter grid
print("\n[2/5] Building parameter grid (Non-IID split)...")
records = build_parameter_grid(cfg, rng)
print_dataset_stats(records, [])

if DRY_RUN:
    print("\n[DRY-RUN] Done. Set DRY_RUN = False to generate files.")

else:
    # 3 — Directories
    print("\n[3/5] Creating directory structure...")
    for node in range(1, cfg["n_nodes"] + 1):
        (out_dir / f"node_{node}" / "stl").mkdir(parents=True, exist_ok=True)
        (out_dir / f"node_{node}" / "voxels").mkdir(parents=True, exist_ok=True)

    # 4 — Generation (safe single-thread OR multiprocessing)
    print(f"\n[4/5] Generating {cfg['n_samples']} scaffolds "
          f"({'single-threaded' if SAFE_MODE else str(cfg['n_workers']) + ' workers'})...")

    # Initialise worker state in THIS process (needed for single-thread mode)
    _worker_init(lookups, cfg)

    registry  = []
    n_done    = 0
    last_mark = 0

    if SAFE_MODE:
        # ── Single-threaded: 100% reliable on Windows/Jupyter ────────────────
        for rec in records:
            row = process_sample(rec)
            registry.append(row)
            n_done += 1
            if n_done - last_mark >= 50:           # print every 50 samples
                elapsed = time.time() - t0
                rate    = n_done / elapsed
                eta     = (cfg["n_samples"] - n_done) / max(rate, 1e-6)
                print(f"    {n_done:>4}/{cfg['n_samples']}  "
                      f"elapsed {elapsed:>5.0f}s  "
                      f"rate {rate:.1f}/s  "
                      f"ETA {eta:>5.0f}s")
                last_mark = n_done

    else:
        # ── Multiprocessing Pool (faster but may stall on Windows/Jupyter) ───
        with mp.Pool(processes=cfg["n_workers"],
                     initializer=_worker_init,
                     initargs=(lookups, cfg)) as pool:
            for row in pool.imap_unordered(process_sample, records, chunksize=8):
                registry.append(row)
                n_done += 1
                if n_done - last_mark >= 100:
                    elapsed = time.time() - t0
                    rate    = n_done / elapsed
                    eta     = (cfg["n_samples"] - n_done) / max(rate, 1e-6)
                    print(f"    {n_done:>4}/{cfg['n_samples']}  "
                          f"elapsed {elapsed:>5.0f}s  "
                          f"rate {rate:.1f}/s  "
                          f"ETA {eta:>5.0f}s")
                    last_mark = n_done

    # 5 — Write outputs
    print("\n[5/5] Writing registry, README, visualisation...")
    fieldnames = ["sample_id", "node_id", "tpms_type", "target_porosity",
                  "actual_porosity", "unit_cell_size", "iso_level_C",
                  "stl_path", "voxel_path"]
    registry_sorted = sorted(registry, key=lambda r: r["sample_id"])
    with open(out_dir / "master_registry.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(registry_sorted)
    print(f"       master_registry.csv ({len(registry_sorted)} rows) written.")

    write_readme(cfg, registry_sorted, out_dir)
    visualize_samples(registry_sorted, cfg, n_show=3)
    print_dataset_stats(records, registry_sorted)
    print(f"\nDone in {time.time() - t0:.1f}s  |  {out_dir.resolve()}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

POROSITY  = 0.65
CELL_SIZE = 2.0
RES       = 48      # lower = faster render; raise to 96 for publication quality

_lk = build_all_lookups(CONFIG)
fig = plt.figure(figsize=(16, 4))

for i, ttype in enumerate(TPMS_TYPES):
    C = porosity_to_C(ttype, POROSITY, _lk)
    _, vox, phi = generate_volume(ttype, CELL_SIZE, C, RES, 3)
    try:
        verts, faces, _, _ = marching_cubes(vox.astype(float), level=0.5)
    except ValueError:
        continue
    ax = fig.add_subplot(1, 5, i + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        verts[faces], alpha=0.40, facecolor="#3a86ff", edgecolor="none"))
    ax.set(xlim=(0, RES), ylim=(0, RES), zlim=(0, RES))
    ax.set_title(f"{ttype.replace('_', ' ')}\n" + r"$\varphi$=" + f"{phi:.2f}",
                 fontsize=9)
    ax.set_axis_off()

fig.suptitle(
    f"TPMS Topology Comparison  |  target porosity={POROSITY}  |  "
    f"cell size={CELL_SIZE} mm",
    fontsize=11,
)
plt.tight_layout()
plt.show()

%matplotlib inline
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import pandas as pd
from pathlib import Path

# ── Pull real values from your generated dataset ──────────────────────────────
registry_path = Path(OUTPUT_DIR) / "master_registry.csv"
df = pd.read_csv(registry_path)

samples = (
    df.groupby("tpms_type", group_keys=False)
    .apply(lambda g: g.iloc[(g["actual_porosity"] - g["actual_porosity"].median()).abs().argsort()].iloc[0])
    .reset_index(drop=True)
)

print("Samples selected for preview:")
print(samples[["sample_id", "tpms_type", "actual_porosity", "unit_cell_size"]].to_string(index=False))
print()

# ── Render ────────────────────────────────────────────────────────────────────
RES = 48

_lk = build_all_lookups(CONFIG)
fig = plt.figure(figsize=(16, 4))

for i, row in samples.iterrows():
    ttype = row["tpms_type"]
    phi   = row["actual_porosity"]
    cs    = row["unit_cell_size"]
    C     = porosity_to_C(ttype, phi, _lk)
    _, vox, measured_phi = generate_volume(ttype, cs, C, RES, CONFIG["n_cells"])
    try:
        verts, faces, _, _ = marching_cubes(vox.astype(float), level=0.5)
    except ValueError:
        continue
    ax = fig.add_subplot(1, len(samples), i + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        verts[faces], alpha=0.40, facecolor="#3a86ff", edgecolor="none"))
    ax.set(xlim=(0, RES), ylim=(0, RES), zlim=(0, RES))
    ax.set_title(f"{ttype.replace('_', ' ')}\nφ={measured_phi:.2f}  cs={cs:.1f}mm",
                 fontsize=9)
    ax.set_axis_off()

fig.suptitle("TPMS Topology Comparison — Representative Samples from Dataset",
             fontsize=11)
plt.tight_layout()
display(fig)
plt.close(fig)



if __name__ == "__main__":
    main()
