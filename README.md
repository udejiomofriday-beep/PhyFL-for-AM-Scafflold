# TPMS Scaffold Dataset — CSME 2026
        ## Privacy-Preserving Federated Learning for Biomedical Scaffold Optimization

        **Generated :** 2026-02-11 22:28 UTC  
        **Total samples :** 1200  
        **Federation nodes :** 5  
        **Author :** Friday Udeji · Dept. Mechanical Engineering · University of Manitoba

        ---

        ## 1. Directory Structure

        ```
        tpms_dataset/
        ├── node_1/ ... node_5/        # 5 non-IID federation partitions
        │   ├── stl/                   # STL meshes (128³ MC, units mm)
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
        Domain: 3×3×3 unit cells per scaffold.

        ---

        ## 3. Parametric Ranges

        | Parameter          | Range          | Steps |
        |--------------------|----------------|-------|
        | Target porosity φ  | 40 % – 80 %    | 20    |
        | Unit cell size     | 1.0 – 3.0 mm   | 12    |
        | TPMS families      | 5              | —     |
        | **Total samples**  | **1200**   | —     |

        Topology breakdown:
        | Type | Samples |
        |------|---------|
        | Schwarz_P | 240 |
| Gyroid | 240 |
| Schwarz_D | 240 |
| IWP | 240 |
| Neovius | 240 |

        ---

        ## 4. Federation Node Profiles (Non-IID)

        The five nodes simulate clinically realistic data heterogeneity.
        See [Hsieh et al., 2020] for a formal treatment of non-IID FL.

        | Node | Samples | Institutional Bias |
        |------|---------|--------------------|
        | 1 |     126 | Gyroid-heavy (~55 % Gyroid). Models inter-institution topology bias. |
| 2 |     284 | High-porosity focus (φ = 0.60–0.80). Simulates osteoporotic-bone institutions. |
| 3 |     589 | Uniform mixed baseline — closest to IID reference node. |
| 4 |     119 | Small cell-size focus (cs = 1.0–1.8 mm). Mimics pediatric scaffold centres. |
| 5 |      82 | Schwarz_D + IWP heavy. Represents post-oncology/radiation cohort institutions. |

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
        print(f"Triangles: {n_tri}")  # typically 50k–300k for 128³
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
        python tpms_generator.py   # random_seed=42 baked in
        ```
        All stochastic elements use `numpy.random.default_rng(seed=42)`.
