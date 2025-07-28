# data_preprocessor.py

import numpy as np
from typing import Dict

def normalize_and_derive_coordinates(
    S1xyz: np.ndarray,
    S2xyz: np.ndarray,
    Moxyz: np.ndarray,
    n: int,
    m: int
) -> Dict[str, np.ndarray]:
    """
    Normalizes z-coordinates and creates derived coordinate sets (A, B, C, D).
    """
    reference_z = np.array([
        1.565, 1.5869, 1.6008, 1.6225, 1.6441, 1.6656, 1.687,
        1.7081, 1.7291, 1.7498, 1.7703
    ])
    dy = 0.315 - 0.1575

    # --- Z-value normalization ---
    for i in range(n):
        S1xyz[i, :, 2] -= reference_z[i]
        S2xyz[i, :, 2] -= reference_z[i]
        Moxyz[i, :, 2] -= reference_z[i]

    # --- Derive coordinate sets ---
    S1_sets = {'A': [], 'B': [], 'C': [], 'D': []}
    S2_sets = {'A': [], 'B': [], 'C': [], 'D': []}

    pair_count = m // 2
    for i in range(n):
        # Temp lists for the current row
        s1_row = {key: [] for key in S1_sets}
        s2_row = {key: [] for key in S2_sets}

        for p in range(pair_count):
            odd_idx, even_idx = 2 * p, 2 * p + 1

            # S1xyz processing
            s1_row['B'].append(S1xyz[i, odd_idx])
            s1_row['C'].append(S1xyz[i, even_idx])
            up1 = S1xyz[i, even_idx].copy()
            up1[1] += 2 * dy
            s1_row['A'].append(up1)

            # S2xyz processing
            s2_row['B'].append(S2xyz[i, odd_idx])
            s2_row['C'].append(S2xyz[i, even_idx])
            up2 = S2xyz[i, even_idx].copy()
            up2[1] += 2 * dy
            s2_row['A'].append(up2)

        # The original code has a peculiar condition for D sets
        for p in range(pair_count - 1): # p > 0 is equivalent to p from 1 to 7
             even_idx = 2 * (p + 1)
             down1 = S1xyz[i, even_idx].copy()
             down1[1] -= 2 * dy
             s1_row['D'].append(down1)

             down2 = S2xyz[i, even_idx].copy()
             down2[1] -= 2* dy
             s2_row['D'].append(down2)


        for key in S1_sets:
            S1_sets[key].append(s1_row[key])
            S2_sets[key].append(s2_row[key])

    return {
        "S1xyzA": np.array(S1_sets['A']), "S1xyzB": np.array(S1_sets['B']),
        "S1xyzC": np.array(S1_sets['C']), "S1xyzD": np.array(S1_sets['D']),
        "S2xyzA": np.array(S2_sets['A']), "S2xyzB": np.array(S2_sets['B']),
        "S2xyzC": np.array(S2_sets['C']), "S2xyzD": np.array(S2_sets['D']),
        "MoxyzA": Moxyz[:, :m-1, :]
    }
