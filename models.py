# models.py

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import warnings

# Suppress optimization warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Data Structures ---
@dataclass
class MaterialProperties:
    """Container for material properties and system parameters."""
    mMo: float = 95.95
    mS: float = 32.065
    QS: float = -0.5
    QMo: float = 1.0
    m: int = 16
    n: int = 11

@dataclass
class WaveVectorData:
    """Container for wave vector data."""
    kk: np.ndarray

# --- Physics Calculations ---
class PhysicsCalculator:
    """Handles physics calculations for dipole moments and center of mass."""
    def __init__(self, props: MaterialProperties):
        self.props = props

    def calculate_dipole_moments(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dipole moments pA and pB."""
        n, m = self.props.n, self.props.m
        QMo, QS = self.props.QMo, self.props.QS

        MoxyzA = data['MoxyzA']
        S1xyzA, S2xyzA = data['S1xyzA'], data['S2xyzA']
        S1xyzB, S2xyzB = data['S1xyzB'], data['S2xyzB']
        S1xyzC, S2xyzC = data['S1xyzC'], data['S2xyzC']
        S1xyzD, S2xyzD = data['S1xyzD'], data['S2xyzD']

        pA = np.array([[
                MoxyzA[j, i*2] * QMo + (
                    S1xyzA[j, i] + S2xyzA[j, i] + S1xyzB[j, i] +
                    S2xyzB[j, i] + S1xyzC[j, i] + S2xyzC[j, i]
                ) * QS / 3
                for i in range(m // 2)]
            for j in range(n)])

        pB = np.array([[
                MoxyzA[j, i*2 + 1] * QMo + (
                    S1xyzB[j, i+1] + S2xyzB[j, i+1] + S1xyzC[j, i] +
                    S2xyzC[j, i] + S1xyzD[j, i] + S2xyzD[j, i]
                ) * QS / 3
                for i in range(m // 2 - 1)]
            for j in range(n)])
        return np.round(pA, 6), np.round(pB, 6)

    def calculate_center_of_mass(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate center of mass rA and rB."""
        n, m = self.props.n, self.props.m
        mMo, mS = self.props.mMo, self.props.mS

        MoxyzA = data['MoxyzA']
        S1xyzA, S2xyzA = data['S1xyzA'], data['S2xyzA']
        S1xyzB, S2xyzB = data['S1xyzB'], data['S2xyzB']
        S1xyzC, S2xyzC = data['S1xyzC'], data['S2xyzC']
        S1xyzD, S2xyzD = data['S1xyzD'], data['S2xyzD']

        rA = np.array([[
                (MoxyzA[j, i*2] * mMo + (
                    S1xyzA[j, i] + S2xyzA[j, i] + S1xyzB[j, i] +
                    S2xyzB[j, i] + S1xyzC[j, i] + S2xyzC[j, i]
                ) * mS) / (mMo + 6 * mS)
                for i in range(m // 2)]
            for j in range(n)])

        rB = np.array([[
                (MoxyzA[j, i*2 + 1] * mMo + (
                    S1xyzB[j, i+1] + S1xyzC[j, i] + S2xyzC[j, i+1] +
                    S1xyzD[j, i] + S2xyzD[j, i]
                ) * mS) / (mMo + 6 * mS)
                for i in range(m // 2 - 1)]
            for j in range(n)])
        return rA, rB

    @staticmethod
    def combine_arrays(arrayA: np.ndarray, arrayB: np.ndarray, n: int, m: int) -> np.ndarray:
        """Combine pA/pB or rA/rB arrays into a single array."""
        combined_list = []
        for i in range(n):
            temp = []
            for j in range(m // 2 - 1):
                temp.extend([arrayA[i, j], arrayB[i, j]])
            temp.append(arrayA[i, -1])
            combined_list.append(temp)
        return np.array(combined_list)

# --- Fitting Logic ---
class FittingFunctions:
    """Collection of static fitting functions."""
    _kz = 1.0
    _kx = 1.0

    @classmethod
    def set_k_values(cls, kz: float, kx: float):
        cls._kz, cls._kx = kz, kx

    @staticmethod
    def f1(x: np.ndarray, a: float) -> np.ndarray:
        """Cosine function: a * cos(kz * x)"""
        return a * np.cos(FittingFunctions._kz * x)

    @staticmethod
    def f2(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear + sine function: a * x + b * sin(kx * x)"""
        return a * x + b * np.sin(FittingFunctions._kx * x)

    @staticmethod
    def f3(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Sine + cosine function: a * sin(kz * x) + b * cos(kz * x)"""
        return a * np.sin(FittingFunctions._kz * x) + b * np.cos(FittingFunctions._kz * x)

    @staticmethod
    def f4(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Sine + cosine + constant: a * sin(kx * x) + b * cos(kx * x) + c"""
        return a * np.sin(FittingFunctions._kx * x) + b * np.cos(FittingFunctions._kx * x) + c

class NonlinearFitter:
    """Handles nonlinear curve fitting operations."""
    def __init__(self, wave_data: WaveVectorData):
        self.wave_data = wave_data
        self.functions = FittingFunctions()

    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def fit_data(self, x_data: np.ndarray, y_data: np.ndarray,
                 func: Callable, initial_params: List[float],
                 use_kx: bool = False) -> Tuple[List[np.ndarray], List[float]]:
        """Generic fitting function for all data types."""
        n = len(x_data)
        params_list = []
        r2_list = []

        for i in range(n):
            kz = self.wave_data.kk[i]
            kx = 2 * kz if use_kx else kz
            self.functions.set_k_values(kz, kx)

            try:
                popt, _ = curve_fit(func, x_data[i], y_data[i], p0=initial_params, maxfev=5000)
                params_list.append(popt)
                y_pred = func(x_data[i], *popt)
                r2_list.append(self.r_squared(y_data[i], y_pred))
            except Exception as e:
                print(f"Fitting failed for index {i}: {e}")
                params_list.append(np.array(initial_params))
                r2_list.append(0.0)
        return params_list, r2_list

# --- Symbolic Analysis ---
class SymbolicCalculator:
    """Handles symbolic mathematics using SymPy."""
    def __init__(self):
        self.symbols = self._define_symbols()
        self.equations = self._define_equations()

    def _define_symbols(self) -> Dict[str, sp.Symbol]:
        """Define all symbolic variables."""
        symbol_names = [
            'x', 'z', 'U0', 'V0', 'W0', 'cU', 'dzs', 'dzc',
            'dxs', 'dxc', 'cd', 'gd', 'k', 'kz', 'kx',
            'e_zzx', 'f_zxzx', 'e_xxx', 'f_xxxx'
        ]
        return {name: sp.Symbol(name) for name in symbol_names}

    def _define_equations(self) -> Dict[str, sp.Eq]:
        """Define symbolic equations."""
        s = self.symbols
        fUz_sym = s['U0'] * sp.cos(s['k'] * s['x'])
        fUx_sym = s['V0'] * s['x'] + s['W0'] * sp.sin(2 * s['k'] * s['x']) + s['cU']

        equzf_lhs = s['dzs'] * sp.sin(s['k'] * s['x']) + s['dzc'] * sp.cos(s['k'] * s['x'])
        equzf_rhs = s['e_zzx'] * sp.diff(fUz_sym, s['x']) / 2 + s['f_zxzx'] * sp.diff(fUz_sym, s['x'], 2) / 2

        equxf_lhs = s['dxs'] * sp.sin(2*s['k']*s['x']) + s['dxc'] * sp.cos(2*s['k']*s['x']) + s['cd']
        equxf_rhs = s['e_xxx'] * sp.diff(fUx_sym, s['x']) + s['f_xxxx'] * sp.diff(fUx_sym, s['x'], 2)

        return {'equzf': sp.Eq(equzf_lhs, equzf_rhs), 'equxf': sp.Eq(equxf_lhs, equxf_rhs)}

    def solve_coefficients(self) -> Dict[str, sp.Expr]:
        """Solve for the symbolic coefficients."""
        s, eq = self.symbols, self.equations
        f3131 = sp.solve(eq['equzf'].subs({s['dzs']: 0, s['e_zzx']: 0}), s['f_zxzx'])[0]
        e331 = sp.solve(eq['equzf'].subs({s['dzc']: 0, s['f_zxzx']: 0}), s['e_zzx'])[0]
        f1111 = sp.solve(eq['equxf'].subs({s['dxc']: 0, s['e_xxx']: 0, s['cd']: 0}), s['f_xxxx'])[0]
        e111 = sp.solve(eq['equxf'].subs({s['dxs']: 0, s['f_xxxx']: 0, s['cd']: 0, s['V0']: 0}), s['e_xxx'])[0]
        return {'f3131': f3131, 'e331': e331, 'f1111': f1111, 'e111': e111}
