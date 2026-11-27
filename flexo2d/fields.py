"""Turning charges and geometry into the fields that get fitted.

For every configuration this builds, one value per cluster:

* the cluster **dipole moment** ``p`` (e.nm), and
* the cluster **centroid** ``r`` (nm, mass-weighted by default),

then the **displacement field** ``U = r - r_reference``. Those are exactly the
quantities the constitutive model in :mod:`flexo2d.modes` is fitted to.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .charges import ChargeModel
from .clusters import Cluster, ClusterScheme
from .structure import ConfigSeries, Structure


@dataclass
class FieldSet:
    """Per-configuration cluster fields.

    Attributes
    ----------
    centroids
        ``(n_config, n_cluster, 3)`` cluster centroids in nm.
    dipoles
        ``(n_config, n_cluster, 3)`` cluster dipole moments in e.nm.
    displacements
        ``(n_config, n_cluster, 3)`` centroid displacement from the reference
        configuration, in nm.
    net_charges
        ``(n_config, n_cluster)`` net charge per cluster, in e. Should be ~0.
    amplitudes
        The series control parameter, ``(n_config,)``.
    wavevectors
        Ripple wavevector per configuration, ``(n_config,)`` in 1/nm.
    cluster_area
        In-plane area per cluster, in nm^2 (cell area / number of clusters).
    """

    centroids: np.ndarray
    dipoles: np.ndarray
    displacements: np.ndarray
    net_charges: np.ndarray
    amplitudes: np.ndarray
    wavevectors: np.ndarray
    cluster_area: float

    @property
    def n_config(self) -> int:
        return self.centroids.shape[0]

    @property
    def n_cluster(self) -> int:
        return self.centroids.shape[1]

    def profile(self, quantity: str, component: int, config: int,
                axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """A 1D profile ``(coordinate, value)`` ready for fitting.

        Parameters
        ----------
        quantity
            ``"dipole"``, ``"displacement"`` or ``"centroid"``.
        component
            Cartesian component of the quantity (0=x, 1=y, 2=z).
        config
            Configuration index.
        axis
            Which centroid coordinate to use as the abscissa (the ripple
            direction, normally 0 = x).
        """
        data = {
            "dipole": self.dipoles,
            "displacement": self.displacements,
            "centroid": self.centroids,
        }
        try:
            arr = data[quantity]
        except KeyError as exc:
            raise KeyError(
                f"unknown quantity {quantity!r}; choose from {sorted(data)}"
            ) from exc
        x = self.centroids[config, :, axis]
        y = arr[config, :, component]
        order = np.argsort(x)
        return x[order], y[order]


def cluster_dipole(structure: Structure, cluster: Cluster,
                   charges: np.ndarray, origin: np.ndarray | None = None
                   ) -> np.ndarray:
    """Dipole moment of one cluster, in e.nm.

    `origin` is the point the moment is taken about. For a charge-neutral
    cluster the result does not depend on it; for a cluster carrying net charge
    it does, and the cluster's own centroid is then the only sensible choice --
    pass it in.
    """
    pos = cluster.positions(structure)
    if origin is not None:
        pos = pos - np.asarray(origin, dtype=float)
    q = charges[cluster.members] * cluster.weights
    return q @ pos


def cluster_centroid(structure: Structure, cluster: Cluster,
                     weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted centroid of one cluster, in nm.

    `weights` defaults to atomic masses, giving the centre of mass of the local
    unit -- the natural material point for a continuum displacement field.
    """
    pos = cluster.positions(structure)
    w = (structure.masses if weights is None else np.asarray(weights, float))
    w = w[cluster.members] * cluster.weights
    total = w.sum()
    if total == 0:
        raise ValueError(f"cluster centred on atom {cluster.center} has zero weight")
    return (w @ pos) / total


def cluster_net_charge(cluster: Cluster, charges: np.ndarray) -> float:
    return float(np.sum(charges[cluster.members] * cluster.weights))


def build_fields(
    series: ConfigSeries,
    charge_model: ChargeModel,
    cluster_scheme: ClusterScheme,
    *,
    axis: int = 0,
    n_periods: int = 1,
    centroid_weights: str = "mass",
    neutrality_tol: float = 1e-6,
    dipole_origin: str = "centroid",
    wavevectors: Sequence[float] | None = None,
) -> FieldSet:
    """Build cluster dipoles, centroids and displacements for a whole series.

    Parameters
    ----------
    series
        The deformation series.
    charge_model
        Supplies per-atom charges.
    cluster_scheme
        Supplies the local dipole units.
    axis
        Ripple propagation direction (0 = x).
    n_periods
        Number of ripple periods in the supercell, used for the wavevector.
    centroid_weights
        ``"mass"`` for the centre of mass, ``"uniform"`` for the geometric
        centre.
    neutrality_tol
        Largest per-cluster net charge tolerated before a warning is issued.
    dipole_origin
        Point each cluster's dipole is taken about. ``"centroid"`` (default)
        uses the cluster's own centroid, which keeps the dipole well defined
        even when per-atom charges leave a cluster slightly non-neutral; for a
        neutral cluster it is identical to ``"cell"``, which uses the cell
        origin.
    wavevectors
        Override the geometric wavevectors (rarely needed).

    Returns
    -------
    FieldSet
    """
    if centroid_weights not in ("mass", "uniform"):
        raise ValueError("centroid_weights must be 'mass' or 'uniform'")
    if dipole_origin not in ("centroid", "cell"):
        raise ValueError("dipole_origin must be 'centroid' or 'cell'")

    ref = series.reference
    clusters = cluster_scheme.clusters(ref)
    n_cl = len(clusters)

    centroids = np.zeros((len(series), n_cl, 3))
    dipoles = np.zeros((len(series), n_cl, 3))
    nets = np.zeros((len(series), n_cl))

    for i, structure in enumerate(series):
        # Re-derive clusters per configuration: bonds can change under large
        # deformation, and we want the grouping that matches each geometry.
        cl = cluster_scheme.clusters(structure)
        if len(cl) != n_cl:
            raise ValueError(
                f"configuration {i} gives {len(cl)} clusters but the reference "
                f"gives {n_cl}; the cutoff is marginal for this deformation"
            )
        q = charge_model.charges(structure, i)
        w = None if centroid_weights == "mass" else np.ones(structure.n_atoms)
        for j, c in enumerate(cl):
            centroid = cluster_centroid(structure, c, w)
            centroids[i, j] = centroid
            origin = centroid if dipole_origin == "centroid" else None
            dipoles[i, j] = cluster_dipole(structure, c, q, origin)
            nets[i, j] = cluster_net_charge(c, q)

    worst = np.abs(nets).max()
    if worst > neutrality_tol:
        message = (
            f"clusters carry net charge (worst |net| = {worst:.3e} e > "
            f"{neutrality_tol:g}); "
        )
        if dipole_origin == "cell":
            raise ValueError(
                message + "with dipole_origin='cell' the dipole is then "
                "origin-dependent and the result is meaningless. Use "
                "dipole_origin='centroid', or neutralise the charges with "
                "flexo2d.charges.neutralize."
            )
        warnings.warn(
            message + "their dipoles are taken about each cluster's own "
            "centroid, so they stay well defined, but charge transferred "
            "between clusters does not contribute to the polarization.",
            RuntimeWarning,
            stacklevel=2,
        )

    displacements = centroids - centroids[series.reference_index][None, :, :]

    if wavevectors is None:
        k = series.wavevectors(axis=axis, n_periods=n_periods)
    else:
        k = np.asarray(wavevectors, dtype=float)
        if len(k) != len(series):
            raise ValueError(
                f"{len(k)} wavevectors for {len(series)} configurations"
            )

    return FieldSet(
        centroids=centroids,
        dipoles=dipoles,
        displacements=displacements,
        net_charges=nets,
        amplitudes=series.amplitudes,
        wavevectors=k,
        cluster_area=ref.in_plane_area / n_cl,
    )
