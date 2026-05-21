"""Stability correction profile functions for all AeroBulk algorithms.

All functions are pure and JAX-differentiable, operating elementwise.
"""

import jax.numpy as jnp
from jaxerobulk.constants import rpi, vkarmn


def psi_m_coare(pzeta):
    """COARE 3.0/3.6 stability function for momentum.

    Blended Kansas (Paulson 1970) + convective form for unstable.
    Beljaars & Holtslag (1991) for stable.
    """
    zphi_m = jnp.abs(1.0 - 15.0 * pzeta) ** 0.25

    zpsi_k = (
        2.0 * jnp.log((1.0 + zphi_m) / 2.0)
        + jnp.log((1.0 + zphi_m * zphi_m) / 2.0)
        - 2.0 * jnp.arctan(zphi_m)
        + 0.5 * rpi
    )

    zphi_c = jnp.abs(1.0 - 10.15 * pzeta) ** 0.3333

    zpsi_c = (
        1.5 * jnp.log((1.0 + zphi_c + zphi_c * zphi_c) / 3.0)
        - 1.7320508 * jnp.arctan((1.0 + 2.0 * zphi_c) / 1.7320508)
        + 1.813799447
    )

    zf = pzeta * pzeta
    zf = zf / (1.0 + zf)
    zc = jnp.minimum(50.0, 0.35 * pzeta)
    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)

    unstable = (1.0 - zf) * zpsi_k + zf * zpsi_c
    stable = 1.0 + 1.0 * pzeta + 0.6667 * (pzeta - 14.28) / jnp.exp(zc) + 8.525
    return (1.0 - zstab) * unstable - zstab * stable


def psi_h_coare(pzeta):
    """COARE 3.0/3.6 stability function for heat/moisture."""
    zphi_h = jnp.abs(1.0 - 15.0 * pzeta) ** 0.5

    zpsi_k = 2.0 * jnp.log((1.0 + zphi_h) / 2.0)

    zphi_c = jnp.abs(1.0 - 34.15 * pzeta) ** 0.3333

    zpsi_c = (
        1.5 * jnp.log((1.0 + zphi_c + zphi_c * zphi_c) / 3.0)
        - 1.7320508 * jnp.arctan((1.0 + 2.0 * zphi_c) / 1.7320508)
        + 1.813799447
    )

    zf = pzeta * pzeta
    zf = zf / (1.0 + zf)
    zc = jnp.minimum(50.0, 0.35 * pzeta)
    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)

    unstable = (1.0 - zf) * zpsi_k + zf * zpsi_c
    stable = jnp.abs(1.0 + 2.0 * pzeta / 3.0) ** 1.5 + 0.6667 * (pzeta - 14.28) / jnp.exp(zc) + 8.525
    return (1.0 - zstab) * unstable - zstab * stable


def psi_m_ncar(pzeta):
    """NCAR (Large & Yeager 2004) stability function for momentum.

    Paulson (1970) for unstable, -5*zeta for stable.
    """
    zx2 = jnp.sqrt(jnp.abs(1.0 - 16.0 * pzeta))
    zx2 = jnp.maximum(zx2, 1.0)
    zx = jnp.sqrt(zx2)

    zpsi_unst = (
        2.0 * jnp.log((1.0 + zx) * 0.5)
        + jnp.log((1.0 + zx2) * 0.5)
        - 2.0 * jnp.arctan(zx)
        + rpi * 0.5
    )

    zpsi_stab = -5.0 * pzeta

    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)
    return zstab * zpsi_stab + (1.0 - zstab) * zpsi_unst


def psi_h_ncar(pzeta):
    """NCAR (Large & Yeager 2004) stability function for heat/moisture."""
    zx2 = jnp.sqrt(jnp.abs(1.0 - 16.0 * pzeta))
    zx2 = jnp.maximum(zx2, 1.0)

    zpsi_unst = 2.0 * jnp.log(0.5 * (1.0 + zx2))

    zpsi_stab = -5.0 * pzeta

    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)
    return zstab * zpsi_stab + (1.0 - zstab) * zpsi_unst


def psi_m_ecmwf(pzeta):
    """ECMWF (IFS Cy40r1) stability function for momentum.

    Paulson (1970) for unstable, Beljaars-Holtslag (1991) for stable.
    """
    zx = jnp.sqrt(jnp.abs(1.0 - 16.0 * pzeta))

    zpsi_unst = (
        jnp.log((1.0 + zx) * 0.5)
        + jnp.log((1.0 + zx * zx) * 0.5)
        - 2.0 * jnp.arctan(zx)
        + 0.5 * rpi
    )

    zc = jnp.minimum(50.0, 0.35 * pzeta)
    zpsi_stab = (
        -1.0 * (1.0 + pzeta + 0.6667 * (pzeta - 14.28) / jnp.exp(zc) + 8.525)
    )

    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)
    return zstab * zpsi_stab + (1.0 - zstab) * zpsi_unst


def psi_h_ecmwf(pzeta):
    """ECMWF (IFS Cy40r1) stability function for heat/moisture."""
    zx = jnp.sqrt(jnp.abs(1.0 - 16.0 * pzeta))

    zpsi_unst = 2.0 * jnp.log(0.5 * (1.0 + zx))

    zc = jnp.minimum(50.0, 0.35 * pzeta)
    zpsi_stab = -1.0 * (
        jnp.abs(1.0 + 2.0 * pzeta / 3.0) ** 1.5
        + 0.6667 * (pzeta - 14.28) / jnp.exp(zc)
        + 8.525
    )

    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)
    return zstab * zpsi_stab + (1.0 - zstab) * zpsi_unst


def psi_m_andreas(pzeta):
    """Andreas stability function for momentum.

    Paulson (1970) for unstable, Grachev et al. (2007) for stable.
    """
    zx = jnp.sqrt(jnp.abs(1.0 - 16.0 * pzeta))

    zpsi_unst = (
        jnp.log((1.0 + zx) * 0.5)
        + jnp.log((1.0 + zx * zx) * 0.5)
        - 2.0 * jnp.arctan(zx)
        + 0.5 * rpi
    )

    zpsi_stab = -5.0 * pzeta / (1.0 + pzeta * (0.156 + pzeta * (0.3555 + 0.26 * pzeta)))

    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)
    return zstab * zpsi_stab + (1.0 - zstab) * zpsi_unst


def psi_h_andreas(pzeta):
    """Andreas stability function for heat/moisture.

    Paulson (1970) for unstable, Grachev et al. (2007) for stable.
    """
    zx = jnp.sqrt(jnp.abs(1.0 - 16.0 * pzeta))

    zpsi_unst = 2.0 * jnp.log(0.5 * (1.0 + zx))

    zpsi_stab = -5.0 * pzeta / (1.0 + pzeta * (0.156 + pzeta * (0.3555 + 0.26 * pzeta)))

    zstab = jnp.where(pzeta >= 0, 1.0, 0.0)
    return zstab * zpsi_stab + (1.0 - zstab) * zpsi_unst
