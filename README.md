# JaxeroBulk

A differentiable implementation of [AeroBulk](https://github.com/brodeau/aerobulk) aerodynamic bulk formulae in Python/JAX/Equinox.

## What It Does

JaxeroBulk computes turbulent air-sea fluxes (wind stress, sensible heat, latent heat, evaporation) using aerodynamic bulk formulae with Monin-Obukhov similarity theory. It provides:

- **5 bulk algorithms**: COARE 3.0, COARE 3.6, NCAR, ECMWF, Andreas (sea ice)
- **Cool-skin/warm-layer**: For COARE and ECMWF algorithms
- **Full differentiability**: Forward and reverse mode autodiff through JAX

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jaxerobulk.thermodynamics import q_sat, rho_air, l_vap

# Saturation specific humidity at 20°C, 1010 hPa
qs = q_sat(jnp.float64(293.15), jnp.float64(101000.0))

# Air density at 20°C, q=0.01 kg/kg, 1010 hPa
rho = rho_air(jnp.float64(293.15), jnp.float64(0.01), jnp.float64(101000.0))

# Differentiable!
grad_qs = jax.grad(lambda T: q_sat(T, jnp.float64(101000.0)))(jnp.float64(293.15))
```

## Project Status

See [docs/project_status.md](docs/project_status.md) for detailed implementation progress and differentiability status.

**Iteration 1 (current)**: Constants and core thermodynamic functions — 61 tests passing, all differentiable.

## Algorithms

| Algorithm | Description | CSWL Support |
|-----------|-------------|-------------|
| COARE 3.0 | Fairall et al. 2003 | Yes |
| COARE 3.6 | Edson et al. 2013 | Yes |
| NCAR | Large & Yeager 2004 | No |
| ECMWF | IFS Cy40r1 | Yes |
| Andreas | Andreas et al. 2015 (ice) | No |

## Testing

```bash
JAX_ENABLE_X64=True pytest -q
```

## References

- Brodeau, L., B. Barnier, S. Gulev, and C. Woods, 2016: Climatologically significant effects of some approximations in the bulk parameterizations of turbulent air-sea fluxes. J. Phys. Oceanogr., doi:10.1175/JPO-D-16-0169.1.

## License

MIT
