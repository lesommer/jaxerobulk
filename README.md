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

from jaxerobulk.api import aerobulk_model

result = aerobulk_model(
    zt=2.0, zu=10.0,
    sst=jnp.float64(300.0), t_zt=jnp.float64(298.0),
    hum_zt=jnp.float64(0.018),
    U_zu=jnp.float64(5.0), V_zu=jnp.float64(0.0),
    slp=jnp.float64(101000.0),
    algo="coare3p6",
)
print(f"Cd={float(result['Cd']):.6e}  Qlat={float(result['Qlat']):.2f} W/m^2")

# Fully differentiable!
grad_cd = jax.grad(lambda sst: aerobulk_model(2., 10., sst, jnp.float64(298.), jnp.float64(0.018), jnp.float64(5.), jnp.float64(0.), jnp.float64(101000.), "coare3p6")["Cd"])(jnp.float64(300.))
```

## Project Status

All 7 iterations complete. See [docs/project_status.md](docs/project_status.md) for detailed progress and differentiability status.

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
pytest -q
```

## References

- Brodeau, L., B. Barnier, S. Gulev, and C. Woods, 2016: Climatologically significant effects of some approximations in the bulk parameterizations of turbulent air-sea fluxes. J. Phys. Oceanogr., doi:10.1175/JPO-D-16-0169.1.

## License

GPL-3.0-or-later (see [LICENSE](LICENSE))
