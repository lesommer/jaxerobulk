# JaxeroBulk Project Context

## Purpose
This repository began as a demonstration during the 2026 general assembly of [AI4PEX](https://ai4pex.org) EU project. 

The purpose of the project is to port the FORTRAN90-based library Aerobulk (https://github.com/brodeau/aerobulk), which implement aerodynamic bulk formulae used in ocean circulation models, to Python/Jax/Equinox. 

The project primary objective is to harden Opencode development skills with a real scientific-computing codebase, testing JAX + Equinox  implementation quality and  Opencode context and directive-writing quality.

## Contracts
- **Exposes**: CLI and API inherited from the original https://github.com/brodeau/aerobulk (tag. 1.0.0)
- **Guarantees**: behavior changes are test-backed; docs track current intent and constraints
- **Expects**: contributors optimize for correctness with respect to the reference implementation, reproducibility, not feature volume.

## Dependencies
- **Uses**: JAX ecosystem (`jax`, `equinox`, `lineax`, `optimistix`), `scipy`.
- **Used by**: CLI workflows, regression/IO tests, and coding-agent hardening exercises.
- **Boundary**: runtime code in `src/` should be entirely in python and should not include any compiled module from the original Fortran implementation

## Invariants
- `pytest -q` must stay green after behavior changes.
- CLI remains a thin imperative shell around library modules.
- Documentation must explicitly reflect mission or contract shifts.

## Key Decisions
- Treat `docs/implementation_plan.md` and `docs/project_status.md` as historical baseline plus dated reassessments.
- Use this root `AGENTS.md` as canonical project context

## Commands
- `pytest -q` - run full test suite.
- `python -m jaxerobulk.cli --help` - inspect CLI surface.
- `python -m jaxerobulk.cli <command> --help` - inspect a specific workflow.

## Project Structure
- `src/jaxerobulk/` - package runtime 
- `tests/` - regression, IO, and CLI behavior tests.
- `docs/` - project plans and licensing guidance.

## Behavior Requirements For Agents
- Prefer test-first or test-coupled changes for any behavioral modification.
- Update context docs when project goals, contracts, or boundaries change.
- Report concrete verification evidence (commands and outcomes), not assumptions.

## Boundaries
- Safe to edit: `src/`, `tests/`, `docs/`, `README.md`, `AGENTS.md`.
- Do not copy source text from `AeroBulk` into GNU-licensed runtime code.
