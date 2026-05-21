## GOAL
Your goal is to port the FORTRAN90-based library Aerobulk (https://github.com/brodeau/aerobulk), which implement aerodynamic bulk formulae used in ocean circulation models, to Python/Jax/Equinox. 

## GUIDELINES
The code should implement strictly the algorithm of the FORTRAN90 based implementation. The code should be implemented in Jax/Equinox, with proven differentiability for the entire codebase (forward and reverse mode autodiff). The code should be callable from python with an API replicating the behavior of the FORTRAN and C++ API, and strictly replicating the python API available in the current Fortran implementation. The code should also include a CLI replicating the CLI of the Fortran implementation (with python scripts replicating the binary executables available with the orginal aerobulk after compilation and all the bash scripts). The codebase should include (and pass) all the tests provided in the FORTRAN implementation, and include additional extensive test guaranteeing the differentiability of all the routine and functions.  

## STEPS

- scan the  FORTRAN implementation of AeroBulk available locally from `../aerobulk/` or online at `https://github.com/brodeau/aerobulk/`
- write in `docs/bulk_formulae.md` a desccription of the general formulation of bulk formulae and of the specific version to implemented in AeroBulk 
- prepare and write a plan in  `docs/implementation_plan.md` 
- proceed withe plan with a ralph loop. at the end of each iteration, the code should be pip installable, and fonctional, with tests corresponding to the current level of implementation. all code should be commited locally and pushed to the GH repos. large development should be done in dedicated branches and merged only when fonctional
- the README.mld should be in phase with the code before each commit (content, desription, CLI, algorithm, project status, ...). The plots contained in README.md should be generated with one of JaxeroBulk diagnostic Python scripts
- at the end of each iteration, keep a clear and short description of the project status in `docs/project_status.md`, including information on the  differentiability of the codebase. 
