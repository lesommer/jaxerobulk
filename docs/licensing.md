# Licensing

jaxerobulk is dual-licensed under **GPL-3.0-or-later OR MIT**, at the
option of the user of the code (SPDX license expression). The license
texts live in [LICENSE.GPL](../LICENSE.GPL) and
[LICENSE.MIT](../LICENSE.MIT); the top-level [LICENSE](../LICENSE) file
states the choice.

## Why the dual license (2026-11-02)

The project was originally GPL-3.0-or-later only, following the license
of the reference implementation ([AeroBulk](https://github.com/brodeau/aerobulk),
tag 1.0.0). The code is an original Python/JAX/Equinox port (behavior is
test-matched against AeroBulk outputs, but no AeroBulk source text is
copied), and the sole copyright holder is Julien Le Sommer, so the
license can be offered at the holder's option.

The practical motivation: the reproduction package of the neutral-bench
ecosystem (MIT) needs to import jaxerobulk to compute CORE.v2 air-sea
fluxes for a water-mass-transformation figure. A GPL-only jaxerobulk
would make the MIT repro package effectively GPL-restricted. With the
dual license, downstream MIT projects consume jaxerobulk under MIT,
while GPL-3.0-or-later remains available for copyleft uses.

## Constraints the dual license does not relax

- Do not copy source text from AeroBulk into runtime code: AeroBulk is
  GPL-3.0, and copied text could not be offered under the MIT option.
  Behavior must be matched through tests, not transcription.
