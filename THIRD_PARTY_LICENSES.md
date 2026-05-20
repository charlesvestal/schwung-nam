# Third-party licenses

This module links against and/or bundles content from the projects listed
below. Their licenses are honored separately from this module's own MIT
license (see `LICENSE`).

## NeuralAudio (DSP engine)

- **Project:** [`mikeoliphant/NeuralAudio`](https://github.com/mikeoliphant/NeuralAudio)
- **Author:** Mike Oliphant
- **License:** MIT
- **How it's used:** Statically linked as `libNeuralAudio.a` (built from
  `deps/NeuralAudio`).

## RTNeural

- **Project:** [`jatinchowdhury18/RTNeural`](https://github.com/jatinchowdhury18/RTNeural)
- **Author:** Jatin Chowdhury and contributors
- **License:** BSD 3-Clause
- **How it's used:** Statically linked via NeuralAudio's submodule.

## NeuralAmpModelerCore

- **Project:** [`sdatkinson/NeuralAmpModelerCore`](https://github.com/sdatkinson/NeuralAmpModelerCore)
- **Author:** Steven Atkinson
- **License:** MIT
- **How it's used:** Headers pulled in via NeuralAudio for `.nam` model
  loading.

## Eigen

- **Project:** [Eigen](https://eigen.tuxfamily.org/)
- **License:** MPL2
- **How it's used:** Header-only, pulled in transitively via RTNeural.

