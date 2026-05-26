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

## Bundled NAM models (`src/models/`)

- **Files:** 30 community-contributed `.nam` captures (see
  `src/models/README.md` for the contributor table).
- **Source:** [`pelennor2170/NAM_models`](https://github.com/pelennor2170/NAM_models)
- **License:** GPL-3.0 (full text in `src/models/GPL-3.0.txt`).
- **How it relates to the rest of the module:** The bundled `.nam` files are
  data, loaded at runtime by `nam.so` (MIT). They are not linked into the
  plugin binary and do not form a combined work. The tarball is an "aggregate"
  under GPL-3 § 5: the GPL-3 license applies to the model files only, and the
  MIT-licensed plugin code is unaffected. When redistributing, keep
  `GPL-3.0.txt` alongside the `.nam` files so § 6's source-availability
  requirement is met — the model files themselves are the "source" form, and
  the upstream repository above provides the same files for any user who
  wants them separately.

