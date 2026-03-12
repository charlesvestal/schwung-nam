# Move Everything NAM

Neural Amp Modeler audio effect module for Move Everything, based on [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) by Mike Oliphant and [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) by Steven Atkinson.

## Prerequisites

- [Move Everything](https://github.com/charlesvestal/move-everything) installed on your Ableton Move

## Installation

```bash
./scripts/install.sh
```

## Features

- **Neural amp/effect modeling**: Run trained NAM models for realistic amp and pedal emulation
- **Cabinet IR convolution**: Apply cabinet impulse responses with optional bypass
- **Model browser**: Hierarchical file browser for selecting `.nam` model files
- **Cabinet browser**: Browse and load `.wav` cabinet IR files
- **Input/Output level**: Independent gain staging controls

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| input_level | 0.0-1.0 | 0.5 | Input gain before model processing |
| output_level | 0.0-1.0 | 0.5 | Output gain after processing |
| cab_bypass | 0-1 | 0 | Bypass cabinet IR convolution |

## Adding Models and Cabinets

Place `.nam` model files and `.wav` cabinet IRs in the module directory on your Move:

```
/data/UserData/move-anything/modules/chain/audio_fx/nam/models/
/data/UserData/move-anything/modules/chain/audio_fx/nam/cabs/
```

NAM models can be trained with the [Neural Amp Modeler Trainer](https://github.com/sdatkinson/neural-amp-modeler).

## Building

```bash
./scripts/build.sh      # Build for ARM64 via Docker
./scripts/install.sh    # Deploy to Move
```

## Credits

- **NeuralAmpModelerCore**: [Steven Atkinson](https://github.com/sdatkinson/NeuralAmpModelerCore) (MIT License)
- **NeuralAudio**: [Mike Oliphant](https://github.com/mikeoliphant/NeuralAudio) (MIT License)
- **RTNeural**: [Jatin Chowdhury](https://github.com/jatinchowdhury18/RTNeural) (BSD 3-Clause License)
- **math_approx**: [Jatin Chowdhury](https://github.com/jatinchowdhury18/math_approx) (BSD 3-Clause License)
- **Move Everything port**: Charles Vestal

## License

MIT License - See LICENSE file for details.

## AI Assistance Disclaimer

This module is part of Move Everything and was developed with AI assistance, including Claude, Codex, and other AI assistants.

All architecture, implementation, and release decisions are reviewed by human maintainers.  
AI-assisted content may still contain errors, so please validate functionality, security, and license compatibility before production use.
