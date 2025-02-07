# Introduction

This repository contains code for the spatial probes presented in [Dual Computational Systems in the Development and Evolution of Mammalian Brains](https://www.biorxiv.org/content/10.1101/2024.11.19.624321v1.full.pdf). 

## Project Structure
```
├── analysis/         # SOM computation and descriptive statistics
├── models/           # Model probing code (organized by modality)
├── output/           # Saved experimental outputs
├── quickstart.ipynb  # Basic experiment examples
├── main.py           # Main experiment runner
├── setup.py          # Download/train base models
└── config.py         # Hyperparameter configuration
```

# Getting Started

Environment requirements:

```
pip install numpy pandas scipy scikit-learn matplotlib seaborn pillow ipykernel
pip install torch torchvision torchaudio
pip install altair datasets
pip install numba numbasom==0.0.5
pip install tensorboard

apt-get install -y ffmpeg # required for audio model
```
 

## Base Model Setup

The code will be default source models from the following:
- Vision: Download pretrained from PyTorch
- Audio: Download pretrained from PyTorch
- Olfaction: Use saved weights
- Memory: Use saved weights
- Touch: Use saved weights

To retrain olfaction and touch locally, run the below. Additionally, memory can be run by setting tem_train=True in config.py

```
python setup_base_models.py
```

Training Olfaction/Touch take 5-10 minutes using a 4090 GPU, with peak VRAM usage of approximately 16gb (although this could be lowered by adjusting batch size). The memory model trains in 3-4 hours.

## Running Main Experiments

Main experiments can be run with main.py or in the quickstart notebook. Code loops over each modality and runs the necessary experiments. 

```
python main.py
```

Runs default experiments, with results saved to _output/_. Alternatively, you can use the quickstart.ipynb


# Credits
The main contribution of this repository is standardizing an interface to probe inputs across modalities. This wouldn't be possible without excellent existing repositories for each modality. While the paper contains the full list of citations, there are a handful of repositories that were particularly helpful for making this code, they are:
- [Vision Model](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) 👁️
- [Olfaction Model](https://github.com/gyyang/olfaction_evolution) 👃
- [Touch Model](https://github.com/erkil1452/touch/tree/master) 🖐️
- [Memory Model](https://github.com/jbakermans/torch_tem) 🧠
- [Audio Model](https://github.com/pytorch/audio/blob/main/examples/tutorials/speech_recognition_pipeline_tutorial.py) 👂
- [Calculating SOMs](https://github.com/nmarincic/numbasom) 🗺️

# Citation
If you find this repository helpful, please consider citing our paper:
```
@article {Imam2024.11.19.624321,
	author = {Imam, Nabil and Kielo, Matthew and Trude, Brandon M. and Finlay, Barbara L.},
	title = {Dual Computational Systems in the Development and Evolution of Mammalian Brains},
	year = {2024},
	doi = {10.1101/2024.11.19.624321},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/11/19/2024.11.19.624321},
}
```
