# Introduction

This repository contains code for the spatial probes presented in [Dual Computational Systems in the Development and Evolution of Mammalian Brains](https://www.biorxiv.org/content/10.1101/2024.11.19.624321v1.full.pdf). 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkielo3/mammalian_brains/blob/main/quickstart.ipynb)

# Getting Started

Environment requirements:

```
pip install -q numpy pandas scipy scikit-learn matplotlib seaborn pillow ipykernel torch torchvision torchaudio altair datasets numba numbasom==0.0.5 tensorboard pysal vl-convert-python
apt-get install -y ffmpeg # required for audio model
```

Run sample experiments:

```
python main.py --fast
```

This will produce 5x5 unit SOM maps and receptive fields saved to output/ and runs in about 10 minutes on a M1 Macbook Air on CPU.

Alternatively, you can run the 25x25 unit SOM displayed in the paper, which runs in about 2 hours on the same setup.

```
python main.py
```

For convenience, we include the saved weights of the Olfaction, Somatosensation, and TEM models. If you choose, these can be retrained with:

```
python setup_base_models.py
```

Retraining Olfaction and Somatosensation requires 20gb of memory, and completes in ~10 minutes on an RTX 4090 GPU. The TEM model is much slower, requiring 4-6 hours. You can select which models to retrain in the config.py file. By default, TEM will not be retrained.


## Project Structure
```
├── analysis/         # SOM computation and descriptive statistics
├── models/           # Model probing (organized by modality)
├── output/           # Saved experimental outputs
├── quickstart.ipynb  # Basic experiment examples
├── main.py           # Main experiment runner
├── setup.py          # Download/train base models
└── config.py         # Hyperparameter configuration
```

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
