# Introduction

This repository contains accompanying code for for the spatial probes presented in [Dual Computational Systems in the Development and Evolution of Mammalian Brains](https://www.biorxiv.org/content/10.1101/2024.11.19.624321v1.full.pdf). 

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

```
python setup.py
```

setup.py will download/train the necessary models. Vision and audio load pretrained PyTorch models. Olfaction, Touch, and Relational Memory train a model from scratch. Training the relational memory model takes 3-4 hours, so by default the script will simply load pre-trained weights (which are small enough to be in this repository.)

```
python main.py
```

main.py will run the series of default experiments, saving model output to output/ alternatively, you can use the quickstart.ipynb


# Credits
The main contribution of this repository is standardizing an interface to probe inputs across modalities. This wouldn't be possible without excellent existing repositories for each modality. They are:
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