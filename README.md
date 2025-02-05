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
conda create -n mammal python=3.10
conda activate mammal
conda install numpy pandas scipy scikit-learn matplotlib seaborn pillow ipykernel
conda install pytorch torchvision torchaudio -c pytorch
pip install altair
conda install numba
pip install numbasom==0.0.5
```
 


```
python setup.py
```

To download/train the necessary base models. Vision and audio copy pretrained PyTorch models. Olfaction and Touch are trained from scratch and will benefit from having a gpu available. Relational Memory can be trained in 3-4 hours, but for convenience weights are included in this repository.

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
