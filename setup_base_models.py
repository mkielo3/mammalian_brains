import torch
import pandas as pd

from tqdm import tqdm
from numbasom.core import lattice_closest_vectors
from analysis.som import SOM
from utils import save_output_to_pickle

from models.olfaction.olfaction import Olfaction
from models.vision.vision import Vision
from models.audio.audio import Audio
from models.touch.touch import Touch
from models.memory.memory import Memory

from config import Args

def main():
    args = Args()
    args.setup_models = True
    for modality in ([Olfaction(args), Vision(args), Audio(args), Touch(args), Memory(args)]):
        print ("\n", modality.modality, pd.Timestamp.now())
        modality.setup_model()


if __name__ == "__main__":
    main()