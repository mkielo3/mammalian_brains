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
import argparse
from utils import save_som_plot, save_rf_plot

from config import Args

# Set all seeds
from numba import njit
import numpy as np

@njit
def set_numba_seed(value):
    np.random.seed(value)

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)
set_numba_seed(SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Use smaller SOM size (5,5)')
    cmd_args = parser.parse_args()
    
    fast = cmd_args.fast
    args = Args()
    args.experiment_name = "main_results"
    args.som_size = (5,5) if fast else (25, 25)
    args.experiment_name = "main_results_fast" if fast else "main_results"
    args.fast = fast
    args.som_epochs = 2000 if fast else 100000
    modality_list = [Olfaction(args), Vision(args), Audio(args), Touch(args), Memory(args)]

    for modality in modality_list:
        print ("\n", modality.modality, pd.Timestamp.now())

        # 1. Train/Download Model
        modality.setup_model()
        modality.setup_som()

        # 2. Get activations for each patch
        patches = modality.get_patches()
        activation_list = []
        for p in tqdm(patches):
            p, static = modality.generate_static(p)
            activation = modality.calculate_activations(static)
            activation_list.append([p, activation])

        # 3. Fit SOM
        x_mat = torch.stack([x[1] for x in activation_list]).numpy()
        som = modality.initialize_som(SOM)
        lattice = som.train(x_mat, num_iterations=args.som_epochs, initialize=args.som_init, normalize=False, start_lrate=args.som_lr)
        
        # 4. Get coordinates for each BMU
        coordinate_list = [x[0] for x in activation_list]
        closest = lattice_closest_vectors(x_mat, lattice, additional_list=coordinate_list)

        # 5. Save
        output = {"closest": closest, 
                "coord_map": coordinate_list,
                "x_range": (0, max([x[0][0] for x in activation_list])),
                "y_range": (0, max([x[0][1] for x in activation_list])),
                "lattice": lattice,
                "som": None,
                "samples": modality.sample_data,
                "modality": modality.modality,
                "args": args,
                "activations": activation_list}

        save_output_to_pickle(output, args.experiment_name)
    
    save_som_plot(args.experiment_name, modality_list, args)
    save_rf_plot(args.experiment_name, modality_list)


if __name__ == "__main__":
    main()