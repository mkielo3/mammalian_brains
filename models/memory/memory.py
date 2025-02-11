import torch
import os
from ..base.trainer import ModalityTrainer
from .torch_tem.train import train
from .torch_tem import hooked_model
from .torch_tem import world
import numpy as np

import glob

class Memory(ModalityTrainer): 
    
    def __init__(self, args):
        """This is a wrapper around torch_tem, and offers very little customization"""
        self.modality = 'memory'
        self.project_path = args.project_path + "/models/memory/torch_tem"
        self.reload = args.setup_models
        self.som_size = args.som_size
        self.tem_train = args.tem_train
        self.run = 0
        self.n_rollouts = args.tem_n_rollouts
        self.batch_size = args.tem_batch_size
        self.patch_size = args.tem_patch_size

    def setup_model(self):
        if self.reload and self.tem_train: # has special flag because it takes 3-4 hours
            train(self.project_path)

    def setup_som(self):
        # load model
        params = torch.load(f"{self.project_path}/Summaries/latest/run{self.run}/model/params_12000.pt", weights_only=False)
        tem = hooked_model.Model(params, self.patch_size)
        model_weights = torch.load(f"{self.project_path}/Summaries/latest/run{self.run}/model/tem_12000.pt", weights_only=True)
        tem.load_state_dict(model_weights)
        envs = list(glob.iglob(f'{self.project_path}/Summaries/latest/run{self.run}/script/envs/*'))

        # generate rollouts
        params['n_rollout'] = self.n_rollouts
        params['batch_size'] = self.batch_size
        environments = [world.World(graph, randomise_observations=True, shiny=(params['shiny'] if np.random.rand() < params['shiny_rate'] else None)) for graph in np.random.choice(envs,params['batch_size'])]

        visited = [[False for _ in range(env.n_locations)] for env in environments]
        walk_len = np.random.randint(params['walk_it_min'], params['walk_it_max'])
        walks = [env.generate_walks(params['n_rollout']*walk_len, 1)[0] for env in environments]
        prev_iter = None
        
        chunk = []
        for env_i, walk in enumerate(walks):
            for step in range(params['n_rollout']):
                if len(chunk) < params['n_rollout']:
                    chunk.append([[comp] for comp in walk.pop(0)])
                else:
                    for comp_i, comp in enumerate(walk.pop(0)):
                        chunk[step][comp_i].append(comp)
        for i_step, step in enumerate(chunk):
            chunk[i_step][1] = torch.stack(step[1], dim=0)

        tem.forward(chunk, prev_iter)
        self.activation_matrix = torch.vstack([torch.hstack(x[1]) for x in tem.activation_log]).numpy()
        self.mask_location = [x[0] for x in tem.activation_log for _ in range(params['batch_size'])]
        self.sample_data = chunk

    def generate_static(self, p):
        return ((p[1], 0), self.activation_matrix[p[0]]) # unroll to patch location

    def get_patches(self):
        return [(i, x) for (i, x) in enumerate(self.mask_location)]

    def calculate_activations(self, x):
        # just to fit the pattern, all activations calculated in HookedModel
        return torch.Tensor(x)
