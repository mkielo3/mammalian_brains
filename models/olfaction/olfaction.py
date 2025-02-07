import torch
import os
from ..base.trainer import ModalityTrainer
from .olfaction_trainer.model import load_and_train_model, FullModel


class Olfaction(ModalityTrainer): 
    
    def __init__(self, args):
        self.modality = 'olfaction'
        self.project_path = args.project_path
        self.device = args.device
        self.olfaction_epochs = args.olfaction_epochs
        self.olfaction_samples = args.olfaction_samples
        self.patch_size = args.olfaction_patch_size
        self.train_n_orn = args.olfaction_n_orn
        self.train_n_class = args.olfaction_n_class
        self.reload = args.setup_models
        self.som_size = args.som_size

    def setup_model(self):
        if self.reload:
            save_dir = f"{self.project_path}/models/olfaction/saved_data"
            os.makedirs(save_dir, exist_ok=True)
            model, val_dataset = load_and_train_model(self.device, epochs=self.olfaction_epochs, n_orn=self.train_n_orn, n_class=self.train_n_class)
            torch.save(model.state_dict(), f"{save_dir}/model.pt")
            save_data = val_dataset[:self.olfaction_samples].clone().contiguous().to('cpu')
            torch.save(save_data, f"{save_dir}/val_dataset.pt")
            return model, val_dataset

    def setup_som(self):
        model = FullModel(self.train_n_class, self.train_n_orn)
        model.load_state_dict(torch.load(f"{self.project_path}/models/olfaction/saved_data/model.pt"))
        model.eval()
        self.model = model.to('cpu')
        data = torch.load(f"{self.project_path}/models/olfaction/saved_data/val_dataset.pt").to('cpu')
        self.norm_mu = data.mean()
        self.norm_max = torch.max(data)
        self.sample_data = data

    def generate_static(self, coords):
        w = coords[0]
        static = torch.zeros_like(self.sample_data[0])
        static[w:w+self.patch_size] = torch.rand_like(static[w:w+self.patch_size]) #(0.9 + 0.1* * self.norm_max
        static += self.norm_mu - static.mean()
        return (coords, static)

    def get_patches(self):
        return [(x, 0) for x in range(0, len(self.sample_data[0]), self.patch_size)]

    def calculate_activations(self, x):
        with torch.inference_mode():
            return self.model(x.unsqueeze(dim=0), None).flatten().detach()
