import torch
import os
from ..base.trainer import ModalityTrainer
from .grasp_trainer.trainer import Trainer


class Touch(ModalityTrainer): 
    
    def __init__(self, args):
        self.modality = 'touch'
        self.project_path = args.project_path
        self.device = args.device
        self.touch_frames = args.touch_frames
        self.patch_size = args.touch_patch_size
        self.touch_samples = args.touch_samples
        self.epochs = args.touch_epochs
        self.reload = args.setup_models
        self.som_size = args.som_size

    def setup_model(self):
        if self.reload:
            save_dir = f"{self.project_path}/models/touch/saved_data"
            os.makedirs(save_dir, exist_ok=True)

            trainer = Trainer(save_dir, nFrames=self.touch_frames)
            trainer.run(self.epochs)
            
            model = trainer.model
            val_dataset = trainer.val_loader.dataset

            torch.save(model.model.state_dict(), f"{save_dir}/model.pt")
            torch.save(val_dataset[:self.touch_samples], f"{save_dir}/val_dataset.pt")
            return model, val_dataset

    def setup_som(self):
        model = Trainer(f"{self.project_path}/models/touch/saved_data", nFrames=self.touch_frames).model
        model.model.load_state_dict(torch.load(f"{self.project_path}/models/touch/saved_data/model.pt", map_location=torch.device('cpu')))
        model.model.eval()
        self.model = model.model.module.cpu() #model.model.to('cpu')
        data = torch.load(f"{self.project_path}/models/touch/saved_data/val_dataset.pt", map_location=torch.device('cpu'))
        self.sample_data = [x.squeeze().reshape(self.touch_frames, 32, 32) for x in data[2]]
        self.average_hand = torch.stack(self.sample_data).mean(dim=(0,1))
        self.norm_mu = torch.mean(self.average_hand)
        self.norm_max = torch.max(torch.stack(self.sample_data))
        print (self.norm_max)

    def generate_static(self, coords):
        w1, w2 = coords
        p = self.patch_size
        static = torch.zeros_like(self.sample_data[0])
        static[:,w1:w1+p,w2:w2+p] = (0.9 + 0.1*torch.rand_like(static[:,w1:w1+p,w2:w2+p])) * self.norm_max
        static += self.norm_mu - static.mean()
        return (coords, static.reshape(self.sample_data[0].shape))

    def get_patches(self):
        patch_list = []
        x, y = self.sample_data[0][0].shape
        for i in range(0, x, self.patch_size):
            for j in range(0, y, self.patch_size):
                if self.average_hand[i][j] > 0.0667:
                    patch_list.append((i, j)) # only include patches if on the hand
        return patch_list

    def calculate_activations(self, x):
        with torch.inference_mode():
            acts = self.model(x.unsqueeze(dim=0), net_only=True)
            return acts.squeeze().detach().cpu().flatten()