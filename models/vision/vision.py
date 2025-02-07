import torch
import os
from torchvision import transforms
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from PIL import Image
from ..base.trainer import ModalityTrainer


def get_headless_resnet():
    """Create headless ResNet18 model on specified device"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    headless_model = nn.Sequential(
        model.conv1,
        model.bn1, 
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4
    )
    headless_model.eval()
    return headless_model


def get_data(n_samples):
    # Prep dataset
#    ds = load_dataset("imagenet-1k", split='validation', streaming=True).with_format("torch")
    ds = load_dataset("vaishaal/ImageNetV2", split='train').with_format("torch")
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_images = []
    for i, batch in enumerate(dl):
        if len(processed_images) >= n_samples:
            break
        
        # img = batch['image'][0]
        img = batch['jpeg'][0]
        if img.shape[0] == 3:
            pil_img = Image.fromarray(img.permute(1, 2, 0).numpy())
            img = transform(pil_img)
            processed_images.append(img)

    return torch.stack(processed_images)



class Vision(ModalityTrainer): 
    
    def __init__(self, args):
        self.modality = 'vision'
        self.project_path = args.project_path
        self.device = args.device
        self.vision_samples = args.vision_samples
        self.patch_size = args.vision_patch_size
        self.reload = args.setup_models
        self.som_size = args.som_size

    def setup_model(self):
        if self.reload:
            save_dir = f"{self.project_path}/models/vision/saved_data"
            os.makedirs(save_dir, exist_ok=True)

            model = get_headless_resnet()
            val_dataset = get_data(self.vision_samples)

            torch.save(model.state_dict(), f"{save_dir}/resnet_model.pt")
            torch.save(val_dataset, f"{save_dir}/processed_images.pt")

            return model, val_dataset

    def setup_som(self):
        model = get_headless_resnet()
        model.load_state_dict(torch.load(f"{self.project_path}/models/vision/saved_data/resnet_model.pt"))
        model.eval()
        model.to('cpu')

        self.model = model
        self.sample_data = torch.load(f"{self.project_path}/models/vision/saved_data/processed_images.pt")
        self.norm_mu = self.sample_data.mean(dim=(0, 2, 3)).reshape(3, 1, 1)
        self.norm_max = torch.max(self.sample_data)

    def generate_static(self, coords):
        w1, w2 = coords
        p = self.patch_size
        static = torch.zeros_like(self.sample_data[0])
        static[:, w1:w1+p,w2:w2+p] = torch.rand_like(static[:, w1:w1+p,w2:w2+p]) #self.norm_max #(0.9 + 0.1*torch.rand_like(static[:, w1:w1+p,w2:w2+p])) #* self.norm_max
        static += self.norm_mu - static.mean()
        return (coords, static)

    def get_patches(self):
        patch_list = []
        x, y = self.sample_data[0][0].shape
        for i in range(0, x, self.patch_size):
            for j in range(0, y, self.patch_size):
                patch_list.append((i, j))
        return patch_list

    def calculate_activations(self, x):
        with torch.inference_mode():
            acts = self.model(x.unsqueeze(dim=0))
            return acts.squeeze().detach().cpu().flatten()
