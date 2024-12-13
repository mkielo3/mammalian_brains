import torch
import torchaudio
from torchaudio.utils import download_asset
import os
import numpy as np

from ..base.trainer import ModalityTrainer


def spec2wav(n=4096): # 1024
    return torchaudio.transforms.GriffinLim(
        n_fft=n,
        win_length=None,
        hop_length=160,
        power=2.0,
        n_iter=100
    )


def wav2spec(n=4096): # 1024
    return torchaudio.transforms.Spectrogram(
        n_fft=n,
        win_length=None,
        hop_length=160,
        power=2.0
    )


class Audio(ModalityTrainer): 
    
    def __init__(self, args):
        self.modality = 'audio'
        self.project_path = args.project_path
        self.patch_size = args.audio_patch_size
        self.reload = args.setup_models
        self.som_size = args.som_size
        self.frames = args.audio_frames

    def setup_model(self):
        if self.reload:
            # Create directory if it doesn't exist
            save_dir = f"{self.project_path}/models/audio/saved_data"
            os.makedirs(save_dir, exist_ok=True)

            SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
            waveform, sample_rate = torchaudio.load(SPEECH_FILE)
            waveform = waveform[:,10000:10000+self.frames] # shifted to avoid quiet
            # waveform = waveform[:,:self.frames] # shifted to avoid quiet

            # Get and save model
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            model = bundle.get_model()
            model.eval()

            # Save files
            torch.save(model.state_dict(), f"{save_dir}/wav2vec2_model.pt")
            torch.save(waveform, f"{save_dir}/waveform.pt")
            torch.save(torch.tensor(sample_rate), f"{save_dir}/sample_rate.pt")
            return model, waveform, sample_rate

    def setup_som(self):
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()
        model.load_state_dict(torch.load(f"{self.project_path}/models/audio/saved_data/wav2vec2_model.pt"))
        model.eval()
        self.model = model

        data = torch.load(f"{self.project_path}/models/audio/saved_data/waveform.pt").to('cpu')
        data = wav2spec()(data[0])
        self.norm_mu = data.mean()
        self.norm_max = data.max()
        self.sample_data = data

    def generate_static(self, coords):
        w = coords[0]
        static = torch.zeros_like(self.sample_data)
        static[w:w+self.patch_size,:] = torch.rand_like(static[w:w+self.patch_size,:])*11 + 11 #* 100 + 100 #11 + 11 #(0.9 + 0.1*torch.rand_like(static[0,:])) * self.norm_max
        static_waveform = spec2wav()(static)
        return (coords, static_waveform)

    def get_patches(self):
        return [(x, 0) for x in range(0, self.sample_data.shape[0], self.patch_size)]

    def calculate_activations(self, x):
        # spec2wav is the slow part, so we dont bother with cuda
        with torch.inference_mode():
            x, _ = self.model.feature_extractor(x.unsqueeze(dim=0), None)
            x = self.model.encoder.feature_projection(x)
            return x.flatten().detach().cpu()
   
    def initialize_som(self, SOM):
        return SOM(som_size=self.som_size, is_torus=False, balance=1)