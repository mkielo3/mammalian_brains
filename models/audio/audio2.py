import torch
import torchaudio
from torchaudio.utils import download_asset
import os
import numpy as np
import librosa

from ..base.trainer import ModalityTrainer


# def spec2wav(mel_spec):
#     # print ("spec2wav in", mel_spec.shape)
#     audio = librosa.feature.inverse.mel_to_audio(
#         np.array(mel_spec),
#         sr=16000,
#         n_fft=2048,
#         hop_length=1024
#     )
#     # print ("spec2wav out", audio.shape)
#     return torch.Tensor(audio)


# def wav2spec(waveform):
#     # print ("wav2spec in", waveform.shape)
#     output = torch.tensor(librosa.feature.melspectrogram(
#         y=np.array(waveform),
#         sr=16000,
#         n_fft=2048,
#         hop_length=1024,
#         power=2.0
#     ))
#     print (output.shape)
#     # assert (False)
#     # print ("wav2spec out", output.shape)
#     return output


import librosa
import torch
import numpy as np

def wav2spec(waveform):
    """
    Convert waveform to Mel-spectrogram.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=np.array(waveform),
        sr=16000,
        n_fft=2048,       # Ensure consistent n_fft for analysis and synthesis
        hop_length=512,   # Standard hop length for better time resolution
        n_mels=1024,      # Number of mel bins
        power=2.0         # Power spectrogram
    )
    return torch.tensor(mel_spec)

def spec2wav(mel_spec):
    """
    Convert Mel-spectrogram back to waveform.
    """
    audio = librosa.feature.inverse.mel_to_audio(
        np.array(mel_spec),
        sr=16000,
        n_fft=2048,       # Match n_fft used in wav2spec
        hop_length=512    # Match hop length used in wav2spec
    )
    return torch.tensor(audio)


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
        print (data)
        data = wav2spec(data)[0]
        print (data.shape)
        self.norm_mu = data.mean()
        self.norm_max = data.max()
        self.sample_data = data

    def generate_static(self, coords):
        w = coords[0]
        static = torch.zeros_like(self.sample_data)
        static[w:w+self.patch_size,:] = 11*torch.rand_like(static[0,:])+11 # * 11 + 11 #11 + 11 #(0.9 + 0.1*torch.rand_like(static[0,:])) * self.norm_max
        # static += self.norm_mu - static.mean()
        # if norm == 1:
            # static[w:w+self.patch_size,:] = torch.rand_like(static[w:w+self.patch_size,:]) * 800 + 800 #11 + 11 #(0.9 + 0.1*torch.rand_like(static[0,:])) * self.norm_max
        # elif norm == 2:
        # static[w:w+self.patch_size,:] = torch.rand_like(static[w:w+self.patch_size,:])*(self.norm_max/2) + self.norm_max #11 + 11 #(0.9 + 0.1*torch.rand_like(static[0,:])) * self.norm_max
        # static[w:w+self.patch_size,:] += self.norm_mu - static[w:w+self.patch_size,:].mean()
        # print (static.mean())
        static = torch.clamp(static, min=0)  
        static_waveform = spec2wav(static)
        return static_waveform

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