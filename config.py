from dataclasses import dataclass

@dataclass
class Args:
    project_path: str = "."
    experiment_name: str = "main_results"
    device: str = 'cuda:0' # barely anything here needs gpu
    setup_models: bool = False # flag to train/download basemodels
    train_som: bool = True # flag to train SOM

    """SOM Parameters"""
    som_epochs: int = 100000 # training epochs, 100K seems to work well generally
    som_size: int = (25, 25) # SOM output size
    som_lr: float = 0.1 # SOM learning rate
    som_init: str = 'random' # alternatively set to 'pca'

    """Vision Params"""
    # som_vision: bool = True @TODELETE
    vision_samples: int = 100 # ImageNet samples to save, used to estimate image statistics
    vision_patch_size = 3 # pixel dimension of image patch, default 3x3

    """Touch Params"""
    # som_touch: bool = True @TODELETE
    touch_epochs: int = 30 # training epochs for touch base  model
    touch_frames: int = 1 # touch input is 3d, specify number of frames (picked 1 for speed)
    touch_samples: int = 100 # samples to save, used to estimate hand statistics
    touch_patch_size: int = 1 # default 1x1 patch size

    """Audio Params"""
    # som_audio: bool = True @TODELETE
    audio_patch_size: int = 1 # default 1 frequency channel
    audio_frames: int = 8000 # corresponds to half second

    """Olfaction Params"""
    # som_olfaction: bool = True @TODELETE
    olfaction_epochs: int = 15 # train epochs for olfaction base model
    olfaction_samples: int = 100 # samples to save
    olfaction_patch_size: int = 1 # som patch size
    olfaction_n_orn: int = 4096 # input dimension
    olfaction_n_class: int = 100 # number of classes to predict

    """Memory Params"""
    tem_patch_size: int = 1 # patch size for TEM input, corresponding to a single observational state
    tem_train = False # retrain TEM, takes 3-4 hours so probably false
    tem_n_rollouts = 100 # number of rollouts to train SOM over, with default settings each rollout = 42 samples
    tem_batch_size = 1 # number of batches to train SOM over
