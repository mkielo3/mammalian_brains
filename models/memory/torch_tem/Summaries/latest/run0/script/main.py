from models.olfaction import olfaction
from models.vision import vision
from models.touch import touch
from models.audio import audio2
from dataclasses import dataclass



@dataclass
class Args:
    project_path: str = "/home/gildroid/workspace2024/mammalian_brains"
    device: str = 'cpu'
    setup_models: bool = True # Flag to train/download basemodels
    train_som: bool = True

    """SOM Parameters"""
    som_epochs: int = 100000
    som_shape: int = (25, 25)

    """Vision Params"""
    setup_vision: bool = True
    som_vision: bool = True
    vision_samples: int = 100
    vision_patch_size = 3

    """Touch Params"""
    setup_touch: bool = True
    som_touch: bool = True
    touch_epochs: int = 30
    touch_frames: int = 1
    touch_samples: int = 100
    touch_patch_size: int = 1

    """Audio Params"""
    setup_audio: bool = True
    som_audio: bool = True
    audio_patch_size = 1

    """Olfaction Params"""
    setup_olfaction: bool = True
    som_olfaction: bool = True
    olfaction_epochs: int = 15 # train epochs
    olfaction_samples: int = 100 # samples to save
    olfaction_patch_size = 1 # som patch size
    olfaction_n_orn = 4096 # input dimension
    olfaction_n_class = 100 # smell classes to predict





def main():
    args = Args()
    args.device = 'cuda:0'

    if args.setup_models:

        if args.setup_olfaction:
            olfaction.prep_olfaction(args)

        if args.setup_touch:
            touch.prep_touch(args)

        if args.setup_vision:
            vision.prep_vision(args)

        if args.setup_audio:
            audio2.prep_audio(args)

    if args.train_som:

        modality = audio2
        activation_list = []
        model, sample_data = modality.load_model()
        patches = modality.get_patch_list()
        for p in patches:
            static = modality.generate_static(sample_data, p)
            activation = modality.generate_activation(model, static)
    
        # get activations
        # train som
        # save



if __name__ == "__main__":
    main()