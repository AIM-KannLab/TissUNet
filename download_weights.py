import os
import shutil
import gdown
from tqdm import tqdm

local_remote_mapping = {
    'nnUNet_results/Dataset003_synthrad/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth': 'https://drive.google.com/file/d/1h8_oKmNWd-zdoVZLiEaAlcZj-Ss8L-Ie/view?usp=drive_link',
    'nnUNet_results/Dataset003_synthrad/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth': 'https://drive.google.com/file/d/1zYr5iJr8N1wda9Higa5LOCOPUUsuU3jQ/view?usp=drive_link',
    'nnUNet_results/Dataset003_synthrad/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset_fingerprint.json': 'https://drive.google.com/file/d/17mvHMLKfHVhXtO0Lj9JS7UIr3yb4PCF7/view?usp=drive_link',
    'nnUNet_results/Dataset003_synthrad/nnUNetTrainer__nnUNetPlans__3d_fullres/dataset.json': 'https://drive.google.com/file/d/1_6GXtxtM5o5sG6M95wGcM8T__3zyulLD/view?usp=drive_link',
    'nnUNet_results/Dataset003_synthrad/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json': 'https://drive.google.com/file/d/1Ia5SNyaTyN6PSe4zAFd-f1AiTh4q5sVZ/view?usp=drive_link',
}

if os.path.exists('nnUNet_results'):
    shutil.rmtree('nnUNet_results')
    
for local, remote in tqdm(local_remote_mapping.items()):
    os.makedirs(os.path.dirname(local), exist_ok=True)
    gdown.download(remote, local, quiet=True, fuzzy=True)