import os
from huggingface_hub import hf_hub_url
url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="InstructScene.zip", repo_type="dataset")
os.system(f"wget {url} && unzip InstructScene.zip")
url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="3D-FRONT.zip", repo_type="dataset")
os.system(f"wget {url} && unzip 3D-FRONT.zip")

for file in [
    "bedroom_sg2scdiffusion_objfeat_epoch_01999.pth",
    "bedroom_sgdiffusion_vq_objfeat_epoch_01999.pth",
    "diningroom_sg2scdiffusion_objfeat_epoch_01999.pth",
    "diningroom_sgdiffusion_vq_objfeat_epoch_01239.pth",
    "livingroom_sg2scdiffusion_objfeat_epoch_01999.pth",
    "livingroom_sgdiffusion_vq_objfeat_epoch_01459.pth"
]:
    url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename=file, repo_type="dataset")
    os.system(f"wget {url} && mv {file} checkpoints/")

url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="objfeat_bounds.pkl", repo_type="dataset")
os.system(f"wget {url} && mv objfeat_bounds.pkl checkpoints/objfeat_bounds.pkl")

url = hf_hub_url(repo_id="chenguolin/InstructScene_dataset", filename="threedfront_objfeat_vqvae_epoch_01999.pth", repo_type="dataset")
os.system(f"wget {url} && mv threedfront_objfeat_vqvae_epoch_01999.pth checkpoints/epoch_01999.pth")