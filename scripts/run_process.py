import os
import subprocess

# Imposta la variabile d'ambiente per CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Lancia lo script
subprocess.run(["bash", "scripts/mp3d.sh", "configs/mp3d_gaussian_FR_eccv.yaml"])