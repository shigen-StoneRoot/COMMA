The environment setup is based on U-Mamba (https://github.com/bowang-lab/U-Mamba.git). Please follow the installation instructions provided in the repository to set up the environment.

The open-source data of vessel segmentation for the IXI dataset can be found at: https://drive.google.com/file/d/16a05rBkV29iUkNkVXWpTOaYCkencfEVu/view?usp=sharing

The image data can be downloaded at: http://brain-development.org/ixi-dataset/

## Installation 

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment: `conda create -n umamba python=3.10 -y` and `conda activate umamba `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/bowang-lab/U-Mamba`
5. `cd U-Mamba/umamba` and run `pip install -e .`


sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Data Preparation

### 1. Preprocessing

Place your raw data under `Data/raw_data/` in the following format (example: KiPA):

```
Data/raw_data/KiPA/
├── images/
│   ├── subject1.nii.gz
│   ├── ...
├── masks/
│   ├── subject1.nii.gz
│   ├── ...
```

Then create the necessary folders:

```bash
mkdir ./CKs
mkdir ./Prediction
mkdir ./Data/preprocessed_data
```

Run:

```bash
python preprocessing.py
```

---

### 2. Training

```bash
python main.py
```


### 3. Evaluation

```bash
python eval_save.py
```

---

## Contact

If you encounter any issues, feel free to contact:  
📧 **shigen@buaa.edu.cn**
