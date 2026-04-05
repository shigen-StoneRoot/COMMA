# COMMA: Coordinate-aware Modulated Mamba Network for 3D Dispersed Vessel Segmentation


This repository provides the official implementation of **COMMA**, a coordinate-aware modulated Mamba network for **3D dispersed vessel segmentation**.

Our implementation is built upon [U-Mamba](https://github.com/bowang-lab/U-Mamba.git). Please first follow the official U-Mamba instructions to set up the base environment and dependencies.

---

## News

- **[2025-03]** COMMA is released on arXiv.
- **[2025-03]** The official codebase is released.
- **[2026-04]** Pre-trained model weights are available.
- More updates will be added here.

---

## Resources

- **Pre-trained model weights**:  
  https://drive.google.com/drive/folders/1Dojb-3JmSg4W-CgPWFSvoAG62HcBRR-b?usp=sharing

- **IXI vessel segmentation annotations**:  
  https://drive.google.com/file/d/16a05rBkV29iUkNkVXWpTOaYCkencfEVu/view?usp=sharing

- **IXI image data**:  
  http://brain-development.org/ixi-dataset/

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

## Citation

If you find this work helpful in your research, please consider citing:

```
@article{shi2025comma,
  title={COMMA: Coordinate-aware Modulated Mamba Network for 3D Dispersed Vessel Segmentation},
  author={Shi, Gen and Zhang, Hui and Tian, Jie},
  journal={arXiv preprint arXiv:2503.02332},
  year={2025}
}
```


