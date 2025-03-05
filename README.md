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

1. Preprocess the Data: Run the preprocessing step with:
```
python preprocessing.py
```

2. Train the Model: Start the training procedure by running:
```
python main.py
```

3. Evaluate the Model: Finally, perform the evaluation procedure with:
```
python eval_save.py
```

If you have any problem, please email me with this address: shigen@buaa.edu.cn
