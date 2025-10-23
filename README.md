# SEMD-LensedGW

This repository contains the official implementation of the **SEMD (Squeeze-and-Excitation Multilayer Perceptron Data-efficient Image Transformer)** model described in the paper:  
**“Identification of Strongly Lensed Gravitational Wave Events Using Squeeze-and-Excitation Multilayer Perceptron Data-efficient Image Transformer”**  
🔗 [https://arxiv.org/abs/2508.19311](https://arxiv.org/abs/2508.19311)

SEMD-LensedGW is a **fast pre-screening deep-learning framework** designed for identifying candidate strongly lensed gravitational-wave (GW) event pairs from time–frequency spectrograms.  
The model leverages cross-image attention to efficiently recognize morphological similarities between lensed GW signals, enabling rapid and scalable screening prior to Bayesian parameter estimation.

## Requirements
Tested with:
- Python 3.8+  
- PyTorch 1.10+ (or newer)
- NumPy, SciPy, scikit-learn
- librosa (for audio / spectrogram utilities, if needed)
- matplotlib (for plotting)
- tqdm
- pycbc

Create and activate a virtual environment, then install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

##  Repository Structure

```text
SEMD-LensedGW/
│
├── main.py
├── train.py
├── evaluate.py
├── predict_all.py
├── model.py
├── utils.py
├── checkpoints/
├── data/
│   ├── lensed/
│   └── unlensed/
└── README.md
```

## Data Preparation
⚠️ Due to privacy and licensing restrictions, data and data-generation code are not included in this repository.
To run this project, you need to prepare your own datasets:
- Place lensed and unlensed spectrogram pairs into the folder data/.
- Each sample should be formatted consistently with your data pipeline.
- Ensure both classes are properly labeled for supervised training.

## Quick Start
### Training
Run the following command to train a model:
```
python main.py --model_name your_model --lr 5e-5 --epochs 30 --batch_size 32
```
### Evaluation
After training, evaluate the model using:
```
python predict_all.py --data_root "data" --checkpoint checkpoints/your_model.pth
```
This script will compute classification metrics and generate visual outputs such as ROC curves and confusion matrices.

## NOTE
- This project focuses on pre-screening for lensed GW candidates — providing a fast, reliable filter before full Bayesian parameter estimation.
- Users are encouraged to adapt the training and evaluation scripts to their specific detector sensitivity curves (e.g., LIGO, ET, CE).
