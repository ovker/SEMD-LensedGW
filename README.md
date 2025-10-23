# SEMD-LensedGW

**SEMD-LensedGW** — a fast pre-screening deep-learning framework (SE + MLP + DeiT) for identifying candidate strongly-lensed gravitational-wave event pairs from time–frequency spectrograms.

This repository contains the model code and training / evaluation scripts for the SEMD classifier.  
**Note:** Due to privacy / size constraints, simulated data and the data-generation scripts are *not* included. Users must provide their own prepared datasets (format described below).

---

## Features
- Lightweight DeiT-based backbone with Squeeze-and-Excitation and MLP heads (SEMD).
- Train / validate / test pipelines with configurable hyperparameters.
- Fast inference suitable for pre-screening large numbers of event pairs.
- Simple CLI for training and batch testing.

---

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

                
├── model.py              
├── utils.py
├── evaluate.py
├── train.py
├── predict_all.py
