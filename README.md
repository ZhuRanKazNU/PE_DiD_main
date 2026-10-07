# PE-DiD

PyTorch / Python reproduction of the paper *Estimating the Effect of Physical Education Reform on College Students’ Mental Health with Difference-in-Differences*.


## Repository layout

```
PE_DiD_main/
  main.py
  requirements.txt
  configs/default.yaml
  src/
    data/          
    models/        
    train.py
    test.py
    infer.py
    estimate.py    
  data/raw/        
  data/processed/  
  results/tables/
  results/figures/
  checkpoints/
```

## Installation

```bash
cd PE_DiD_main
pip install -r requirements.txt
```

Python 3.10+ is required. GPU is optional (`device: cuda` in `configs/default.yaml`).

## Usage

From this directory:

```bash
# 1) train PSM, DID, PSM-DID, and mediation networks
python main.py train

# 2) test
python main.py test

# 3) inference
python main.py infer
python main.py infer --student_id S0001

# 4)full pipeline
python main.py run_all
```


## Configuration

All sample sizes, Table 2 targets, DGP effects, matching caliper, and training hyperparameters live in `configs/default.yaml`.
