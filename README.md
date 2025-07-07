# QDetection

## Project Overview

This repository contains comparative experiments on **poison-sample detection for deep-learning models**.
We evaluate several algorithms—**DCM**, **LossScan**, **AutoencoderScan**, **Meta-Sift**, and the proposed **Q-Detection**—under three canonical poisoning scenarios (label-flipping, Narcissus, BadNets).
Performance is reported with two primary metrics:

* **NCR (Normalized Contamination Rate)** – lower is better (ideal 0 %, random baseline ≈ 100 %);
* **Post-filter Accuracy** – overall test accuracy and target-label accuracy after removing suspicious samples.

Q-Detection, a quantum-accelerated method, is shown to achieve the best balance between NCR and accuracy.


## Core Features

* **Algorithm Benchmarks** – DCM, LossScan, AutoencoderScan, Q-Detection, Meta-Sift.
* **Attack Coverage** – targeted label-flipping, Narcissus, and BadNets.
* **Flexible Hyper-parameters** – poisoning rate (3 % – 30 %), number of samples kept (`subset_num`), etc.

## Experimental Configuration

| Parameter    | Meaning                          | Example                                                   |
| ------------ | -------------------------------- | --------------------------------------------------------- |
| `methodname` | Attack type                      | `"targeted_label_filpping"` / `"narcissus"` / `"badnets"` |
| `rate`       | Poisoning rate (%)               | `3`, `5`, `10`, `20`, `30`                                |
| `subset_num` | # samples selected by the filter | `4000` (default)                                          |

## How to Run

1. **Set up the environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   Download dataset gtsrb_dataset.h5 form this link https://drive.google.com/file/d/1SKYMwrnjEyFjjc7UWTdAyAjFI_demNtD/view and put it under './dataset' folder.
    
2. **Adjust hyper-parameters** in `main.py` (`methodname`, `rate`, `subset_num`).
3. **Launch** via an IDE run of `main.py` or:

   ```bash
   python main.py
   ```

   The script prints each method’s NCR and the accuracy of the model retrained on the filtered dataset.

## Dependencies

Add the following to **requirements.txt** (tested with Python ≥ 3.8):

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=1.0.2
Pillow>=8.3.2
tqdm>=4.64.0
h5py>=3.2.1               # for HDF5 datasets
dimod>=0.12.1             # binary quadratic models
neal>=0.5.8               # simulated annealing sampler
dwave-system>=1.17.0      # optional: run on real QA hardware
kaiwu>=0.1.0              # Kaiwu SDK for CIM simulation
matplotlib>=3.4.3         # plotting (optional)
```# Q-Detection-A-Quantum-Classical-Hybrid-Poisoning-Attack-Detection-Method
