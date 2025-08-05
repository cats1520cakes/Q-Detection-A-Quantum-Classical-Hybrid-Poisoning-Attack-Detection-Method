# Q-Detection Repository

**Q-Detection: A Quantum–Classical Hybrid Poisoning Attack Detection Method**  
IJCAI 2025 Accepted Paper

---

## 🚀 Project Overview

This repository implements and benchmarks several poison-sample detection algorithms for deep-learning models under three canonical poisoning scenarios:

- **Label-Flipping**  
- **Narcissus**  
- **BadNets**

We compare:

- **DCM**  
- **LossScan**  
- **AutoencoderScan**  
- **Meta-Sift**  
- **Q-Detection** (our proposed quantum-accelerated method)

**Metrics**  
- **NCR**: lower is better (ideal 0 %, random ≈ 100 %)  
- **Post-filter Accuracy**: test accuracy (overall and target-class) after filtering

---

## 🔧 Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the GTSRB dataset archive
   Place `gtsrb_dataset.h5` under `./dataset/` (see link below).


## ⚙️ Configuration

Edit the hyperparameters in `main.py` (or pass via CLI if you extend):

| Parameter    | Description                               | Example                                                 |
| ------------ | ----------------------------------------- | ------------------------------------------------------- |
| `methodname` | Attack type                               | `"targeted_label_flipping"`, `"narcissus"`, `"badnets"` |
| `rate`       | Poisoning rate (%)                        | `3`, `5`, `10`, `20`, `30`                              |
| `subset_num` | Number of samples to keep after filtering | `4000` (default)                                        |

Download link for the dataset:
[https://drive.google.com/file/d/1SKYMwrnjEyFjjc7UWTdAyAjFI\_demNtD/view](https://drive.google.com/file/d/1SKYMwrnjEyFjjc7UWTdAyAjFI_demNtD/view)

---

## ▶️ Usage

Run the main script directly:

```bash
python main.py
```

This will:

1. Load the specified poisoning scenario and rate
2. Apply each detection method
3. Print NCR and post-filter test accuracies

---

## 📦 Dependencies

```text
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=1.0.2
Pillow>=8.3.2
h5py>=3.2.1
tqdm>=4.64.0
dimod>=0.12.1
neal>=0.5.8
dwave-system>=1.17.0
```

*(Tested with Python ≥ 3.8), Windows and Linux.*

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@article{he2025q,
  title={Q-Detection: A Quantum-Classical Hybrid Poisoning Attack Detection Method},
  author={He, Haoqi and Lin, Xiaokai and Chen, Jiancai and Xiao, Yan},
  journal={arXiv preprint arXiv:2507.06262},
  year={2025}
}
```

---

## 🔗 Links

* **Paper (arXiv)**: [https://arxiv.org/abs/2507.06262](https://arxiv.org/abs/2507.06262)
* **Dataset (GTSRB)**: [https://drive.google.com/file/d/1SKYMwrnjEyFjjc7UWTdAyAjFI\_demNtD/view](https://drive.google.com/file/d/1SKYMwrnjEyFjjc7UWTdAyAjFI_demNtD/view)


