# StreetSense: Edge‑First Multimodal Road-Defect Mapper

StreetSense is a portfolio-grade AIML project that detects and prioritizes road defects (e.g., potholes, cracks, speed bumps) using **smartphone sensors**: microphone (audio), IMU (accelerometer & gyroscope), and the camera. It is designed to be unique by combining:

- **Self-supervised pretraining** (SimCLR-style) for audio & IMU windows.
- **Lightweight vision backbone** (MobileNetV3-like) for frames.
- **Transformer-based sensor fusion** for sequence alignment across modalities.
- **Active learning** to pick the most informative trips for annotation.
- **Causal prioritization** (Difference-in-Differences / CausalImpact-like) to rank fixes by estimated safety/comfort gains.
- **Privacy-first** simulation via federated averaging for on-device models.

> This repository includes **runnable code on synthetic data** so you can train, evaluate, and demo end-to-end without special hardware.

## Quickstart

```bash
# 1) Create virtual environment (optional) and install deps
pip install -r requirements.txt

# 2) Generate synthetic multimodal dataset
python streetsense/data/make_synthetic.py --n_trips 30 --seed 42

# 3) Pretrain encoders (self-supervised) on unlabeled windows
python streetsense/train_selfsup.py --epochs 2

# 4) Fine-tune fusion model (supervised) on labeled subset
python streetsense/train_supervised.py --epochs 2

# 5) Run active learning round
python streetsense/active_learning.py --budget 50

# 6) Score new trips and export a prioritized fix list
python streetsense/prioritize_causal.py --export out/prioritized_repairs.csv
```

All scripts run in minutes on CPU with the synthetic dataset.

## Project Structure

- `streetsense/data/` — data schema, synthetic generation, loaders
- `streetsense/models/` — encoders, fusion transformer, heads
- `streetsense/selfsup/` — SimCLR-style losses & views
- `streetsense/train_selfsup.py` — self-supervised pretraining
- `streetsense/train_supervised.py` — supervised fine-tuning
- `streetsense/active_learning.py` — entropy-based sampling
- `streetsense/prioritize_causal.py` — simple diff-in-diff prioritization
- `cli.py` — unified command-line interface
- `notebooks/demo.ipynb` — walkthrough (optional placeholder)


## Datasets (Real-World Options to Extend)

- Road surface datasets (e.g., pothole imagery from public datasets)
- Your own smartphone recordings while biking/driving safe routes (respect local laws)
- City open data for **accident/complaint** rates for causal evaluation

## License

MIT for code. You are responsible for data collection, consent, and compliance in your region.
