# ReCoGait
# ReCoGait

Official implementation of **ReCoGait: Reliability-Aware Cross-Modal Collaboration for Robust Silhouette--Skeleton Gait Recognition**.

ReCoGait is a dual-modal gait recognition framework that jointly exploits silhouette and skeleton-map representations. It improves cross-modal collaboration through Cross Gate Fusion (CGF), Temporal Gradient Interaction (TGI), and a Quality-Aware Loss for reliability-aware optimization.

## Dependency

This project is developed based on [OpenGait](https://github.com/ShiqiYu/OpenGait).

Please first configure OpenGait by following its official installation and dataset-preparation instructions. The ReCoGait implementation in this repository follows the OpenGait project structure. After OpenGait is configured, you can directly copy the provided files into the corresponding directories of an existing OpenGait installation, or use this repository as an OpenGait-based project.

## Repository Structure

```text
ReCoGait/
├── configs/
│   └── recogait/
│       └── recogait_SUSTech1K.yaml
├── datasets/
│   └── SUSTech1K/
│       └── SUSTech1K.json
├── opengait/
│   ├── modeling/
│   │   ├── models/
│   │   │   └── recogait.py
│   │   └── losses/
│   │       └── quality_aware_triplet_loss.py
│   └── ...
├── train.sh
├── test.sh
└── README.md
```

## Method Overview

ReCoGait contains three main components:

- **Cross Gate Fusion (CGF):** Generates modality-specific gates and performs bidirectional cross-modal feature recalibration between silhouette and skeleton-map streams.
- **Temporal Gradient Interaction (TGI):** Encodes local backward and forward temporal feature differences to enhance motion-sensitive representations.
- **Quality-Aware Loss ($\mathcal{L}_{\mathrm{qa}}$):** Uses estimated skeleton-map reliability to adaptively scale the auxiliary triplet margin.

The overall training objective is:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{tri}}
+
\mathcal{L}_{\mathrm{ce}}
+
\mathcal{L}_{\mathrm{qa}}.
```

## Training

Before training, update the dataset path and other environment-specific settings in `configs/recogait/recogait_SUSTech1K.yaml`.

You can launch training with:

```bash
bash train.sh
```

## Evaluation

To evaluate a trained checkpoint, update the restore settings in the YAML configuration file and run:

```bash
bash test.sh
```

