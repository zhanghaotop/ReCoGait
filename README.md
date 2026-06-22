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

<p align="center">
  <img src="assets/results.png" width="80%" alt="Comparison of ReCoGait with representative gait recognition methods across different datasets and modality settings.">
</p>

<p align="center">
  <em>Comparison of ReCoGait with representative gait recognition methods across different datasets and modality settings. The figure highlights the performance differences among single-modal, alternative multimodal, and silhouette--skeleton collaborative approaches.</em>
</p>

ReCoGait is designed around a single objective: maintaining effective silhouette--skeleton collaboration when the two modalities are affected by different forms of degradation. Given paired silhouette sequences and dense skeleton heat maps, the framework first extracts modality-specific features in a shared representation space. It then uses cross-modal response patterns to recalibrate the complementary stream, allowing silhouette and skeleton cues to interact before deeper spatiotemporal modeling.

The fused representation is subsequently processed by the backbone, where local bidirectional temporal differences are incorporated to enrich motion-sensitive features. During training, ReCoGait further estimates sequence-level skeleton reliability from the input heat maps and uses this estimate to scale an auxiliary triplet margin. In this way, reliable skeleton sequences receive stronger auxiliary discriminative supervision, while lower-reliability pairs contribute a weaker auxiliary constraint.

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

