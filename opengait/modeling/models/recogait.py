import copy

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..base_model import BaseModel
from ..modules import (
    BasicBlock2D,
    BasicBlockP3D,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
    conv1x1,
    conv3x3,
)


class ReCoGait(BaseModel):
    """
    ReCoGait: Reliability-Aware Cross-Modal Collaboration for
    Robust Silhouette--Skeleton Gait Recognition.

    Input modalities:
        - skeleton maps:  [N, 2, T, H, W]
        - silhouettes:    [N, 1, T, H, W]

    Main components:
        - CrossGateFusion (CGF)
        - TemporalGradientInteraction (TGI)
        - Skeleton-reliability scores for the auxiliary quality-aware loss
    """

    def build_network(self, model_cfg):
        blocks = model_cfg["Backbone"]["blocks"]
        base_channels = model_cfg["Backbone"]["C"]
        self.inference_use_emb = model_cfg.get("use_emb2", False)

        self.inplanes = 32 * base_channels

        self.silhouette_stem = SetBlockWrapper(
            nn.Sequential(
                conv3x3(1, self.inplanes, 1),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
            )
        )
        self.skeleton_stem = SetBlockWrapper(
            nn.Sequential(
                conv3x3(2, self.inplanes, 1),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
            )
        )

        self.silhouette_encoder = SetBlockWrapper(
            self.make_layer(
                BasicBlock2D,
                32 * base_channels,
                stride=[1, 1],
                blocks_num=blocks[0],
                mode="2d",
            )
        )
        self.skeleton_encoder = copy.deepcopy(self.silhouette_encoder)

        self.cross_gate_fusion = CrossGateFusion(32 * base_channels)

        self.layer2 = self.make_layer(
            BasicBlockP3D,
            64 * base_channels,
            stride=[2, 2],
            blocks_num=blocks[1],
            mode="p3d",
        )
        self.layer3 = self.make_layer(
            BasicBlockP3D,
            128 * base_channels,
            stride=[2, 2],
            blocks_num=blocks[2],
            mode="p3d",
        )

        stage3_channels = 128 * base_channels * BasicBlockP3D.expansion
        self.temporal_gradient_interaction = TemporalGradientInteraction(
            channels=stage3_channels
        )

        self.layer4 = self.make_layer(
            BasicBlockP3D,
            256 * base_channels,
            stride=[1, 1],
            blocks_num=blocks[3],
            mode="p3d",
        )

        self.temporal_pooling = PackSequenceWrapper(torch.max)
        self.horizontal_pooling = HorizontalPoolingPyramid(bin_num=[16])
        self.part_fcs = SeparateFCs(16, 256 * base_channels, 128 * base_channels)
        self.bn_necks = SeparateBNNecks(
            16,
            128 * base_channels,
            class_num=model_cfg["SeparateBNNecks"]["class_num"],
        )

    def make_layer(self, block, planes, stride, blocks_num, mode="2d"):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == "3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=[1, 1, 1],
                        stride=stride,
                        padding=[0, 0, 0],
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            elif mode == "2d":
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            elif mode == "p3d":
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=[1, 1, 1],
                        stride=[1, *stride],
                        padding=[0, 0, 0],
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                raise ValueError(f"Unsupported layer mode: {mode}")
        else:
            downsample = lambda x: x

        layers = [
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes * block.expansion

        residual_stride = [1, 1] if mode in ("2d", "p3d") else [1, 1, 1]
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=residual_stride))

        return nn.Sequential(*layers)

    def inputs_pretreament(self, inputs):
        """
        Preserve identical augmentation and spatial alignment for skeleton maps
        and silhouettes before delegating batching/padding to BaseModel.
        """
        paired_sequences = inputs[0]
        merged_sequences = []

        for skeleton_map, silhouette in zip(
            paired_sequences[0], paired_sequences[1]
        ):
            silhouette = silhouette[:, np.newaxis, ...]  # [T, 1, H, W]

            map_height, map_width = skeleton_map.shape[-2:]
            silhouette_height, silhouette_width = silhouette.shape[-2:]

            if silhouette_height != silhouette_width and map_height == map_width:
                crop_margin = (silhouette_height - silhouette_width) // 2
                skeleton_map = skeleton_map[..., crop_margin:-crop_margin]

            merged_sequences.append(
                np.concatenate([skeleton_map, silhouette], axis=1)
            )  # [T, 3, H, W]

        merged_inputs = [
            [merged_sequences],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
        ]
        return super().inputs_pretreament(merged_inputs)

    @staticmethod
    def estimate_skeleton_reliability(skeleton_maps):
        """
        Estimate the sequence-level skeleton reliability score used by the
        auxiliary Quality-Aware Loss (L_qa).

        Args:
            skeleton_maps: [N, 2, T, H, W]

        Returns:
            normalized_scores: [N], normalized within the current mini-batch.
        """
        raw_scores = skeleton_maps.mean(dim=(1, 2, 3, 4))
        return (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-6
        )

    def forward(self, inputs):
        input_tensor, labels, _, _, seqL = inputs

        # BaseModel provides [N, T, C, H, W]; convert to [N, C, T, H, W].
        input_tensor = input_tensor.transpose(1, 2).contiguous()
        assert input_tensor.size(-1) in [44, 48, 64, 88, 96]

        skeleton_maps = input_tensor[:, :2, ...]
        silhouettes = input_tensor[:, 2:3, ...]
        skeleton_reliability = self.estimate_skeleton_reliability(skeleton_maps)

        skeleton_features = self.skeleton_stem(skeleton_maps)
        skeleton_features = self.skeleton_encoder(skeleton_features)

        silhouette_features = self.silhouette_stem(silhouettes)
        silhouette_features = self.silhouette_encoder(silhouette_features)

        fused_features = self.cross_gate_fusion(
            silhouette_features,
            skeleton_features,
        )

        stage2_features = self.layer2(fused_features)
        stage3_features = self.layer3(stage2_features)
        refined_stage3_features = self.temporal_gradient_interaction(
            stage3_features
        )
        stage4_features = self.layer4(refined_stage3_features)

        pooled_features = self.temporal_pooling(
            stage4_features,
            seqL,
            options={"dim": 2},
        )[0]
        part_features = self.horizontal_pooling(pooled_features)

        embedding_before_bn = self.part_fcs(part_features)
        embedding_after_bn, logits = self.bn_necks(embedding_before_bn)

        embedding = (
            embedding_after_bn
            if self.inference_use_emb
            else embedding_before_bn
        )

        return {
            "training_feat": {
                "triplet": {
                    "embeddings": embedding_before_bn,
                    "labels": labels,
                },
                "softmax": {
                    "logits": logits,
                    "labels": labels,
                },
                "quality_aware": {
                    "embeddings": embedding_before_bn,
                    "labels": labels,
                    "quality_scores": skeleton_reliability,
                },
            },
            "visual_summary": {
                "image/inputs": rearrange(
                    input_tensor * 255.0,
                    "n c t h w -> (n t) c h w",
                ),
            },
            "inference_feat": {
                "embeddings": embedding,
            },
        }


class CrossGateFusion(nn.Module):
    """
    Cross Gate Fusion (CGF).

    Gates generated from skeleton features recalibrate silhouette features,
    while gates generated from silhouette features recalibrate skeleton features.
    """

    def __init__(self, channels):
        super().__init__()

        self.skeleton_to_silhouette_gate = SetBlockWrapper(
            nn.Sequential(
                conv1x1(channels, channels),
                nn.BatchNorm2d(channels),
                nn.Sigmoid(),
            )
        )
        self.silhouette_to_skeleton_gate = SetBlockWrapper(
            nn.Sequential(
                conv1x1(channels, channels),
                nn.BatchNorm2d(channels),
                nn.Sigmoid(),
            )
        )
        self.fusion_projection = SetBlockWrapper(
            nn.Sequential(
                conv1x1(channels * 2, channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        )

    def forward(self, silhouette_features, skeleton_features):
        skeleton_gate = self.skeleton_to_silhouette_gate(skeleton_features)
        recalibrated_silhouette = silhouette_features * skeleton_gate

        silhouette_gate = self.silhouette_to_skeleton_gate(silhouette_features)
        recalibrated_skeleton = skeleton_features * silhouette_gate

        fused_features = torch.cat(
            [recalibrated_silhouette, recalibrated_skeleton],
            dim=1,
        )
        return self.fusion_projection(fused_features)


class TemporalGradientInteraction(nn.Module):
    """
    Temporal Gradient Interaction (TGI).

    Encodes backward and forward first-order feature differences and refines
    the original feature sequence through a residual connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        previous_features = torch.cat(
            [features[:, :, :1, :, :], features[:, :, :-1, :, :]],
            dim=2,
        )
        next_features = torch.cat(
            [features[:, :, 1:, :, :], features[:, :, -1:, :, :]],
            dim=2,
        )

        backward_difference = features - previous_features
        forward_difference = next_features - features

        temporal_features = torch.cat(
            [features, backward_difference, forward_difference],
            dim=1,
        )
        refined_features = self.refinement(temporal_features)

        return features + refined_features
