# model.py

import torch.nn as nn
import timm


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

def get_deit_tiny_distilled_enhanced(
    num_classes=2,
    pretrained=True,
    hidden_dim=512,
    dropout_rate=0.5,
    freeze_backbone=False
):

    model = timm.create_model(
        'deit_tiny_distilled_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    embed_dim = model.embed_dim


    if freeze_backbone:
        for p in model.parameters(): p.requires_grad = False


    enhanced_head = nn.Sequential(

        nn.LayerNorm(embed_dim),
        SEBlock1D(embed_dim, reduction=4),
        nn.Dropout(dropout_rate),
        nn.Linear(embed_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_classes)
    )
    enhanced_dist = nn.Sequential(
        nn.LayerNorm(embed_dim),
        SEBlock1D(embed_dim, reduction=4),
        nn.Dropout(dropout_rate),
        nn.Linear(embed_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, num_classes)
    )

    model.head = enhanced_head
    model.head_dist = enhanced_dist

    return model
