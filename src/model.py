"""
Çoklu-görünüm (Sagittal/Coronal/Axial), çoklu-slice diz MR modeli.

Mimari fikri: her plandaki her slice, paylaşılan bir 2D CNN backbone'dan (ImageNet
pretrained, örn. efficientnet/resnet) geçirilir -> slice embedding'leri o plan içinde
attention-pooling ile birleştirilir -> 3 plan embedding'i birleştirilir (concat) ->
12 çıkışlı sınıflandırma başlığı (multi-label, sigmoid).

Not: Bu bir baseline mimari şablonudur; gerçek eğitimde backbone/pooling/augmentation
üzerinde iyileştirme yapılması beklenir (bkz. README "Yapılacaklar").
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

N_LABELS = 12
N_PLANES = 3


class SlicePooling(nn.Module):
    """Bir plandaki T slice embedding'ini attention-weighted ortalama ile tek vektöre indirger."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x):  # x: (B, T, dim)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, dim)


class KneeMultiViewModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True,
                 n_labels: int = N_LABELS):
        super().__init__()
        if timm is None:
            raise ImportError("Bu model timm kütüphanesi gerektirir: pip install timm")

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                           num_classes=0, in_chans=1)
        feat_dim = self.backbone.num_features

        self.slice_pool = SlicePooling(feat_dim)
        self.plane_norm = nn.LayerNorm(feat_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim * N_PLANES, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_labels),
        )

    def forward(self, x):  # x: (B, n_planes=3, T, H, W)
        B, P, T, H, W = x.shape
        x = x.view(B * P * T, 1, H, W)
        feats = self.backbone(x)  # (B*P*T, feat_dim)
        feats = feats.view(B, P, T, -1)

        plane_embeds = []
        for p in range(P):
            pooled = self.slice_pool(feats[:, p])  # (B, feat_dim)
            plane_embeds.append(self.plane_norm(pooled))

        combined = torch.cat(plane_embeds, dim=1)  # (B, feat_dim * n_planes)
        logits = self.head(combined)  # (B, n_labels)
        return logits
