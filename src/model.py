"""
Çoklu-görünüm (study'deki TÜM MR serileri, max_views'e kadar), çoklu-slice diz MR modeli.

Mimari: her görünümdeki (seri) her slice, paylaşılan bir 2D CNN backbone'dan (ImageNet
pretrained) geçirilir -> slice embedding'leri o g�örünüm içinde attention-pooling ile
birleştirilir -> görünüm embedding'leri, mask kullanılarak (padding'i yoksayan) bir
attention pooling ile TEK bir study embedding'ine indirgenir -> 12 çıkışlı sınıflandırma
başlığı (multi-label, sigmoid).

Backbone büyüklüğü configs/*.yaml üzerinden seçilir (örn. efficientnet_b0 -> b3) --
ensemble için birden fazla config ile ayrı ayrı eğitilip tahminleri submission
notebook'unda ortalanır.
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

N_LABELS = 12


class AttnPooling(nn.Module):
    """Bir dizi embedding'i (T, dim) attention-weighted ortalama ile tek vektöre indirger.
    mask verilirse (B, T), padding konumları -inf skorla maskelenip softmax'a girmez."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, mask=None):  # x: (B, T, dim), mask: (B, T) veya None
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
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

        self.slice_pool = AttnPooling(feat_dim)   # slice'lar -> tek seri embedding'i
        self.view_pool = AttnPooling(feat_dim)    # seriler (view'lar) -> tek study embedding'i
        self.norm = nn.LayerNorm(feat_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_labels),
        )

    def forward(self, x, view_mask=None):
        # x: (B, V, T, H, W)  -- V=max_views (padding dahil), view_mask: (B, V)
        B, V, T, H, W = x.shape
        x = x.view(B * V * T, 1, H, W)
        feats = self.backbone(x)              # (B*V*T, feat_dim)
        feats = feats.view(B, V, T, -1)

        view_embeds = []
        for v in range(V):
            pooled = self.slice_pool(feats[:, v])  # (B, feat_dim) -- her serinin kendi slice'ları
            view_embeds.append(pooled)
        view_embeds = torch.stack(view_embeds, dim=1)  # (B, V, feat_dim)

        study_embed = self.view_pool(view_embeds, mask=view_mask)  # (B, feat_dim)
        study_embed = self.norm(study_embed)

        logits = self.head(study_embed)  # (B, n_labels)
        return logits
