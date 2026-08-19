import timm
import torch
import torch.nn as nn
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKBONE_PATH = PACKAGE_ROOT / "models" / "pytorch_model.bin"


def set_vit_dropout(model, mlp_drop=0.3, attn_drop=0.1, drop_path=0.1):
    for block in model.blocks:
        block.mlp.drop1.p = mlp_drop
        block.mlp.drop2.p = mlp_drop
        block.attn.attn_drop.p = attn_drop
        if hasattr(block, 'drop_path1') and isinstance(block.drop_path1, nn.Dropout):
            block.drop_path1.p = drop_path
        if hasattr(block, 'drop_path2') and isinstance(block.drop_path2, nn.Dropout):
            block.drop_path2.p = drop_path
    if hasattr(model, 'head_drop'):
        model.head_drop.p = mlp_drop


class ThymomaTransformerClassifier(nn.Module):
    def __init__(self, backbone_name='vit_tiny_patch16_224', num_classes=2,
                 drop_out=0.2, num_layers=4, max_slices=32,
                 backbone_checkpoint=DEFAULT_BACKBONE_PATH):
        super().__init__()
        self.max_tokens = max_slices

        # CNN Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        if backbone_checkpoint is not None:
            backbone_checkpoint = Path(backbone_checkpoint)
            if not backbone_checkpoint.exists():
                raise FileNotFoundError(
                    f"Backbone checkpoint not found: {backbone_checkpoint}"
                )
            state_dict = torch.load(backbone_checkpoint, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
        set_vit_dropout(self.backbone, mlp_drop=0.1, attn_drop=0.1, drop_path=0.1)

        self.feature_dim = self.backbone.num_features
        print("self.feature_dim", self.feature_dim)

        # Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_tokens + 1, self.feature_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nhead = max(1, self.feature_dim // 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=nhead,
            dim_feedforward=self.feature_dim // 2,
            dropout=drop_out,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(drop_out),
            nn.Linear(self.feature_dim, num_classes)
        )

    def encode_tokens(self, x):
        """
        Args:
            x: [B, N, 3, H, W] - image slices, where N is the number of slices
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        # CNN features
        feat = self.backbone(x)  # [B*N, D]

        # reshape back to B, N, D
        slice_embeddings = feat.view(B, N, self.feature_dim)

        # Transformer
        cls_token = self.cls_token.expand(B, 1, self.feature_dim)
        tokens = torch.cat([cls_token, slice_embeddings], dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.size(1)].expand_as(tokens)

        tokens = self.transformer(tokens)  # no padding mask
        cls_embedding = tokens[:, 0]
        return {
            "cls_embedding": cls_embedding,
            "slice_embeddings": slice_embeddings,
            "transformer_tokens": tokens,
        }

    def forward(self, x, return_dict=False):
        encoded = self.encode_tokens(x)
        logits = self.classifier(encoded["cls_embedding"])
        if return_dict:
            encoded["logits"] = logits
            return encoded
        return logits
