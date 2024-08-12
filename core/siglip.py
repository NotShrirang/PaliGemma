from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:

    def __init__(
        self,
        # Size of embedding vector of this vision transformer
        hidden_size: int = 768,
        # Size of the "intermediate" layer in the transformer encoder
        intermediate_size: int = 3072,
        # Number of hidden layers in the transformer
        num_hidden_layers: int = 12,
        # Number of attention heads in MultiHeadAttention
        num_attention_heads: int = 12,
        # Number of channels in the input image (RGB = 3)
        num_channels: int = 3,
        # Size of the input image (224x224)
        image_size: int = 224,
        # Size of the image patch (16x16)
        patch_size: int = 16,
        # Epsilon value for layer normalization
        layer_norm_eps: float = 1e-6,
        # Dropout probability for attention layers
        attention_dropout: float = 0.0,
        # Number of image tokens this will output
        num_image_tokens: int = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",  # No padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(
            self.num_positions,
            self.embed_dim
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # [batch_size, num_channels, height, width]
        _, _, height, width = pixel_values.shape
        # [batch_size, hidden_size, num_patches_h, num_patches_w]
        patch_embeds = self.patch_embedding(pixel_values)
        # num_patches = num_patches_h * num_patches_w
        patch_embeds = patch_embeds.flatten(2)
        # [batch_size, num_patches, hidden_size]
        embeddings = patch_embeds.transpose(1, 2)
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        # embeddings: [batch_size, num_patches, hidden_size]
        return embeddings


class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, Channels, Height, Width] -> [batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        # [batch_size, Channels, Height, Width] -> [batch_size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
