import math
import torch
from torch import nn
from torchinfo import summary


class NewGELUActivation(nn.Module):
    def forward(self, input):
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(
        self, image_size: int, patch_size: int, num_channels: int, hidden_size: int
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(
            self.num_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        hidden_size: int,
        hidden_dropout: float,
    ):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = bias
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                attention_dropout,
                self.qkv_bias,
            )
            self.heads.append(head)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat(
            [attention_output for attention_output in attention_outputs], dim=-1
        )
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        return attention_output


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout: float):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            bias=bias,
        )
        self.layernorm_1 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
        )
        self.layernorm_2 = nn.LayerNorm(hidden_size)

    def forward(self, x, output_attentions=False):
        attention_output = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        return x


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_hidden_layers):
            block = Block(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                bias=bias,
            )
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        for block in self.blocks:
            x = block(x, output_attentions=output_attentions)
        return x


class VisionTransformer(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        num_classes: int,
        hidden_size: int,
        hidden_dropout: float,
        num_hidden_layers: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embedding = Embeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            hidden_dropout=hidden_dropout,
        )
        self.encoder = Encoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            bias=bias,
        )
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        logits = self.classifier(encoder_output[:, 0, :])
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.cls_token.dtype)


if __name__ == "__main__":
    vit = VisionTransformer(
        image_size=32,
        patch_size=4,
        num_channels=3,
        num_classes=10,
        hidden_size=48,
        hidden_dropout=0.0,
        num_hidden_layers=4,
        intermediate_size=4 * 48,
        num_attention_heads=4,
        attention_dropout=0.0,
        bias=True,
    )
    summary(vit, input_size=(1, 3, 32, 32))
