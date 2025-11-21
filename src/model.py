import torch
from torch import Tensor
import torch.nn as nn
from safetensors.torch import load_model
from jaxtyping import Bool, Int, Float
from src.embedding import InputEmbeddings, PositionalEncoding
from src.modules import Encoder, Decoder


class Generator(nn.Module):
    """
    Implements the final Linear (Projection) layer and Softmax.

    This module takes the final output of the Decoder stack (B, T, D)
    and projects it onto the vocabulary space (B, T, vocab_size)
    to produce the logits.

    (This layer's weights can be tied with the
    target embedding layer, which we will handle in the main
    'Transformer' model class).
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the Generator (Output Projection) layer.

        Args:
            d_model (int): The dimension of the model (D).
            vocab_size (int): The size of the target vocabulary.
        """
        super().__init__()

        self.proj: nn.Linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: Float[Tensor, "B T_tgt D"]
    ) -> Float[Tensor, "B T_tgt vocab_size"]:
        """
        Forward pass for the Generator.

        Args:
            x (Tensor): The final output tensor from the Decoder stack.

        Returns:
            Tensor: The output logits over the vocabulary.
        """
        # (B, T_tgt, D) -> (B, T_tgt, vocab_size)
        logits = self.proj(x)
        return logits


class Transformer(nn.Module):
    """
    The main Transformer model architecture, combining the Encoder
    and Decoder stacks, as described in "Attention Is All You Need".

    This implementation follows modern best practices (Pre-LN) and
    is designed for a sequence-to-sequence task (e.g., translation).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,  # N=6 in the paper
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,  # Max length for positional encoding
    ) -> None:
        """
        Initializes the full Transformer model.

        Args:
            src_vocab_size (int): Vocabulary size for the source language.
            tgt_vocab_size (int): Vocabulary size for the target language.
            d_model (int): The dimension of the model (D).
            n_heads (int): The number of attention heads (H).
            n_layers (int): The number of Encoder/Decoder layers (N).
            d_ff (int): The inner dimension of the Feed-Forward Network (D_FF).
            dropout (float): The dropout rate.
            max_seq_len (int): The maximum sequence length for positional encoding.
        """
        super().__init__()

        self.d_model = d_model

        # --- 1. Source (Encoder) Embeddings ---
        # We create two separate embedding layers
        self.src_embed: InputEmbeddings = InputEmbeddings(d_model, src_vocab_size)

        # --- 2. Target (Decoder) Embeddings ---
        self.tgt_embed: InputEmbeddings = InputEmbeddings(d_model, tgt_vocab_size)

        # --- 3. Positional Encoding ---
        # We use "one" PositionalEncoding module
        # and share it for both source and target.
        self.pos_enc: PositionalEncoding = PositionalEncoding(
            d_model, max_seq_len, dropout
        )

        # --- 4. Encoder Stack ---
        self.encoder: Encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout)

        # --- 5. Decoder Stack ---
        self.decoder: Decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout)

        # --- 6. Final Output Projection (Generator) ---
        self.generator: Generator = Generator(d_model, tgt_vocab_size)

        # --- Weight Typing ---
        # We tie the weights of the target embedding and the generator.
        # This saves parameters and improves performance.
        self.generator.proj.weight = self.tgt_embed.token_emb.weight

        # --- Initialize weights ---
        # This is crucial for stable training.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Applies Xavier/Glorot uniform initialization to linear layers.
        This is a common and effective initialization strategy.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Embedding):
            # Initialize embeddings (e.g., from a normal distribution)
            nn.init.normal_(module.weight, mean=0, std=self.d_model**-0.5)

    def forward(
        self,
        src: Int[Tensor, "B T_src"],  # Source token IDs (e.g., English)
        tgt: Int[Tensor, "B T_tgt"],  # Target token IDs (e.g., Vietnamese)
        src_mask: Bool[Tensor, "B 1 1 T_src"],  # Source padding mask
        tgt_mask: Bool[Tensor, "B 1 T_tgt T_tgt"],  # Target combined mask
    ) -> Float[Tensor, "B T_tgt vocab_size"]:
        """
        Defines the main forward pass of the Transformer model.

        Args:
            src (Tensor): Source sequence token IDs.
            tgt (Tensor): Target sequence token IDs (shifted right).
            src_mask (Tensor): Padding mask for the source sequence.
            tgt_mask (Tensor): Combined padding and look-ahead mask
                               for the target sequence.

        Returns:
            Tensor: The output logits from the model (B, T_tgt, vocab_size).
        """
        # 1. Encode the source sequence
        # (B, T_src) -> (B, T_scr, D)
        src_embeded = self.src_embed(src)
        src_with_pos = self.pos_enc(src_embeded)

        # (B, T_src, D) -> (B, T_src, D)
        # This 'memory' will be used by every DecoderLayer
        enc_output: Tensor = self.encoder(src_with_pos, src_mask)

        # 2. Decode the target sequence
        # (B, T_tgt) -> (B, T_tgt, D)
        tgt_embeded = self.tgt_embed(tgt)
        tgt_with_pos = self.pos_enc(tgt_embeded)

        # (B, T_tgt, D) -> (B, T_tgt, D)
        dec_output: Tensor = self.decoder(tgt_with_pos, enc_output, src_mask, tgt_mask)

        # 3. Generate final logits
        # (B, T_tgt, D) -> (B, T_tgt, vocab_size)
        logits: Tensor = self.generator(dec_output)

        return logits


def load_trained_model(
    config_obj, checkpoint_path, device: torch.device
) -> Transformer:
    print("Instantiating the Transformer model...")
    model = Transformer(
        src_vocab_size=config_obj.VOCAB_SIZE,
        tgt_vocab_size=config_obj.VOCAB_SIZE,
        d_model=config_obj.D_MODEL,
        n_heads=config_obj.N_HEADS,
        n_layers=config_obj.N_LAYERS,
        d_ff=config_obj.D_FF,
        dropout=config_obj.DROPOUT,
        max_seq_len=config_obj.MAX_SEQ_LEN,
    ).to(device)

    print(f"Loading model from: {checkpoint_path}")
    load_model(model, filename=checkpoint_path)

    return model
