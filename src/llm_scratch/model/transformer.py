"""WP05.3 pre-norm decoder block; full decoder/LM head/loss follow in WP05.4."""

from __future__ import annotations

from torch import Tensor, nn

from llm_scratch.model.attention import CausalSelfAttention
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RMSNorm, SwiGLU
from llm_scratch.model.masking import valid_token_mask


class DecoderBlock(nn.Module):
    """Two pre-norm residual sublayers, input/output [batch, sequence, hidden].

    Clear padding before the first norm/residual so nonfinite padding cannot enter
    RMSNorm gradients or bypass attention through the identity path. Mask again
    after each residual. Valid-token residuals are exact identity when branches vanish.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_hidden_size)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        valid = valid_token_mask(x, attention_mask, self.config)
        padding = ~valid.unsqueeze(-1)
        x = x.masked_fill(padding, 0.0)
        attended = self.attention(self.attention_norm(x), valid, position_ids)
        x = (x + self.residual_dropout(attended)).masked_fill(padding, 0.0)
        transformed = self.ffn(self.ffn_norm(x))
        return (x + self.residual_dropout(transformed)).masked_fill(padding, 0.0)
