import jax
from flax import nnx
import jax.numpy as jnp
import einops
class SimpleModel(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(din, 32, kernel_size=(3, 3),padding='SAME', rngs=rngs)
        self.linear = nnx.Linear(32 * 28 * 28, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.relu(self.conv(x))
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)



class AttentionBlock(nnx.Module):
    def __init__(self, num_features: int, num_heads: int, dropout_rate: float = 0.1, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        self.qkv_proj = nnx.Linear(num_features, num_features * 3, rngs=rngs)
        self.out_proj = nnx.Linear(num_features, num_features, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool = True) -> jnp.ndarray:
        # 1. Project and Split
        qkv = self.qkv_proj(x)
        # Shape: (batch, seq, 3*features) -> (3, batch, heads, seq, head_dim)
        q, k, v = einops.rearrange(
            qkv, 'b seq (three n head_dim) -> three b n seq head_dim',
            three=3, n=self.num_heads
        )
        # (b, n, l, d) @ (b, n, d, l) -> (b, n, l, l)
        scale = jnp.sqrt(q.shape[-1])
        logits = (q @ k.transpose(0, 1, 3, 2)) / scale
        weights = jax.nn.softmax(logits, axis=-1)
        weights = self.dropout(weights, deterministic=not train)
        out = weights @ v
        out = einops.rearrange(out, 'b n l d -> b l (n d)')
        return self.out_proj(out)




class FFNModernBlock(nnx.Module):
    def __init__(self, embed_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        # 1. Standard SOTA Scaling: (4 * d_model) * (2 / 3)
        hidden_dim = int(embed_dim * 4 * 2 / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64
        self.wi_0 = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.wi_1 = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(hidden_dim, embed_dim, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        gate = nnx.gelu(self.wi_0(x)) # MUST BE DIFFERENT WEIGHTS THAN LINEAR
        x = gate * self.wi_1(x) # not just a single param model can learn when to close the gate less static

        x = self.dropout(x, deterministic=not train)
        return self.wo(x)
        class TransformerBlock(nnx.Module):
            def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
                self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
                self.attention = AttentionBlock(embed_dim, num_heads, dropout_rate, rngs=rngs)

                self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
                self.ffn = FFNModernBlock(embed_dim, dropout_rate, rngs=rngs)

            def __call__(self, x: jnp.ndarray, *, train: bool = True):
                x = x + self.attention(self.ln1(x), train=train)
                x = x + self.ffn(self.ln2(x), train=train)
                return x
