import jax
from flax import nnx
import jax.numpy as jnp
import einops

class Encoder(nnx.Module):
    def __init__(self, input_dim,num_layers,num_channels, rngs:nnx.Rngs):
        # idea test different number of output features.
        self.num_layers= num_layers
        for i in range(num_layers):
            if i == 0:
                setattr(self, f'layer{i}', nnx.Conv(num_channels,8, kernel_size=(3,3), strides = 2,rngs=rngs))
            else:
                setattr(self, f'layer{i}', nnx.Conv(2**(2+i), 2**(3+i), kernel_size=(3,3), strides = 2,rngs=rngs))
    def __call__(self,x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x)
        return x
class Decoder(nnx.Module):
    def __init__(self, input_dim,num_layers,num_channels, rngs:nnx.Rngs):
        # idea test different number of output features.
        self.num_layers = num_layers
        for i in range(num_layers):  # 64,32,16,8,3
            if i == num_layers-1:
                setattr(self, f'layer{i}', nnx.ConvTranspose(8,num_channels, kernel_size=(3,3), strides = 2,rngs=rngs))
            else:
                setattr(self, f'layer{i}', nnx.ConvTranspose(2**(num_layers-i-1+3), 2**(num_layers-i-1+2), kernel_size=(3,3), strides = 2,rngs=rngs))
    def __call__(self,x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer{i}')(x)
        return x

class AutoEncoder(nnx.Module):
   def __init__ (self, input_dim,num_classes, num_layers,num_channels, rngs:nnx.Rngs):
       self.encoder = Encoder(input_dim, num_layers, num_channels, rngs)
       self.class_projection = nnx.Param(jax.random.uniform(rngs.params(),(4*4*64,10)))
       self.decoder = Decoder(input_dim, num_layers, num_channels, rngs)
   def __call__(self, x):
       x = self.encoder(x)
       reshaped_latent = einops.rearrange(x,'b h w c -> b (h w c)')
       label = reshaped_latent @ self.class_projection.value
       x = self.decoder(x)
       return label, x


class SimpleModel(nnx.Module):
    def __init__(self, din: int, dout: int,image_shape:tuple, rngs: nnx.Rngs):
        #self.conv = nnx.Conv(din, 32, kernel_size=(3, 3),padding='SAME', rngs=rngs)
        breakpoint()
        self.depthwise_conv = nnx.Conv(3,3, kernel_size=(3,3), feature_group_count=3, rngs=rngs)
        self.pointwise_conv = nnx.Conv(3, 8, kernel_size=(1,1), rngs=rngs)
        self.depthwise_conv2 = nnx.Conv(8,8, kernel_size=(3,3), feature_group_count=8, rngs=rngs)
        self.pointwise_conv2 = nnx.Conv(8, 32, kernel_size=(1,1), rngs=rngs)
        self.embedding = nnx.Param(jax.random.uniform(rngs.params(),(*image_shape,3)))
        self.attention = TransformerBlock(embed_dim=32, num_heads=4, dropout_rate=0.1, rngs=rngs)
        self.linear = nnx.Linear(32, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = x + self.embedding.value[None,:,:,:]
        x = nnx.relu(self.pointwise_conv(self.depthwise_conv(x)))
        x = nnx.relu(self.pointwise_conv2(self.depthwise_conv2(x)))
        x = einops.rearrange(x, 'b h w c -> b (h w) c')
        x = self.attention(x)
        x = jnp.mean(x,axis=1)
        return self.linear(x)

class Factorized_layer(nnx.Module):
    def __init__(self, din: int, dout: int,rank, rngs: nnx.Rngs):
        self.matrix_a = nnx.Linear(din, rank,use_bias=False,rngs=rngs)
        self.matrix_b = nnx.Linear(rank, dout,use_bias=False,rngs=rngs)

    def __call__(self, x: jax.Array)-> jax.Array:
       return self.matrix_b(self.matrix_a(x))

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
        #self.wi_0 = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs)
        #self.wi_1 = nnx.Linear(embed_dim, hidden_dim, use_bias=False, rngs=rngs)
        #self.wo = nnx.Linear(hidden_dim, embed_dim, use_bias=False, rngs=rngs)
        self.wi_0 = Factorized_layer(embed_dim, hidden_dim, rank=4, rngs=rngs)
        self.wi_1 = Factorized_layer(embed_dim, hidden_dim, rank=4, rngs=rngs)
        self.wo = Factorized_layer(hidden_dim, embed_dim, rank=4, rngs=rngs)
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
