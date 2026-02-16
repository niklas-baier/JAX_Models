import jax
from flax import nnx 

class SimpleModel(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        self.conv = nnx.Conv(din, 32, kernel_size=(3, 3),padding='SAME', rngs=rngs)
        self.linear = nnx.Linear(32 * 28 * 28, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.relu(self.conv(x))
        x = x.reshape(x.shape[0], -1) 
        return self.linear(x)


