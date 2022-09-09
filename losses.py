import jax
import jax.numpy as jnp


def celoss_int_labels(model):
    """Cross entropy loss with some clipping to prevent NaNs"""
    @jax.jit
    def _apply(params, X, Y):
        logits, batch_stats = model.apply(params, X, mutable=['batch_stats'])
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))), batch_stats
    return _apply


def celoss(model):
    """Cross entropy loss with some clipping to prevent NaNs"""
    @jax.jit
    def _apply(params, X, Y):
        logits, batch_stats = model.apply(params, X, mutable=['batch_stats'])
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        return -jnp.mean(jnp.einsum("bl,bl -> b", Y, jnp.log(logits))), batch_stats
    return _apply
