import jax
import jax.numpy as jnp


def celoss_int_labels(model):
    """Cross entropy loss with some clipping to prevent NaNs"""
    @jax.jit
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _apply


def celoss(model):
    """Cross entropy loss with some clipping to prevent NaNs"""
    @jax.jit
    def _apply(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        return -jnp.mean(jnp.einsum("bl,bl -> b", Y, jnp.log(logits)))
    return _apply



def robust_loss(loss, alpha=0.5, epsilon=0.25):
    """Adversarially robust training as proposed in https://arxiv.org/abs/1412.6572"""
    @jax.jit
    def _apply(params, X, Y):
        normal = alpha * loss(params, X, Y)
        robust = (1 - alpha) * loss(params, X + epsilon * jnp.sign(jax.grad(loss, argnums=1)(params, X, Y)), Y)
        return normal + robust
    return _apply
