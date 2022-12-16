from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax


class DPState(NamedTuple):
    rng_key: jnp.array


def dp_aggregate(clipping: float, noise_multiplier: float, seed: int) -> optax.GradientTransformation:
    noise_std = clipping**2  * noise_multiplier**2

    def init_fn(params: optax.Params) -> DPState:
        del params
        return DPState(jax.random.PRNGKey(seed))

    def update_fn(grads: optax.Updates, state: DPState, params: optax.Params=None) -> (optax.Updates, DPState):
        norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
        leaf_count = len(jax.tree_util.tree_leaves(params))
        new_key, *keys = jax.random.split(state.rng_key, leaf_count + 1)
        keys = iter(keys)
        updates = jax.tree_util.tree_map(lambda x: x / jnp.maximum(norm / clipping, 1), grads) 
        updates = jax.tree_util.tree_map(
            lambda x: x + jax.random.normal(next(keys), x.shape) * noise_std, updates
        )
        return updates, DPState(new_key)

    return optax.GradientTransformation(init_fn, update_fn)


def dpsgd(
    learning_rate: float|optax.Schedule, clipping: float, noise_multiplier: float, seed: int
) -> optax.GradientTransformation:
    return optax.chain(
        dp_aggregate(clipping, noise_multiplier, seed),
        optax.sgd(learning_rate)
    )
