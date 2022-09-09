"""
Attack proposed in https://arxiv.org/abs/2202.10546
"""


import argparse
import math
from typing import Callable, Tuple

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import matplotlib.pyplot as plt

import models


def find_in_dict(data: dict, key: str) -> dict:
    """Find part of the data corresponding to the key, from within a linear dictionary"""
    cur_keys = data.keys()
    if key in cur_keys:
        return data[key]
    return find_in_dict(data[next(iter(cur_keys))], key)


def cosine_dist(A: jnp.array, B: jnp.array) -> float:
    denom = jnp.maximum(jnp.linalg.norm(A, axis=1) * jnp.linalg.norm(B, axis=1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V: jnp.array) -> float:
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def atloss(
    model: nn.Module, params: optax.Params, true_reps: jnp.array, lamb_tv: float = 1e-3
) -> Callable[[jnp.array], float]:
    def _apply(Z: jnp.array) -> float:
        dist = cosine_dist(
            model.apply(
                params, Z, representation=True, mutable=['batch_stats']
            )[0],
            true_reps
        )
        return dist + lamb_tv * total_variation(Z)
    return _apply


def train_step(
    opt: optax.GradientTransformation, loss: Callable[[jnp.array], float]
) -> Callable[[jnp.array, optax.OptState], Tuple[jnp.array, optax.OptState, float]]:
    @jax.jit
    def _apply(Z: jnp.array, opt_state: optax.OptState):
        loss_val, grads = jax.value_and_grad(loss)(Z)
        updates, opt_state = opt.update(grads, opt_state, Z)
        Z = jnp.clip(optax.apply_updates(Z, updates), 0, 1)
        return Z, opt_state, loss_val
    return _apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=500, help="Steps of training to perform.")
    parser.add_argument('--robust', action="store_true", help="Attack a robustly trained model.")
    parser.add_argument('--dp', type=float, nargs='*', help="Perform differentially private training.")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch size to perform the attack on.")
    args = parser.parse_args()

    model = getattr(models, args.model)()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((32, 224, 224, 3)))
    if args.dp is not None:
        if len(args.dp) == 2:
            S, sigma = args.dp
        else:
            S, sigma = 0.1, 0.1
    fn = "data/{}{}{}.variables".format(
        args.model, '-robust' if args.robust else '',
        f'-dp-S{S}-sigma{sigma}' if args.dp is not None else ''
    )
    with open(fn, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
    fn = "data/{}{}{}.grads".format(
        args.model, '-robust' if args.robust else '',
        f'-dp-S{S}-sigma{sigma}' if args.dp is not None else ''
    )
    with open(fn, 'rb') as f:
        true_grads = serialization.from_bytes(params, f.read())
    labels = jnp.argsort(
        jnp.min(find_in_dict(true_grads['params'], 'predictions')['kernel'], axis=0)
    )[:args.batch_size]
    true_reps = find_in_dict(true_grads['params'], 'predictions')['kernel'].T[labels.tolist()]
    Z = jax.random.normal(jax.random.PRNGKey(42), (args.batch_size, 224, 224, 3))
    Z = jnp.clip(Z, 0, 1)
    opt = optax.adam(0.01)
    opt_state = opt.init(Z)
    trainer = train_step(opt, atloss(model, params, true_reps))
    for _ in (pbar := trange(args.steps)):
        Z, opt_state, loss_val = trainer(Z, opt_state)
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    # Plot the results
    if args.batch_size > 1:
        if args.batch_size > 3:
            nrows, ncols = round(math.sqrt(args.batch_size)), round(math.sqrt(args.batch_size))
        else:
            nrows, ncols = 1, args.batch_size
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(f"Label: {labels[i]}")
            ax.imshow(Z[i], cmap='binary')
    else:
        plt.title(f"Label: {labels[0]}")
        plt.imshow(Z[0], cmap="binary")
    plt.tight_layout()
    plt.show()
