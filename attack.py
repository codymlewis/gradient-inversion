"""
Attack proposed in https://arxiv.org/abs/2202.10546
"""


import argparse
import math

from flax import serialization
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import matplotlib.pyplot as plt

import models


def cosine_sim(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A, axis=1) * jnp.linalg.norm(B, axis=1), 1e-15)
    return jnp.mean(jnp.einsum('br,br -> b', A, B) / denom)


def total_variation(V):
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def atloss(model, params, true_reps):
    def _apply(Z):
        # The paper says to use cos distance, but that makes the grads go in the wrong direction
        return cosine_sim(model.apply(params, Z, representation=True), true_reps) #- 1e-9 * total_variation(Z)
    return _apply


def train_step(opt, loss):
    @jax.jit
    def _apply(Z, opt_state):
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
    parser.add_argument('--batch-size', type=int, default=1,
                        help="Batch size to perform the attack on.")
    args = parser.parse_args()

    model = getattr(models, args.model)()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((32, 28, 28, 1)))
    fn = f"data/{args.model}{'-robust' if args.robust else ''}.params"
    with open(fn, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
    fn = f"data/{args.model}{'-robust' if args.robust else ''}.grads"
    with open(fn, 'rb') as f:
        true_grads = serialization.from_bytes(params, f.read())
    labels = jnp.argsort(
        jnp.min(true_grads['params']['classifier']['kernel'], axis=0)
    )[:args.batch_size]
    true_reps = true_grads['params']['classifier']['kernel'].T[labels.tolist()]
    Z = jax.random.normal(jax.random.PRNGKey(42), (args.batch_size, 28, 28, 1))
    Z = jnp.clip(Z, 0, 1)
    opt = optax.adam(0.01)
    opt_state = opt.init(Z)
    trainer = train_step(opt, atloss(model, params, true_reps))
    for _ in (pbar := trange(args.steps)):
        Z, opt_state, loss_val = trainer(Z, opt_state)
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    # Plot the results
    if args.batch_size > 3:
        nrows, ncols = round(math.sqrt(args.batch_size)), round(math.sqrt(args.batch_size))
    else:
        nrows, ncols = 1, args.batch_size
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(f"Label: {labels[i]}")
        ax.imshow(Z[i], cmap='binary')
    plt.tight_layout()
    plt.show()
