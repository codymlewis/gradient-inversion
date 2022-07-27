"""
Attack proposed in https://arxiv.org/abs/2202.10546
"""


import argparse

from flax import serialization
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import matplotlib.pyplot as plt

import models


def cosine_distance(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A) * jnp.linalg.norm(B), 1e-15)
    return jnp.mean(1 - (jnp.dot(A, B) / denom))


def total_variation(V):
    return abs(V[:, 1:] - V[:, :-1]).sum()


def atloss(model, params, true_reps):
    def _apply(Z):
        reps = model.apply(params, Z, representation=True)
        return cosine_distance(reps, true_reps) + total_variation(reps)
    return _apply


def train_step(opt, loss):
    @jax.jit
    def _apply(Z, opt_state):
        loss_val, grads = jax.value_and_grad(loss)(Z)
        updates, opt_state = opt.update(grads, opt_state, Z)
        Z = optax.apply_updates(Z, updates)
        return Z, opt_state, loss_val
    return _apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=50000, help="Steps of training to perform.")
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
    true_reps = true_grads['params']['classifier']['kernel'].T[tuple(labels)]
    Z = jax.random.normal(jax.random.PRNGKey(42), (args.batch_size, 28, 28, 1))
    opt = optax.sgd(0.1)
    opt_state = opt.init(Z)
    trainer = train_step(opt, atloss(model, params, true_reps))
    for _ in (pbar := trange(args.steps)):
        Z, opt_state, loss_val = trainer(Z, opt_state)
        pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    plt.imshow(Z[0], cmap='binary')
    plt.show()
