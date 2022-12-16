"""
Perform gradient inversion attacks
"""


import argparse
import math
import operator

from flax import serialization
import jax
import jax.numpy as jnp
import optax
import jaxopt
from tqdm import trange
import matplotlib.pyplot as plt

import models
import losses


def load_model(model_name, dp, robust):
    """
    Load the model, parameters and the true gradients
    """
    model = getattr(models, model_name)()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((32, 28, 28, 1)))
    if dp is not None:
        if len(dp) == 2:
            S, sigma = dp
        else:
            S, sigma = 0.1, 0.1
    fn = "data/{}{}{}.params".format(
        model_name, '-robust' if robust else '',
        f'-dp-S{S}-sigma{sigma}' if dp is not None else ''
    )
    with open(fn, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
    fn = "data/{}{}{}.grads".format(
        model_name, '-robust' if robust else '',
        f'-dp-S{S}-sigma{sigma}' if dp is not None else ''
    )
    with open(fn, 'rb') as f:
        true_grads = serialization.from_bytes(params, f.read())
    return model, params, true_grads

# Deep Leakage from Gradients loss from https://arxiv.org/abs/1906.08935

def dlg_loss(loss, params, true_grads):
    """Finds the euclidean distance beween the gradient from dummy data and the true gradient"""
    @jax.jit
    def _apply(X, Y):
        norm_tree = jax.tree_map(lambda a, b: jnp.sum((a - b)**2), jax.grad(loss)(params, X, Y), true_grads)
        return jax.tree_util.tree_reduce(operator.add, norm_tree)
    return _apply


# Representation inversion attack proposed in https://arxiv.org/abs/2202.10546
def cosine_dist(A, B):
    denom = jnp.maximum(jnp.linalg.norm(A, axis=1) * jnp.linalg.norm(B, axis=1), 1e-15)
    return 1 - jnp.mean(jnp.abs(jnp.einsum('br,br -> b', A, B)) / denom)


def total_variation(V):
    return abs(V[:, 1:, :] - V[:, :-1, :]).sum() + abs(V[:, :, 1:] - V[:, :, :-1]).sum()


def atloss(model, params, true_reps, lamb_tv=1e-3):
    def _apply(Z):
        dist = cosine_dist(model.apply(params, Z, representation=True), true_reps)
        return dist + lamb_tv * total_variation(Z)
    return _apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a gradient inversion attack.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to attack.")
    parser.add_argument('--steps', type=int, default=500, help="Steps of the attack to perform.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the attack")
    parser.add_argument('--robust', action="store_true", help="Attack a robustly trained model.")
    parser.add_argument('--dp', type=float, nargs='*', help="Attack a differentially private trained model.")
    parser.add_argument('--lbfgs', action="store_true", help="Use the LBFGS optimizer.")
    parser.add_argument('-a', '--attack', type=str, default="rep", help="Attack to perform.")
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help="Batch size to perform the attack on.")
    args = parser.parse_args()

    model, params, true_grads = load_model(args.model, args.dp, args.robust)

    labels = jnp.argsort(
        jnp.min(true_grads['params']['classifier']['kernel'], axis=0)
    )[:args.batch_size]
    true_reps = true_grads['params']['classifier']['kernel'].T[labels.tolist()]

    rngkey = jax.random.PRNGKey(42)
    xkey, ykey = jax.random.split(rngkey)
    Z = jax.random.normal(xkey, (args.batch_size, 28, 28, 1))

    if "rep" in args.attack:
        loss_fn = atloss(model, params, true_reps)
        opt = optax.adam(args.lr)
        pre_update = lambda X, s: (jnp.clip(X, 0., 1.), s)
    elif "dlg" in args.attack:
        if "i" in args.attack:
            # improved deep leakage attack from https://arxiv.org/abs/2001.02610
            loss_fn = lambda X: dlg_loss(losses.celoss(model), params, true_grads)(X, jax.nn.one_hot(labels, 10))
            pre_update = lambda X, s: (jnp.clip(X, 0., 1.), s)
        else:
            Y = jax.random.uniform(ykey, (args.batch_size, 10))
            loss_fn = lambda XY: dlg_loss(losses.celoss(model), params, true_grads)(*XY)
            pre_update = lambda XY, s: ((jnp.clip(XY[0], 0., 1.), jnp.clip(XY[1], 0., 1.)), s)
            Z = (Z, Y)
        opt = optax.sgd(args.lr)

    if args.lbfgs:
        solver = jaxopt.LBFGS(loss_fn, tol=1e-5, history_size=100, maxiter=20)
    else:
        solver = jaxopt.OptaxSolver(loss_fn, opt, pre_update=pre_update)
    state = solver.init_state(Z)
    trainer = jax.jit(solver.update)

    # Perform the attack
    for _ in (pbar := trange(args.steps)):
        Z, state = trainer(Z, state)
        pbar.set_postfix_str(f"LOSS: {state.value:.5f}")

    # Plot the results
    if isinstance(Z, tuple):
        Z, labels = Z
        labels = jnp.argmax(labels, axis=-1)
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
