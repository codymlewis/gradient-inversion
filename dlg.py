import argparse
import operator

from flax import serialization
import jax
import jax.numpy as jnp
import jaxopt
import optax
from tqdm import trange
import matplotlib.pyplot as plt

import losses
import models


def dlg_loss(loss, params, true_grads):
    @jax.jit
    def _apply(X, Y):
        norm_tree = jax.tree_map(lambda a, b: jnp.sum((a - b)**2), jax.grad(loss)(params, X, Y), true_grads)
        return jax.tree_util.tree_reduce(operator.add, norm_tree)
    return _apply


def train_step(opt, loss):
    @jax.jit
    def _apply(X, Y, Xopt_state, Yopt_state):
        loss_val, Xgrads = jax.value_and_grad(loss)(X, Y)
        Ygrads = jax.grad(loss, argnums=1)(X, Y)
        Xupdates, Xopt_state = opt.update(Xgrads, Xopt_state, X)
        X = optax.apply_updates(X, Xupdates)
        Yupdates, Yopt_state = opt.update(Ygrads, Yopt_state, Y)
        Y = optax.apply_updates(Y, Yupdates)
        return X, Y, Xopt_state, Yopt_state, loss_val
    return _apply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=50000, help="Steps of training to perform.")
    parser.add_argument('--target', type=int, default=0, help="Class to target with the attack.")
    parser.add_argument('--robust', action="store_true", help="Attack a robustly trained model.")
    parser.add_argument('--lbfgs', action="store_true", help="Use the LBFGS optimizer.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate to use.")
    args = parser.parse_args()

    model = getattr(models, args.model)()
    rngkey = jax.random.PRNGKey(42)
    xkey, ykey = jax.random.split(rngkey)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((32, 28, 28, 1)))
    fn = f"data/{args.model}{'-robust' if args.robust else ''}.params"
    with open(fn, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
    fn = f"data/{args.model}{'-robust' if args.robust else ''}.grads"
    with open(fn, 'rb') as f:
        true_grads = serialization.from_bytes(params, f.read())
    X = jax.random.uniform(xkey, (1, 28, 28, 1))
    Y = jax.random.uniform(xkey, (1, 10))
    model_loss = losses.celoss(model)
    loss = dlg_loss(model_loss, params, true_grads)
    if args.lbfgs:
        solver = jaxopt.LBFGS(lambda XY: loss(*XY), tol=1e-5, history_size=100, maxiter=20)
        state = solver.init_state((X, Y))
        trainer = jax.jit(solver.update)
        for _ in (pbar := trange(args.steps)):
            (X, Y), state = trainer((X, Y), state)
            pbar.set_postfix_str(f"LOSS: {state.value:.5f}")
    else:
        opt = optax.sgd(args.lr)
        Xopt_state = opt.init(X)
        Yopt_state = opt.init(Y)
        trainer = train_step(opt, loss)
        for _ in (pbar := trange(args.steps)):
            X, Y, Xopt_state, Yopt_state, loss_val = trainer(X, Y, Xopt_state, Yopt_state)
            pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
    plt.imshow(X[0], cmap='binary')
    plt.title(f"Label {jnp.argmax(Y[0])}")
    plt.show()
