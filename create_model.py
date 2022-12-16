import argparse
import os
import operator

import datasets
import einops
import numpy as np
import optax
import jax
import jax.numpy as jnp
from flax import serialization
from tqdm import trange

import losses
import models
import optimizers


def accuracy(model, params, X, Y, batch_size=1000):
    """Accuracy metric using batch size to prevent OOM errors"""
    acc = 0
    ds_size = len(Y)
    for i in (pbar := trange(0, ds_size, batch_size)):
        end = min(i + batch_size, ds_size)
        acc += jnp.mean(jnp.argmax(model.apply(params, X[i:end]), axis=-1) == Y[i:end])
        pbar.set_postfix_str(f"Running Acc: {acc / jnp.ceil((i + 1) / batch_size):.3%}")
    return acc / jnp.ceil(ds_size / batch_size)


def value_and_grad(loss):
    def _apply(params, X, Y):
        return jax.value_and_grad(loss)(params, X, Y)
    return _apply


def dp_value_and_grad(loss, S, sigma, rng=np.random.default_rng()):
    """DP-FedAVG step from https://openreview.net/forum?id=BJ0hF1Z0b"""
    def _apply(params, X, Y):
        loss_val, grads = jax.value_and_grad(loss)(params, X, Y)
        norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])
        grads = jax.tree_util.tree_map(lambda x: x / jnp.maximum(norm / S, 1), grads)
        grads = jax.tree_util.tree_map(lambda x: x + rng.normal(0, S**2 * sigma**2, x.shape), grads)
        return loss_val, grads
    return _apply


def train_step(opt, value_and_grad):
    """The training function using optax, also returns the training loss"""
    @jax.jit
    def _apply(params, opt_state, X, Y):
        loss_val, grads = value_and_grad(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply


def robust_train_step(opt, loss, value_and_grad, epsilon=0.3, lr=0.001, steps=40):
    """AT training step proposed in https://arxiv.org/pdf/1706.06083.pdf"""
    @jax.jit
    def _apply(params, opt_state, X, Y):
        X_nat = X
        for _ in range(steps):
            grads = jax.grad(loss, argnums=1)(params, X, Y)
            X = X + lr * jnp.sign(grads)
            X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
            X = jnp.clip(X, 0, 1)
        loss_val, grads = value_and_grad(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply


def load_dataset():
    """Load and preprocess the MNIST dataset"""
    ds = datasets.load_dataset('mnist')
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=3000, help="Steps of training to perform.")
    parser.add_argument('--grad-steps', type=int, default=1, help="Number of steps for gradient production.")
    parser.add_argument('--checkpoint', action="store_true", help="Skip training and only make gradients.")
    parser.add_argument('--robust', action="store_true", help="Perform adversarially robust training.")
    parser.add_argument('--dp', action="store_true", help="Perform differentially private training.")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size of the final gradient.")
    args = parser.parse_args()

    ds = load_dataset()
    X, Y = ds['train']['X'], ds['train']['Y']
    model = getattr(models, args.model)()
    key = jax.random.PRNGKey(42)
    key, pkey = jax.random.split(key)
    params = model.init(pkey, X[:32])
    if args.dp is not None:
        if len(args.dp) == 2:
            S, sigma = args.dp
        else:
            S, sigma = 0.1, 0.1
        opt = optimizers.dpsgd(0.1, S, sigma, 0)
    else:
        opt = optax.sgd(0.1)
    opt_state = opt.init(params)
    loss = losses.celoss_int_labels(model)
    if args.dp:
        v_and_g = dp_value_and_grad(loss, 0.1,  0.1)
    else:
        v_and_g = value_and_grad(loss)
    if args.robust:
        trainer = robust_train_step(opt, loss, v_and_g)
    else:
        trainer = train_step(opt, v_and_g)
    rng = np.random.default_rng()
    train_len = len(Y)
    fn = f"data/{args.model}{'-robust' if args.robust else ''}{'-dp' if args.dp else ''}.params"
    if args.checkpoint:
        with open(fn, 'rb') as f:
            params = serialization.from_bytes(params, f.read())
    else:
        for _ in (pbar := trange(args.steps)):
            idx = rng.choice(train_len, 32, replace=False)
            params, opt_state, loss_val = trainer(params, opt_state, X[idx], Y[idx])
            pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
        print(f"Final accuracy: {accuracy(model, params, ds['test']['X'], ds['test']['Y']):.3%}")
        os.makedirs('data', exist_ok=True)
        with open(fn, 'wb') as f:
            f.write(serialization.to_bytes(params))
        print(f'Saved final model to {fn}')
    # Generate the gradients to be attacked
    new_params = params
    for _ in (pbar := trange(args.grad_steps)):
        idx = rng.choice(train_len, args.batch_size, replace=False)
        new_params, opt_state, loss_val = trainer(new_params, opt_state, X[idx], Y[idx])
    grads = jax.tree_util.tree_map(operator.sub, params, new_params)
    fn = f"data/{args.model}{'-robust' if args.robust else ''}{'-dp' if args.dp else ''}.grads"
    with open(fn, 'wb') as f:
        f.write(serialization.to_bytes(grads))
    print(f'Saved final gradient to {fn}')
