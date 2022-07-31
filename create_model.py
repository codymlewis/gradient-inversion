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
from flax.core.frozen_dict import FrozenDict
from tqdm import trange

import models


def celoss(model):
    """Cross entropy loss with some clipping to prevent NaNs"""
    @jax.jit
    def _apply(variables, X, Y):
        logits, batch_stats = model.apply(variables, X, mutable=['batch_stats'])
        logits = jnp.clip(logits, 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits))), batch_stats
    return _apply


def accuracy(model, params, X, Y, rng=jax.random.PRNGKey(0), batch_size=1000):
    """Accuracy metric using batch size to prevent OOM errors"""
    acc = 0
    ds_size = len(Y)
    for i in range(0, ds_size, batch_size):
        rng, use_rng = jax.random.split(rng)
        end = min(i + batch_size, ds_size)
        logits = model.apply(params, X[i:end], rngs={'dropout': use_rng}, train=False)
        acc += jnp.mean(jnp.argmax(logits, axis=-1) == Y[i:end])
    return acc / jnp.ceil(ds_size / batch_size)


def value_and_grad(loss):
    def _apply(variables, X, Y):
        return jax.value_and_grad(loss, has_aux=True)(variables, X, Y)
    return _apply


def dp_value_and_grad(loss, S, sigma, rng=np.random.default_rng()):
    """DP-FedAVG step from https://openreview.net/forum?id=BJ0hF1Z0b"""
    def _apply(params, X, Y):
        (loss_val, batch_stats), grads = jax.value_and_grad(loss, has_aux=True)(params, X, Y)
        norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(params)[0])
        grads = jax.tree_util.tree_map(lambda x: x / jnp.maximum(norm / S, 1), grads)
        grads = jax.tree_util.tree_map(lambda x: x + rng.normal(0, S**2 * sigma**2, x.shape), grads)
        return (loss_val, batch_stats), grads
    return _apply


def train_step(opt, value_and_grad):
    """The training function using optax, also returns the training loss"""
    @jax.jit
    def _apply(variables, opt_state, X, Y, *args):
        (loss_val, batch_stats), grads = value_and_grad(variables, X, Y)
        updates, opt_state = opt.update(grads, opt_state, variables)
        params = optax.apply_updates(variables, updates)
        state = FrozenDict({'params': params['params'], 'batch_stats': batch_stats['batch_stats']})
        return state, opt_state, loss_val, None
    return _apply


def robust_train_step(opt, loss, value_and_grad, epsilon=0.3, lr=0.001, steps=40):
    """AT training step proposed in https://arxiv.org/pdf/1706.06083.pdf"""
    @jax.jit
    def _apply(variables, opt_state, X, Y, rng):
        X_nat = X
        for _ in range(steps):
            use_rng, rng = jax.random.split(rng)
            grads = jax.grad(loss, argnums=1)(variables, X, Y, use_rng)
            X = X + lr * jnp.sign(grads)
            X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
            X = jnp.clip(X, 0, 1)
        (loss_val, batch_stats), grads = value_and_grad(variables, X, Y)
        updates, opt_state = opt.update(grads, opt_state, variables)
        params = optax.apply_updates(variables, updates)
        state = FrozenDict({'params': params['params'], 'batch_stats': batch_stats['batch_stats']})
        return state, opt_state, loss_val, rng
    return _apply


def robust_celoss(model):
    """CE Loss for the robust training"""
    @jax.jit
    def _apply(variables, X, Y, rng):
        logits = jnp.clip(model.apply(variables, X, train=False, rngs={'dropout': rng}), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
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
    opt = optax.sgd(0.1)
    opt_state = opt.init(params)
    loss = celoss(model)
    if args.dp:
        v_and_g = dp_value_and_grad(loss, 0.1,  0.1)
    else:
        v_and_g = value_and_grad(loss)
    if args.robust:
        trainer = robust_train_step(opt, robust_celoss(model), v_and_g)
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
            params, opt_state, loss_val, key = trainer(params, opt_state, X[idx], Y[idx], key)
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
        new_params, opt_state, loss_val, key = trainer(new_params, opt_state, X[idx], Y[idx], key)
    grads = jax.tree_util.tree_map(operator.sub, params, new_params)
    fn = f"data/{args.model}{'-robust' if args.robust else ''}{'-dp' if args.dp else ''}.grads"
    with open(fn, 'wb') as f:
        f.write(serialization.to_bytes(grads))
    print(f'Saved final gradient to {fn}')
