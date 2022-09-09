import math
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
from flax.core.frozen_dict import freeze
from tqdm import trange
from sklearn import metrics

import losses
import models
import optimizers


def accuracy(model, variables, X, Y, batch_size=32):
    """Accuracy metric using batch size to prevent OOM errors"""
    @jax.jit
    def apply(X, k):
        return jnp.argmax(model.apply(variables, X, rngs={'dropout': k}, train=False), axis=-1)
    keys = iter(jax.random.split(jax.random.PRNGKey(52), math.ceil(len(Y) / batch_size)))
    preds = [apply(X[i:min(i + batch_size, len(Y))], next(keys)) for i in trange(0, len(Y), batch_size)]
    return metrics.accuracy_score(Y, jnp.concatenate(preds))


def train_step(opt, loss):
    """The training function using optax, also returns the training loss"""
    @jax.jit
    def _apply(params, opt_state, X, Y):
        (loss_val, batch_stats), grads = jax.value_and_grad(loss, has_aux=True)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return freeze({'params': params['params'], 'batch_stats': batch_stats['batch_stats']}), opt_state, loss_val
    return _apply


def robust_train_step(opt, loss, epsilon=0.3, lr=0.001, steps=40):
    """AT training step proposed in https://arxiv.org/pdf/1706.06083.pdf"""
    @jax.jit
    def _apply(params, opt_state, X, Y):
        X_nat = X
        for _ in range(steps):
            grads = jax.grad(loss, argnums=1)(params, X, Y)
            X = X + lr * jnp.sign(grads)
            X = jnp.clip(X, X_nat - epsilon, X_nat + epsilon)
            X = jnp.clip(X, 0, 1)
        loss_val, grads = jax.value_and_grad(loss)(params, X, Y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val
    return _apply


def normalize(x, mu, sigma):
    return (x - mu) / sigma


def process(batch):
    batch['X'] = np.array(
        [
            normalize(
                np.array(x.resize((224, 224)).convert('RGB'), dtype=np.float32) / 255.0,
                np.array([0.485, 0.456, 0.406], dtype=np.float32),
                np.array([0.229, 0.224, 0.225], dtype=np.float32)
            )
            for x in batch['X']
        ]
    )
    batch['Y'] = np.array(batch['Y'])
    return batch


def load_dataset():
    """Load and preprocess the imagenet dataset"""
    ds = datasets.load_dataset('imagenet-1k', ignore_verifications=True, use_auth_token=True)
    ds = ds.rename_column('image', 'X')
    ds = ds.rename_column('label', 'Y')
    ds.set_transform(process)
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a model to be attacked.")
    parser.add_argument('--model', type=str, default="Softmax", help="Model to train.")
    parser.add_argument('--steps', type=int, default=3000, help="Steps of training to perform.")
    parser.add_argument('--grad-steps', type=int, default=1, help="Number of steps for gradient production.")
    parser.add_argument('--checkpoint', action="store_true", help="Skip training and only make gradients.")
    parser.add_argument('--robust', action="store_true", help="Perform adversarially robust training.")
    parser.add_argument('--dp', type=float, nargs='*', help="Perform differentially private training.")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size of the final gradient.")
    args = parser.parse_args()

    ds = load_dataset()
    model = getattr(models, args.model)()
    key = jax.random.PRNGKey(42)
    key, pkey = jax.random.split(key)
    params = model.init(pkey, ds['train'][:32]['X'])
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
    if args.robust:
        trainer = robust_train_step(opt, loss)
    else:
        trainer = train_step(opt, loss)
    rng = np.random.default_rng()
    train_len = len(ds['train'])
    fn = "data/{}{}{}.variables".format(
        args.model, '-robust' if args.robust else '',
        f'-dp-S{S}-sigma{sigma}' if args.dp is not None else ''
    )
    if args.checkpoint:
        with open(fn, 'rb') as f:
            params = serialization.from_bytes(params, f.read())
    else:
        for _ in (pbar := trange(args.steps)):
            idx = rng.choice(train_len, 32, replace=False)
            params, opt_state, loss_val = trainer(
                params, opt_state, ds['train'][idx]['X'], ds['train'][idx]['Y']
            )
            pbar.set_postfix_str(f"LOSS: {loss_val:.5f}")
        validation_batch = ds['validation'][:10_000]
        print(f"Final accuracy: {accuracy(model, params, validation_batch['X'], validation_batch['Y']):.3%}")
        os.makedirs('data', exist_ok=True)
        with open(fn, 'wb') as f:
            f.write(serialization.to_bytes(params))
        print(f'Saved final model to {fn}')
    # Generate the gradients to be attacked
    new_params = params
    for _ in (pbar := trange(args.grad_steps)):
        idx = rng.choice(train_len, args.batch_size, replace=False)
        new_params, opt_state, loss_val = trainer(
            new_params, opt_state, ds['train'][idx]['X'], ds['train'][idx]['Y']
        )
    grads = jax.tree_util.tree_map(operator.sub, params, new_params)
    fn = "data/{}{}{}.grads".format(
        args.model, '-robust' if args.robust else '',
        f'-dp-S{S}-sigma{sigma}' if args.dp is not None else ''
    )
    with open(fn, 'wb') as f:
        f.write(serialization.to_bytes(grads))
    print(f'Saved final gradient to {fn}')
