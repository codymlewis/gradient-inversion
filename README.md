# Gradient Inversion

Some jax implementations of the gradient inversion attacks against federated learning

## Running

First install the [JAX library](https://github.com/google/jax) then use pip to install the requirements from the requirements.txt file.

Then train and save a model and gradient by running `python create_model.py`.

And finally run the attack of choice, e.g. `python attack.py -a dlg` for the deep leakage from gradients attack.
