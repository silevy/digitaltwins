# NB: this example is ported from https://github.com/ctallec/pyvarinf/blob/master/main_regression.ipynb
import numpy as np; np.random.seed(0)
import tqdm
from flax import linen as nn
from jax import jit, random
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO, autoguide, init_to_feasible
import os 
import matplotlib.pyplot as plt
import jax.numpy as jnp

class Net(nn.Module):
    n_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units)(x[..., None])
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(1)(x)
        rho = nn.Dense(1)(x)
        return mean.squeeze(), rho.squeeze()

def generate_data(n_samples):
    x = np.random.normal(size=n_samples)
    y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
    return x, y

def model(x, y=None, batch_size=None):
    module = Net(n_units=32)
    net = random_flax_module("nn", module, dist.Normal(0, .1), input_shape=())
    with numpyro.plate("batch", x.shape[0], subsample_size=batch_size):
        batch_x = numpyro.subsample(x, event_dim=0)
        batch_y = numpyro.subsample(y, event_dim=0) if y is not None else None
        mean, rho = net(batch_x)
        sigma = nn.softplus(rho)
        numpyro.sample("obs", dist.Normal(mean, sigma), obs=batch_y)

n_train_data = 20000
x_train, y_train = generate_data(n_train_data)
guide = autoguide.AutoNormal(model, init_loc_fn=init_to_feasible)
svi = SVI(model, guide, numpyro.optim.Adam(5e-3), TraceMeanField_ELBO())
n_iterations = 10000
svi_result = svi.run(random.PRNGKey(0), n_iterations, x_train, y_train, batch_size=256)
params, losses = svi_result.params, svi_result.losses
n_test_data = 100
x_test, y_test = generate_data(n_test_data)
predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
y_pred = predictive(random.PRNGKey(1), x_test[:100])["obs"].copy()
assert losses[-1] < 3000
assert np.sqrt(np.mean(np.square(y_test - y_pred))) < 1

# Evaluate predictions
rmse = np.sqrt(np.mean(np.square(y_test - np.mean(y_pred, axis=0))))
print("Root Mean Squared Error:", rmse)

# Calculate mean and 95% credible intervals
y_pred_mean = jnp.mean(y_pred, axis=0)
y_pred_lower = jnp.percentile(y_pred, 2.5, axis=0)
y_pred_upper = jnp.percentile(y_pred, 97.5, axis=0)

# Plot the results
os.makedirs("results/mcmc", exist_ok=True)

# Sort x_test and corresponding predictions
sorted_indices = jnp.argsort(x_test.squeeze())
x_test_sorted = x_test.squeeze()[sorted_indices]
y_pred_mean_sorted = y_pred_mean[sorted_indices]
y_pred_lower_sorted = y_pred_lower[sorted_indices]
y_pred_upper_sorted = y_pred_upper[sorted_indices]

# Plot the results
os.makedirs("results/mcmc", exist_ok=True)
plt.figure(figsize=(10, 6))

# Scatter plot of true data
plt.scatter(x_test.squeeze(), y_test, color='blue', alpha=0.5, label="True Data")

# Plot predicted mean
plt.plot(x_test_sorted, y_pred_mean_sorted, color='red', label="Predicted Mean", lw=2)

# Fill credible interval
plt.fill_between(
    x_test_sorted,
    y_pred_lower_sorted,
    y_pred_upper_sorted,
    color='red',
    alpha=0.2,
    label="95% Credible Interval"
)

# Add labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Posterior Predictive Distribution with 95% Credible Interval")
plt.legend()
plt.grid(alpha=0.3)

# Save the plot
plt.savefig("results/mcmc/posterior_predictive.png")
plt.show()


print("Done.")
