from flax import linen as nn
import jax.numpy as jnp
import numpy as np; np.random.seed(0)
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, Predictive, init_to_uniform, init_to_feasible, init_to_median
import os 
import matplotlib.pyplot as plt

# Define the neural network using Flax
class Net(nn.Module):
    n_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units)(x[..., None])  # Expand dimension for compatibility
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(1)(x)
        rho = nn.Dense(1)(x)
        return mean.squeeze(), rho.squeeze()

# Data generation
def generate_data(n_samples):
    x = np.random.normal(size=n_samples)
    y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
    return x[:, None], y  # Reshape x for compatibility with Flax

# Define the model
def model(x, y=None):
    module = Net(n_units=32)
    net = random_flax_module("nn", module, dist.Normal(0, 0.1), input_shape=(1,))
    mean, rho = net(x)
    sigma = jnp.log1p(jnp.exp(rho))  # Softplus for positive scale
    numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)

# MCMC inference
n_train_data = 5000
x_train, y_train = generate_data(n_train_data)

kernel = NUTS(model, init_strategy=init_to_feasible)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(random.PRNGKey(0), x_train, y_train)

# Posterior predictions
posterior_samples = mcmc.get_samples()
n_test_data = 100
x_test, y_test = generate_data(n_test_data)
predictive = Predictive(model, posterior_samples)
y_pred = predictive(random.PRNGKey(1), x_test)["obs"]

# Calculate mean and 95% credible intervals
y_pred_mean = jnp.mean(y_pred, axis=0)
y_pred_lower = jnp.percentile(y_pred, 2.5, axis=0)
y_pred_upper = jnp.percentile(y_pred, 97.5, axis=0)

# Evaluate predictions
rmse = np.sqrt(np.mean(np.square(y_test - np.mean(y_pred, axis=0))))
print("Root Mean Squared Error:", rmse)

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
plt.savefig("results/mcmc/posterior_predictive_mcmc_flax.png")
plt.show()
