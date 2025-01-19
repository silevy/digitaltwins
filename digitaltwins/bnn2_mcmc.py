# NB: this example is ported from https://github.com/ctallec/pyvarinf/blob/master/main_regression.ipynb
import numpy as np; np.random.seed(0)
import tqdm
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, Predictive, init_to_uniform, init_to_feasible
import os 
import matplotlib.pyplot as plt

# Define the neural network module
class Net:
    def __init__(self, n_units):
        self.n_units = n_units

    def init_layer_params(self, input_dim, output_dim, key):
        """Initialize weights and biases."""
        w_key, b_key = random.split(key)
        weights = random.normal(w_key, shape=(input_dim, output_dim)) * jnp.sqrt(1 / input_dim)
        biases = jnp.zeros(output_dim)
        return weights, biases

    def forward(self, x, params):
        """Forward pass."""
        w1, b1 = params["layer1"]
        w2, b2 = params["layer2"]
        w_mean, b_mean = params["mean"]
        w_rho, b_rho = params["rho"]

        # First dense layer
        x = jnp.matmul(x, w1) + b1
        x = jnp.maximum(0, x)  # ReLU activation

        # Second dense layer
        x = jnp.matmul(x, w2) + b2
        x = jnp.maximum(0, x)  # ReLU activation

        # Output layers
        mean = jnp.matmul(x, w_mean) + b_mean
        rho = jnp.matmul(x, w_rho) + b_rho
        return mean.squeeze(), rho.squeeze()

    def init_params(self, input_dim, key):
        """Initialize all parameters."""
        keys = random.split(key, num=4)
        return {
            "layer1": self.init_layer_params(input_dim, self.n_units, keys[0]),
            "layer2": self.init_layer_params(self.n_units, self.n_units, keys[1]),
            "mean": self.init_layer_params(self.n_units, 1, keys[2]),
            "rho": self.init_layer_params(self.n_units, 1, keys[3]),
        }

# Data generation
def generate_data(n_samples):
    x = np.random.normal(size=n_samples)
    y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
    return x[:, None], y  # x reshaped for compatibility with dense layers

# Define the probabilistic model
# def model(x, y=None):
#     net = Net(n_units=32)
#     input_dim = x.shape[-1]
#     params = net.init_params(input_dim, random.PRNGKey(0))

#     # Define priors for network weights and biases
#     for layer_name, (weights, biases) in params.items():
#         numpyro.sample(f"{layer_name}_weights", dist.Normal(0, .1).expand(weights.shape).to_event(weights.ndim))
#         numpyro.sample(f"{layer_name}_biases", dist.Normal(0, .1).expand(biases.shape).to_event(biases.ndim))

#     # Forward pass
#     mean, rho = net.forward(x, params)
#     sigma = jnp.log1p(jnp.exp(rho))  # Softplus for positive scale
#     numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)

def nonlin(x):
    return jnp.maximum(0.0, x)

def model(X: np.ndarray, Y: np.ndarray, D_H: int, D_Y : int=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), 0.1 * jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), 0.1 *  jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w_mean = numpyro.sample("w_mean", dist.Normal(jnp.zeros((D_H, D_Y)), 0.1 *  jnp.ones((D_H, D_Y))))
    assert w_mean.shape == (D_H, D_Y)
    mean = jnp.matmul(z2, w_mean)  # <= output 1 of the neural network
    assert mean.shape == (N, D_Y)

    w_rho = numpyro.sample("w_rho", dist.Normal(jnp.zeros((D_H, D_Y)), 0.1 * jnp.ones((D_H, D_Y))))
    assert w_rho.shape == (D_H, D_Y)
    rho = jnp.matmul(z2, w_rho)  # <= output 2 of the neural network
    assert rho.shape == (N, D_Y)
    
    if Y is not None:
        assert mean.shape == Y.shape
        assert rho.shape == Y.shape

    mean = mean.squeeze()
    rho = rho.squeeze()
    numpyro.sample("Y", dist.Normal(mean, rho), obs=Y)
    # observe data
    # with numpyro.plate("data", N):
    #     # note we use to_event(1) because each observation has shape (1,)
    #     numpyro.sample("Y", dist.Normal(mean, rho).to_event(1), obs=Y)



# MCMC inference
n_train_data = 5000
x_train, y_train = generate_data(n_train_data)

kernel = NUTS(model=model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(random.PRNGKey(0), x_train, np.expand_dims(y_train, axis=1), 32)

# Posterior predictions
posterior_samples = mcmc.get_samples()
mcmc.print_summary()
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
plt.savefig("results/mcmc/posterior_predictive_mcmc.png")
plt.show()


# # Calculate mean and 95% credible intervals
# y_pred_mean = jnp.mean(y_pred, axis=0)
# y_pred_lower = jnp.percentile(y_pred, 2.5, axis=0)
# y_pred_upper = jnp.percentile(y_pred, 97.5, axis=0)

# # Plot the results
# os.makedirs("results/mcmc", exist_ok=True)
# plt.figure(figsize=(10, 6))
# plt.scatter(x_test.squeeze(), y_test, color='blue', alpha=0.5, label="True Data")
# plt.plot(x_test.squeeze(), y_pred_mean, color='red', label="Predicted Mean", lw=2)
# plt.fill_between(
#     x_test.squeeze(),
#     y_pred_lower,
#     y_pred_upper,
#     color='red',
#     alpha=0.2,
#     label="95% Credible Interval"
# )
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Posterior Predictive Distribution with 95% Credible Interval")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.savefig("results/mcmc/posterior_predictive_mcmc.png")
# plt.show()

print("Done.")


