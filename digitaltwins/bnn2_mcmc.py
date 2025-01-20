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

# Data generation
def generate_data(n_samples):
    x = np.random.normal(size=n_samples)
    y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
    return x[:, None], y  # x reshaped for compatibility with dense layers

def nonlin(x):
    return jnp.maximum(0.0, x)

def model(X: np.ndarray, Y: np.ndarray, D_H: int, D_Y : int=1):
    """
    A 2-hidden-layer Bayesian MLP with biases:
        - hidden layer 1 weights/bias: w1, b1
        - hidden layer 2 weights/bias: w2, b2
        - final layer weights/bias for mean: w_mean, b_mean
        - final layer weights/bias for rho: w_rho, b_rho
    """
    N, D_X = X.shape

    # First hidden layer
    w1 = numpyro.sample(
        "w1",
        dist.Normal(jnp.zeros((D_X, D_H)), 0.1 * jnp.ones((D_X, D_H)))
    )
    b1 = numpyro.sample(
        "b1",
        dist.Normal(jnp.zeros((D_H,)), 0.1 * jnp.ones((D_H,)))
    )
    z1 = nonlin(jnp.matmul(X, w1) + b1)  # shape (N, D_H)

    # Second hidden layer
    w2 = numpyro.sample(
        "w2",
        dist.Normal(jnp.zeros((D_H, D_H)), 0.1 * jnp.ones((D_H, D_H)))
    )
    b2 = numpyro.sample(
        "b2",
        dist.Normal(jnp.zeros((D_H,)), 0.1 * jnp.ones((D_H,)))
    )
    z2 = nonlin(jnp.matmul(z1, w2) + b2)  # shape (N, D_H)

    # Final layer for mean
    w_mean = numpyro.sample(
        "w_mean",
        dist.Normal(jnp.zeros((D_H, D_Y)), 0.1 * jnp.ones((D_H, D_Y)))
    )
    b_mean = numpyro.sample(
        "b_mean",
        dist.Normal(jnp.zeros((D_Y,)), 0.1 * jnp.ones((D_Y,)))
    )
    mean = jnp.matmul(z2, w_mean) + b_mean  # shape (N, D_Y)

    # Final layer for rho
    w_rho = numpyro.sample(
        "w_rho",
        dist.Normal(jnp.zeros((D_H, D_Y)), 0.1 * jnp.ones((D_H, D_Y)))
    )
    b_rho = numpyro.sample(
        "b_rho",
        dist.Normal(jnp.zeros((D_Y,)), 0.1 * jnp.ones((D_Y,)))
    )
    rho = jnp.matmul(z2, w_rho) + b_rho  # shape (N, D_Y)
    
    # If we have observed Y, ensure shapes are consistent
    if Y is not None:
        assert mean.shape == Y.shape
        assert rho.shape == Y.shape

    # Convert rho -> sigma via a softplus function for positivity
    mean = mean.squeeze(-1)   # shape (N,)
    rho = rho.squeeze(-1)     # shape (N,)
    sigma = jnp.log1p(jnp.exp(rho))  # softplus to ensure positivity

    # Observe Y
    numpyro.sample("Y", dist.Normal(mean, sigma), obs=Y)


# def model(X: np.ndarray, Y: np.ndarray, D_H: int, D_Y : int=1):
#     N, D_X = X.shape

#     # sample first layer (we put unit normal priors on all weights)
#     w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), 0.1 * jnp.ones((D_X, D_H))))
#     assert w1.shape == (D_X, D_H)
#     z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
#     assert z1.shape == (N, D_H)

#     # sample second layer
#     w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), 0.1 * jnp.ones((D_H, D_H))))
#     assert w2.shape == (D_H, D_H)
#     z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
#     assert z2.shape == (N, D_H)

#     # sample final layer of weights and neural network output
#     w_mean = numpyro.sample("w_mean", dist.Normal(jnp.zeros((D_H, D_Y)), 0.1 * jnp.ones((D_H, D_Y))))
#     assert w_mean.shape == (D_H, D_Y)
#     mean = jnp.matmul(z2, w_mean)  # <= output 1 of the neural network
#     assert mean.shape == (N, D_Y)

#     w_rho = numpyro.sample("w_rho", dist.Normal(jnp.zeros((D_H, D_Y)), 0.1 * jnp.ones((D_H, D_Y))))
#     assert w_rho.shape == (D_H, D_Y)
#     rho = jnp.matmul(z2, w_rho)  # <= output 2 of the neural network
#     assert rho.shape == (N, D_Y)
    
#     if Y is not None:
#         assert mean.shape == Y.shape
#         assert rho.shape == Y.shape

#     mean = mean.squeeze()
#     rho = rho.squeeze()
#     sigma = jnp.log1p(jnp.exp(rho))  # Softplus for positive scale
#     numpyro.sample("Y", dist.Normal(mean, sigma), obs=Y)
    # observe data
    # with numpyro.plate("data", N):
    #     # note we use to_event(1) because each observation has shape (1,)
    #     numpyro.sample("Y", dist.Normal(mean, rho).to_event(1), obs=Y)



# MCMC inference
n_train_data = 5000
x_train, y_train = generate_data(n_train_data)

kernel = NUTS(model=model, init_strategy=init_to_feasible)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(random.PRNGKey(0), x_train, np.expand_dims(y_train, axis=1), 32)

# Posterior predictions
posterior_samples = mcmc.get_samples()
mcmc.print_summary()
n_test_data = 100
x_test, y_test = generate_data(n_test_data)
predictive = Predictive(model, posterior_samples)
y_pred = predictive(random.PRNGKey(1), x_test, Y=None, D_H=32)["Y"]

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


