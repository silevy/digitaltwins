from flax import linen as nn
import jax.numpy as jnp
import numpy as np; np.random.seed(0)
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.infer import MCMC, NUTS, Predictive, init_to_uniform, init_to_feasible, init_to_median, init_to_sample
import os 
import matplotlib.pyplot as plt
import jax

numpyro.set_platform("cpu")

# Define the neural network using Flax
class Net(nn.Module):
    n_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units, use_bias=False)(x)  # Expand dimension for compatibility
        x = nn.tanh(x)
        x = nn.Dense(self.n_units, use_bias=False)(x)
        x = nn.tanh(x)
        mean = nn.Dense(1, use_bias=False)(x)
        rho = nn.Dense(1, use_bias=False)(x)
        return mean, rho

class NetMeanOnly(nn.Module):
    n_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_units, use_bias=False)(x)  # Expand dimension for compatibility
        x = nn.tanh(x)
        x = nn.Dense(self.n_units, use_bias=False)(x)
        x = nn.tanh(x)
        mean = nn.Dense(1, use_bias=False)(x)
        return mean
    
# Data generation
# def generate_data(n_samples):
#     # x = np.random.normal(size=n_samples)
#     x = jnp.linspace(-1, 1, n_samples)
#     y = np.cos(x * 3) + np.random.normal(size=n_samples) * np.abs(x) / 2
#     return x[:, None], y  # Reshape x for compatibility with Flax

# Function to generate data using JAX
def generate_data(n_samples, rng):
    # Generate linearly spaced x values
    x = jnp.linspace(-3, 3, n_samples)
    # Split RNG key for different random operations
    rng, rng_noise = jax.random.split(rng)
    # Generate noise with JAX
    noise = jax.random.normal(rng_noise, shape=(n_samples,))
    # Compute y values with noise
    y = jnp.cos(x * 3) + noise * jnp.abs(x) / 2
    # Reshape x for compatibility
    return x[:, None], y

# create artificial regression dataset
def generate_data2(n_samples, init_x, rng):
    """
    Generate artificial regression dataset using JAX.

    Args:
        N (int): Number of data points.
        rng (jax.random.PRNGKey): Random key for reproducibility.

    Returns:
        tuple: X (inputs), Y (outputs), X_test (test inputs).
    """
    N = n_samples
    D_X = 3  # Number of input dimensions
    D_Y = 1  # Number of output dimensions
    sigma_obs = 0.05

    if rng is None:
        raise ValueError("A JAX random key (rng) must be provided.")

    # Generate X values
    X = jnp.linspace(-init_x, init_x, N)
    X = jnp.power(X[:, jnp.newaxis], jnp.arange(D_X))

    # Generate random weights W
    rng, rng_W, rng_noise = jax.random.split(rng, 3)
    W = 0.5 * jax.random.normal(rng_W, shape=(D_X,))

    # Generate Y values
    if X.shape[1] == 1:
        Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 0], 2.0) * jnp.sin(4.0 * X[:, 0])
    if X.shape[1] > 1:
        Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    noise = sigma_obs * jax.random.normal(rng_noise, shape=(N,))
    Y += noise
    Y = Y[:, jnp.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)

    return X, Y


# Define the model
def model(X, Y=None):
    N = X.shape[0]
    D_X = X.shape[1]
    module = Net(n_units=5)
    net = random_flax_module("nn", module, dist.Normal(0, 1), input_shape=(N, D_X))
    mean, rho = net(X)
    sigma = nn.softplus(rho)
    # sigma = jnp.log1p(jnp.exp(rho))  # Softplus for positive scale
    # numpyro.sample("obs", dist.Normal(mean, sigma), obs=Y)
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=Y)

# Define the model
def model_mean_only(X, Y=None):
    N = X.shape[0]
    D_X = X.shape[1]
    module = NetMeanOnly(n_units=5)
    net = random_flax_module("nn", module, dist.Normal(0, 1), input_shape=(N, D_X))
    mean = net(X)
    # sigma = jnp.log1p(jnp.exp(rho))  # Softplus for positive scale
    # numpyro.sample("obs", dist.Normal(mean, sigma), obs=y)
    
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # numpyro.sample("obs", dist.Normal(mean, sigma_obs).to_event(1), obs=Y)

    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("obs", dist.Normal(mean, sigma_obs).to_event(1), obs=Y)

# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)

def model_mean_only_jax(X, Y=None, D_H=5, D_Y=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    assert w3.shape == (D_H, D_Y)
    z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (N, D_Y)

    if Y is not None:
        assert z3.shape == Y.shape

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("obs", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)


def model_jax(X, Y=None, D_H=5, D_Y=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    w1_sig = numpyro.sample("w1_sig", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1_sig.shape == (D_X, D_H)
    z1_sig = nonlin(jnp.matmul(X, w1_sig))  # <= first layer of activations
    assert z1_sig.shape == (N, D_H)

    # sample second layer
    w2_sig = numpyro.sample("w2_sig", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2_sig.shape == (D_H, D_H)
    z2_sig = nonlin(jnp.matmul(z1_sig, w2_sig))  # <= second layer of activations
    assert z2_sig.shape == (N, D_H)


    # sample final layer of weights and neural network output
    # w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    # assert w3.shape == (D_H, D_Y)
    # z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    # assert z3.shape == (N, D_Y)

    # if Y is not None:
    #     assert z3.shape == Y.shape

    # # we put a prior on the observation noise
    # prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    # sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # # observe data
    # with numpyro.plate("data", N):
    #     # note we use to_event(1) because each observation has shape (1,)
    #     numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)
    
    # Final layer for mean
    w_mean = numpyro.sample(
        "w_mean",
        dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y)))
    )
    # b_mean = numpyro.sample(
    #     "b_mean",
    #     dist.Normal(jnp.zeros((D_Y,)), jnp.ones((D_Y,)))
    # )
    # mean = jnp.matmul(z2, w_mean) + b_mean  # shape (N, D_Y)
    mean = jnp.matmul(z2, w_mean)   # shape (N, D_Y)
    
    # Final layer for rho
    w_rho = numpyro.sample(
        "w_rho",
        dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y)))
    )
    # b_rho = numpyro.sample(
    #     "b_rho",
    #     dist.Normal(jnp.zeros((D_Y,)), jnp.ones((D_Y,)))
    # )
    # rho = jnp.matmul(z2, w_rho) + b_rho  # shape (N, D_Y)
    rho = jnp.matmul(z2_sig, w_rho)  # shape (N, D_Y)

    # If we have observed Y, ensure shapes are consistent
    if Y is not None:
        assert mean.shape == Y.shape
        assert rho.shape == Y.shape

    # Convert rho -> sigma via a softplus function for positivity
    # mean = mean.squeeze(-1)   # shape (N,)
    # rho = rho.squeeze(-1)     # shape (N,)
    sigma = jnp.log1p(jnp.exp(rho))  # softplus to ensure positivity

    # numpyro.sample("Y", dist.Normal(mean, sigma), obs=Y)
    
    # # Observe Y
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=Y)

# target_model = model
# target_model = model_mean_only
target_model = model_jax
# target_model = model_mean_only_jax

# MCMC inference
n_train_data = 100
rng = jax.random.PRNGKey(42)
x_train, y_train = generate_data2(n_train_data, 1, rng)
n_test_data = 100
rng, subrng = jax.random.split(rng, 2)
x_test, y_test = generate_data2(n_test_data, 1.3, subrng)

kernel = NUTS(target_model)
# kernel = NUTS(model_mean_only, init_strategy=init_to_sample)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1, chain_method="vectorized")
mcmc.run(random.PRNGKey(0), x_train, y_train)

# Posterior predictions
posterior_samples = mcmc.get_samples()
predictive = Predictive(target_model, posterior_samples)
y_pred = predictive(random.PRNGKey(1), x_test)["obs"]

# Calculate mean and 95% credible intervals
y_pred_mean = jnp.mean(y_pred, axis=0)
y_pred_lower = jnp.percentile(y_pred, 2.5, axis=0)
y_pred_upper = jnp.percentile(y_pred, 97.5, axis=0)

# Evaluate predictions
rmse = np.sqrt(np.mean(np.square(y_test - np.mean(y_pred, axis=0))))
print("Root Mean Squared Error:", rmse)

# Sort x_test and corresponding predictions
# sorted_indices = jnp.argsort(x_test[:,1].squeeze())
# x_test_sorted = x_test[:,1].squeeze()[sorted_indices]
# y_pred_mean_sorted = y_pred_mean[sorted_indices]
# y_pred_lower_sorted = y_pred_lower[sorted_indices]
# y_pred_upper_sorted = y_pred_upper[sorted_indices]

# Plot the results
os.makedirs("results/mcmc", exist_ok=True)
plt.figure(figsize=(10, 6))

# Scatter plot of true data
plt.scatter(x_train[:,1], y_train[:,0], color='blue', alpha=0.5, label="True Data")

# Plot predicted mean
plt.plot(x_test[:,1], y_pred_mean[:,0], color='red', label="Predicted Mean", lw=2)

# Fill credible interval
plt.fill_between(
    x_test[:,1],
    y_pred_lower[:,0],
    y_pred_upper[:,0],
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
# plt.savefig("results/mcmc/posterior_predictive_mcmc_flax.png")
# plt.savefig("results/mcmc/posterior_predictive_mcmc_flax_meanonly.png")
plt.savefig(f"bnn_plot_{target_model.__name__}.pdf")
plt.show()




# helper function for prediction
def predict(model, rng_key, samples, X):
    model = numpyro.handlers.substitute(numpyro.handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = numpyro.handlers.trace(model).get_trace(X=X, Y=None)
    return model_trace["obs"]["value"]

# predict Y_test at inputs X_test
vmap_args = (
    posterior_samples,
    random.split(random.PRNGKey(1), 1000),
)
predictions = jax.vmap(
    lambda samples, rng_key: predict(target_model, rng_key, samples, x_test)
)(*vmap_args)
predictions = predictions[..., 0]

# compute mean prediction and confidence interval around median
mean_prediction = jnp.mean(predictions, axis=0)
percentiles = np.percentile(predictions, [2.5, 97.5], axis=0)

# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# plot training data
ax.plot(x_train[:, 1], y_train[:,0], "kx")
# plot 90% confidence level of predictions
ax.fill_between(
    x_test[:, 1], percentiles[0, :], percentiles[1, :], color="lightblue"
)
# plot mean prediction
ax.plot(x_test[:, 1], mean_prediction, "blue", ls="solid", lw=2.0)
ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 95% CI")

plt.savefig(f"bnn_plot2_{target_model.__name__}.pdf")