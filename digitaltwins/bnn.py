from numpyro.contrib.module import random_flax_module
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)
rng = jax.random.PRNGKey(0)
# --- synthetic data
num_observations = 100
a = 0.3
b = 2.5
s = 0.2
x = jnp.linspace(-1, 1, num=num_observations)
eps = jax.random.normal(jax.random.PRNGKey(1), shape=x.shape)
y = a + jnp.sin(x * 2 * jnp.pi / b) + s * eps

# --- constants we'll use
NUM_CHAINS = 4
MCMC_KWARGS = dict(num_warmup=1000, num_samples=1000, num_chains=NUM_CHAINS)

def model(n_obs, x, y=None):
    mlp = random_flax_module(
        "mlp",
        MLP([5, 10, 1], (1,)),
        prior={
            "Dense_0.bias": dist.Cauchy(),
            "Dense_0.kernel": dist.Normal(),
            "Dense_1.bias": dist.Cauchy(),
            "Dense_1.kernel": dist.Normal(),
            "Dense_2.bias": dist.Cauchy(),
            "Dense_2.kernel": dist.Normal(),
        },
        # Or, if using the same prior for all parameters, we can do:
        #  prior=dist.Normal(),
        input_shape=(1,),
    )
    sigma = numpyro.sample("sigma", dist.LogNormal())
    with numpyro.plate("obs", n_obs):
        mu = mlp(x).squeeze()
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

def linear_regression(x, y=None):
    alpha = numpyro.sample("alpha", dist.Normal())
    beta = numpyro.sample("beta", dist.Normal())
    sigma = numpyro.sample("sigma", dist.LogNormal())

    # --- likelihood
    mu = alpha + beta * x
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

kernel = NUTS(linear_regression)
mcmc = MCMC(kernel, **MCMC_KWARGS)
rng, _ = jax.random.split(rng)
# mcmc.run(rng, x, y)
# mcmc.print_summary()

# alpha starts at 0.0, beta at 0.0, sigma at 1.0


def linear_regression_potential(parameters, x, y):
    alpha, beta, sigma = parameters
    # --- priors
    log_prob = (
        dist.Normal().log_prob(alpha)
        + dist.Normal().log_prob(beta)
        + dist.LogNormal().log_prob(sigma)
    )
    # --- likelihood
    mu = alpha + beta * x
    log_likelihood = dist.Normal(mu, sigma).log_prob(y).sum()
    return - (log_prob + log_likelihood)

from functools import partial

potential = partial(linear_regression_potential, x=x, y=y)
kernel = NUTS(potential_fn=potential)
mcmc = MCMC(kernel, **MCMC_KWARGS)

# alpha starts at 0.0, beta at 0.0, sigma at 1.0
single_chain_init = jnp.array([0.0, 0.0, 1.0])
init_params = jnp.tile(single_chain_init, (NUM_CHAINS, 1))

rng, _ = jax.random.split(rng)
# mcmc.run(rng, init_params=init_params)

print("Done.")

# Re-usable logic & nicely structured samples
# We now know how sampling with a custom potential works, but there are 3 tasks we need to address

# 1. Write our code with re-usable logic,
# 2. Obtain nicely structured samples,
# 3. Provide different (and sensible) initializations for each chain.

# To solve our first two problems, it’s useful to know that the argument we feed to our potential function can in fact be of any type. The samples produced by numpyro will maintain the structure of such type. For example, the potential function in the previous section was a function of a jnp.ndarray, so the samples came out as a jnp.ndarray. But if we write our potential function in such a way that it takes a dictionary (or any mapping) as an argument, then the samples will be given to us in the form of a dictionary. To solve the 3rd problem I will initialize each parameter to a random sample from its prior. This will give me different starting points for each chain, but the constraints of each parameter will not be broken (for example, if my parameter has a log-normal prior, I wouldn’t want to initialize it to a negative value as this would be inconsistent with the log-normal distribution.)

# So here’s the solution I came up with. There are probably nicer ways of doing this, but this is the solution that works for me. At least for now. I will specify my models with 3 ingredients:

# 1. A dictionary of prior distributions
# 2. A log-likelihood function
# 3. A dictionary of parameter shapes


priors = {
    "alpha": dist.Normal(),
    "beta": dist.Normal(),
    "sigma": dist.LogNormal(),
}

def linear_regression_log_likelihood(values: dict, x: jnp.ndarray, y: jnp.ndarray):
    alpha = values["alpha"]
    beta = values["beta"]
    sigma = values["sigma"]
    mu = alpha + beta * x
    return dist.Normal(mu, sigma).log_prob(y)

def evaluate_densities_at_values(densities: dict, values: dict):
    evaluate = lambda values, density: density.log_prob(values)
    return jax.tree_map(evaluate, values, densities)


def linear_regression_potential_v2(
    parameter_values: dict,
    parameter_priors: dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    # -- priors
    logprobas = evaluate_densities_at_values(
        parameter_priors,
        parameter_values,
    )
    logprobas, _ = jax.flatten_util.ravel_pytree(logprobas)
    
    # -- likelihood
    log_likelihood = linear_regression_log_likelihood(
        parameter_values,
        x,
        y,
    )    
    # potential
    return -(logprobas.sum() + log_likelihood.sum())

# freeze the priors and the observed data
potential = partial(
    linear_regression_potential_v2,
    parameter_priors=priors,
    x=x,
    y=y,
)
kernel = NUTS(potential_fn=potential)
mcmc = MCMC(kernel, **MCMC_KWARGS)

shapes = {
    "alpha": (),
    "beta": (),
    "sigma": (),
}

def make_rng_tree(rng, tree, is_leaf=None):
    """
    Provide a random seed for each leaf in the tree.
    """
    # hack because jax.tree_structure does not allow a `is_leaf` kwarg.
    raw_tree = jax.tree_map(lambda x: 1, tree, is_leaf=is_leaf)
    structure = jax.tree_structure(raw_tree)
    subkeys = jax.random.split(rng, structure.num_leaves)
    iter_subkeys = iter(subkeys)
    return jax.tree_map(lambda x: next(iter_subkeys), raw_tree)

def init_to_sample(rng,
    parameter_shapes,
    parameter_densities,
    num_chains=NUM_CHAINS
):
    parameter_seeds = make_rng_tree(
        rng,
        parameter_densities,
        is_leaf=lambda x: isinstance(x, dist.Distribution)
    )
    sample_ = lambda d, s, k: d.sample(k, sample_shape=(num_chains, *s))

    return jax.tree_map(
        sample_,
        parameter_densities,
        parameter_shapes,
        parameter_seeds,
        is_leaf=lambda x: isinstance(x, dist.Distribution)
    )

rng, init_rng, mcmc_rng = jax.random.split(rng, 3)
init_params = init_to_sample(init_rng, shapes, priors)

mcmc.run(mcmc_rng, init_params=init_params, chain_method='parallel')
mcmc.print_summary()


import flax.linen as nn
from flax.core import FrozenDict

class MLP(nn.Module):
    layers: list
    input_shape: tuple
    
    @nn.compact
    def __call__(self, x):
        for num_features in self.layers[:-1]:
            x = nn.softplus(nn.Dense(num_features)(x))
        x = nn.Dense(layers[-1])(x)
        return x
    
    def get_priors(self):
        """
        L2 Regularisation
        """
        structure = self._random_init()["params"]
        priors = jax.tree_map(lambda x: dist.Normal(), structure)
        return FrozenDict(priors)
    
    def get_shapes(self):
        init = self._random_init()["params"]
        shapes = jax.tree_map(lambda x: x.shape, init)
        return FrozenDict(shapes)
 
    
    def _random_init(self):
        rng = jax.random.PRNGKey(0)
        return self.init(rng, jnp.ones(self.input_shape))

output_shape = (1,)
input_shape = (1,)
layers = [5, 10, *output_shape] # Yes, it's a very small MLP.
mlp = MLP(layers, input_shape)

mlp_priors = mlp.get_priors()
other_priors = {"sigma": dist.LogNormal()}
# --- combine all parameters
priors = {**mlp_priors.unfreeze(), **other_priors}

def mlp_potential(
    parameter_values: dict,
    parameter_priors: dict,
    apply_fn: callable,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    log_prior = evaluate_densities_at_values(
        parameter_priors,
        parameter_values,
    )
    log_prior, _ = jax.flatten_util.ravel_pytree(log_prior)
    mu = apply_fn({"params": parameter_values}, x).squeeze()
    log_likelihood = dist.Normal(mu, parameter_values["sigma"]).log_prob(y)    
    return -(log_prior.sum() + log_likelihood.sum())

# Freeze the data/known
potential = partial(
    mlp_potential,
    parameter_priors=priors,
    apply_fn=mlp.apply,
    x=x.reshape(-1, 1),
    y=y
)
kernel = NUTS(potential_fn=potential)
mcmc = MCMC(kernel, **MCMC_KWARGS)

mlp_shapes = mlp.get_shapes()
other_shapes = {"sigma": ()}
shapes = {**mlp_shapes.unfreeze(), **other_shapes}
rng, init_rng, mcmc_rng = jax.random.split(rng, 3)
init_params = init_to_sample(rng, shapes, priors)

mcmc.run(mcmc_rng, init_params=init_params)

samples_by_chain = mcmc.get_samples(True)
diagnostics = jax.tree_map(
    numpyro.diagnostics.gelman_rubin,
    samples_by_chain,
)

x_pred = jnp.linspace(-1.3, 1.3, num=100) # Original data range was (-1, 1)
predict_fn = jax.vmap(
    lambda sample: mlp.apply({"params": sample}, x_pred.reshape(-1, 1)).squeeze(),
)
mu_samples = predict_fn(mcmc.get_samples()) # The network autmatically ignores the parameter `sigma`
sigma_samples = mcmc.get_samples()["sigma"]
y_pred = dist.Normal(mu_samples, sigma_samples.reshape(-1, 1)).sample(rng)