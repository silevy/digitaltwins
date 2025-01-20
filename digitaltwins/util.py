import numpy
import flax.linen as nn
from jax import random, numpy as jnp
import numpyro as npr
import numpyro.distributions as dist

from . import main



##################
# AMORTIZING NNs #
##################


import jax
import jax.numpy as jnp
from jax.nn import tanh, softplus

def init_layer_params(input_dim, output_dim, key):
    """Initialize parameters for a single dense layer."""
    w_key, b_key = jax.random.split(key)
    weights = jax.random.normal(w_key, shape=(input_dim, output_dim)) * jnp.sqrt(1 / input_dim)
    # biases = jax.random.normal(b_key, shape=(output_dim,))
    biases = jnp.zeros(output_dim)  # Initialize biases to zeros like Flax
    return weights, biases

def init_ideal_point_nn(input_dim, hidden_dim1, hidden_dim2, output_dim, key):
    """Initialize all parameters for the IdealPointNN."""
    keys = jax.random.split(key, num=4)
    params = {
        "z_layer": init_layer_params(input_dim, hidden_dim1, keys[0]),
        "z_mu_layer": init_layer_params(hidden_dim1, output_dim, keys[1]),
        "z_sig_layer": init_layer_params(hidden_dim1, output_dim, keys[2]),
        # "z_norm_layer": {
        #     "z_scale": jnp.ones(hidden_dim1),
        #     "z_bias": jnp.zeros(hidden_dim1),
        # },
    }
    return params


def init_ideal_point_nn(input_dim, hidden_dim1, hidden_dim2, output_dim, key):
    """Initialize all parameters for the IdealPointNN."""
    keys = jax.random.split(key, num=4)
    params = {
        "phi_layer": init_layer_params(input_dim, hidden_dim1, keys[0]),
        "phi_mu_layer": init_layer_params(hidden_dim1, output_dim, keys[1]),
        "phi_sig_layer": init_layer_params(hidden_dim1, output_dim, keys[2]),
        # "phi_norm_layer": {
        #     "phi_scale": jnp.ones(hidden_dim1),
        #     "phi_bias": jnp.zeros(hidden_dim1),
        # },
    }
    return params

def layer_norm(x, scale, bias):
    """Apply layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(variance + 1e-6)
    return scale * normalized + bias

def forward_ideal_point_nn(x, params):
    """Forward pass through the IdealPointNN."""
    # Layer normalization
    # norm_params = params["z_norm_layer"]
    # x = layer_norm(x, norm_params["z_scale"], norm_params["z_bias"])
    
    # Hidden layer
    w, b = params["z_layer"]
    x = tanh(jnp.dot(x, w) + b)
    
    # Output layers for concentration and rate
    mu_w, mu_b = params["z_mu_layer"]
    sig_w, sig_b = params["z_sig_layer"]
    
    mu = jnp.dot(x, mu_w) + mu_b
    sig = softplus(jnp.dot(x, sig_w) + sig_b)
    
    return mu, sig

def forward_phi_nn(x, params):
    """Forward pass through the IdealPointNN."""
    # Layer normalization
    # norm_params = params["phi_norm_layer"]
    # x = layer_norm(x, norm_params["phi_scale"], norm_params["phi_bias"])
    
    # Hidden layer
    w, b = params["phi_layer"]
    x = tanh(jnp.dot(x, w) + b)
    
    # Output layers for concentration and rate
    mu_w, mu_b = params["phi_mu_layer"]
    sig_w, sig_b = params["phi_sig_layer"]
    
    concentration = softplus(jnp.dot(x, mu_w) + mu_b)
    rate = softplus(jnp.dot(x, sig_w) + sig_b)
    
    return concentration, rate


# class IdealPointNNFlax(nn.Module):
#     params: dict

#     @nn.compact
#     def __call__(self, x):
#         return forward_ideal_point_nn(x, self.params)

# class PhiNNFlax(nn.Module):
#     params: dict

#     @nn.compact
#     def __call__(self, x):
#         return forward_phi_nn(x, self.params)





class PhiNN(nn.Module):
    hidden_size1: int
    hidden_size2: int
    output_size: int

    def setup(self):
        self.norm_layer = nn.LayerNorm()
        # self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        # self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
        self.layer = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
        self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

    # @nn.compact
    def __call__(self, x):
        x = self.norm_layer(x)
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer(x))
        concentration = jnp.log1p(jnp.exp(self.mu_layer(x)))
        rate = jnp.log1p(jnp.exp(self.sig_layer(x)))
        return concentration, rate

class PhiNN_mcmc(nn.Module):
    hidden_size1: int
    hidden_size2: int
    output_size: int

    def setup(self):
        self.norm_layer = nn.LayerNorm()
        # self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        # self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
        self.layer = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
        self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

    # @nn.compact
    def __call__(self, x):
        # x = self.norm_layer(x[...,None])
        x = self.norm_layer(x)
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer(x))
        concentration = jnp.log1p(jnp.exp(self.mu_layer(x)))
        rate = jnp.log1p(jnp.exp(self.sig_layer(x)))
        return concentration, rate


class IdealPointNN_mcmc(nn.Module):
    hidden_size1: int
    hidden_size2: int
    output_size: int

    def setup(self):
        self.norm_layer = nn.LayerNorm()
        # self.norm_layer = nn.LayerNorm(feature_axes=-1)  # Normalize last dimension (it's default)
        # self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        # self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
        self.layer = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
        self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

    # @nn.compact
    def __call__(self, x):
        # x = x.reshape(-1, x.shape[-1])  # Flatten all dimensions except the last one
        # print(f"shape of x: {x.shape}")
        x = self.norm_layer(x)
        # x = self.norm_layer(x[...,None])
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer(x))
        concentration = self.mu_layer(x)
        rate = jnp.log1p(jnp.exp(self.sig_layer(x)))
        return concentration, rate

class IdealPointNN(nn.Module):
    hidden_size1: int
    hidden_size2: int
    output_size: int

    def setup(self):
        # self.norm_layer = nn.LayerNorm()
        # self.norm_layer = nn.LayerNorm(feature_axes=-1)  # Normalize last dimension (it's default)
        # self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        # self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
        self.layer = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
        self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

    # @nn.compact
    def __call__(self, x):
        # x = x.reshape(-1, x.shape[-1])  # Flatten all dimensions except the last one
        # print(f"shape of x: {x.shape}")
        # x = self.norm_layer(x)
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer(x))
        concentration = self.mu_layer(x)
        rate = jnp.log1p(jnp.exp(self.sig_layer(x)))
        return concentration, rate
    


class PhiLinear(nn.Module):
    output_size: int

    def setup(self):
        # self.norm_layer = nn.LayerNorm()
        self.concentration_layer = nn.Dense(self.output_size)
        self.rate_layer = nn.Dense(self.output_size)

    def __call__(self, x):
        # x = self.norm_layer(x)
        concentration = self.concentration_layer(x)
        rate = jnp.exp(self.rate_layer(x))
        return concentration, rate



class IdealPointLinear(nn.Module):
    output_size: int

    def setup(self):
        self.norm_layer = nn.LayerNorm()
        self.mu_layer = nn.Dense(self.output_size)
        self.sig_layer = nn.Dense(self.output_size)

    def __call__(self, x):
        x = self.norm_layer(x)
        mu = self.mu_layer(x)
        sig = jnp.log1p(jnp.exp(self.sig_layer(x)))
        return mu, sig



def simulated_data(rng_key, svi_object, N=10000, J=65, L=25, T=20):
    # simulated data
    rng_key, rng_key_sim = random.split(rng_key, 2)

    Y_logits_11 = random.normal(rng_key_sim, shape=[N, J, T, main.H_CUTOFFS["11"]])
    Y_logits_10 = random.normal(rng_key_sim, shape=[N, J, T, main.H_CUTOFFS["10"]])
    Y_logits_5 = random.normal(rng_key_sim, shape=[N, J, T, main.H_CUTOFFS["5"]])
    Y_logits_2 = random.normal(rng_key_sim, shape=[N, J, T, main.H_CUTOFFS["2"]])
    Y_11 = random.categorical(rng_key_sim, Y_logits_11, axis=3)
    Y_10 = random.categorical(rng_key_sim, Y_logits_10, axis=3)
    Y_5 = random.categorical(rng_key_sim, Y_logits_5, axis=3)
    Y_2 = random.categorical(rng_key_sim, Y_logits_2, axis=3)
    
    c_a = jnp.zeros([len(main.H_CUTOFFS)])
    c_b = jnp.ones([len(main.H_CUTOFFS)])
    
    # create indexing collections, should be static primitives
    Y = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_11, Y_10, Y_5, Y_2) ) }
    J = numpy.asarray([int(y.shape[1]) for y in Y.values()])
    J_dict = { k : v for (k, v) in zip(main.H_CUTOFFS.keys(), J.tolist()) }
    J_idx_end = { k : i for (k, i) in zip( main.H_CUTOFFS.keys(), numpy.cumsum(J, dtype=numpy.int32) ) }
    J_idx_start = { k : i for (k, i) in \
                    zip( main.H_CUTOFFS.keys(), numpy.concatenate([numpy.zeros([1], dtype=numpy.int32), numpy.cumsum(J[:-1], dtype=numpy.int32)]) ) }
    J_total = list(J_idx_end.values())[-1]
    
    # initialize SVI
    rng_key, rng_key_init = random.split(rng_key, 2) 
    return svi_object.init(rng_key_init, 
                                Y_11=Y_11,
                                Y_10=Y_10,
                                Y_5=Y_5,
                                Y_2=Y_2,
                                c_a=c_a, 
                                c_b=c_b,
                                N=N, L=L, T=T,
                                J_total = J_total,
                                J_dict = J_dict,
                                J_idx_start = J_idx_start,
                                J_idx_end = J_idx_end
                                )