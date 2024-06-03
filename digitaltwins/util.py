import numpy
import flax.linen as nn
from jax import random, numpy as jnp

from . import main



##################
# AMORTIZING NNs #
##################

# class PhiNN(nn.Module):
#     hidden_size1: int
#     hidden_size2: int
#     output_size: int
    
#     def setup(self):
#         self.norm_layer = nn.LayerNorm()
#         self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
#         self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
#         self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
#         self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

#     def __call__(self, x):
#         x = self.norm_layer(x)
#         x = nn.tanh(self.layer1(x))
#         x = nn.tanh(self.layer2(x))
#         concentration = jnp.exp(self.mu_layer(x))
#         rate = jnp.exp(self.sig_layer(x))
#         return concentration, rate
    


# class IdealPointNN(nn.Module):
#     hidden_size1: int
#     hidden_size2: int
#     output_size: int

#     def setup(self):
#         self.norm_layer = nn.LayerNorm()
#         self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
#         self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
#         self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
#         self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

#     def __call__(self, x):
#         x = self.norm_layer(x)
#         x = nn.tanh(self.layer1(x))
#         x = nn.tanh(self.layer2(x))
#         mu = self.mu_layer(x)
#         sig = jnp.exp(self.sig_layer(x))
#         return mu, sig
        
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

    @nn.compact
    def __call__(self, x):
        x = self.norm_layer(x)
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer(x))
        concentration = jnp.exp(self.mu_layer(x))
        rate = jnp.exp(self.sig_layer(x))
        return concentration, rate

class IdealPointNN(nn.Module):
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

    @nn.compact
    def __call__(self, x):
        x = self.norm_layer(x)
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer(x))
        concentration = self.mu_layer(x)
        rate = jnp.exp(self.sig_layer(x))
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
        sig = jnp.exp(self.sig_layer(x))
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