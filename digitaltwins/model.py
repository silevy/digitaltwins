import jax
import numpyro as npr
import numpyro.distributions as dist
import numpyro.infer.reparam as reparam
from numpyro.contrib.module import flax_module
from numpyro.contrib.module import random_flax_module

from jax import numpy as jnp

from . import main, util


####################
# HELPER FUNCTIONS #
####################

def f_sample_K(K: int,
               Y_u: dict,
               Y_q: jnp.ndarray,
               z: jnp.ndarray,
               cutpoints: dict,
               beta: jnp.ndarray,
               phi_nn: callable,
               N: int,
               J_u: int,
               J_u_dict: dict,
               J_u_idx_start: dict,
               J_u_idx_end: dict,
               L: int,
               T: int,
               scale_term: float):
    
    # priors, cont.
    with npr.handlers.scale(scale=scale_term):
        with npr.plate('J_u', J_u):
            alpha = npr.sample('alpha_' + str(K), dist.Normal().expand([T]).to_event(1))
        
        with npr.plate('L', L, dim=-2), npr.plate('T', T, dim=-1):
            if phi_nn is not None:
                concen_phi, rate_phi = phi_nn(Y_q)
                phi = npr.sample('phi_' + str(K), dist.Gamma(concen_phi.T, rate_phi.T))
            else:
                phi = npr.sample('phi_' + str(K), dist.Gamma(1.0, 1.0))
                

    # broadcasting to dim(N,J,T)
    alpha = jnp.repeat(alpha[None,:,:], repeats=N, axis=0)
    betaphi = (1/L) * jnp.repeat(jnp.expand_dims(jnp.square(beta) @ phi, 0), repeats=N, axis=0)
    z = jnp.repeat(jnp.expand_dims(z, 1), repeats=J_u, axis=1)
    
    assert alpha.shape == betaphi.shape and alpha.shape == z.shape

    # likelihood
    u = alpha + z * betaphi     # dim(N,J,T)
    for h, y in Y_u.items():
        f_sample_Y(h, N, K, T, y, u, cutpoints, J_u_dict, J_u_idx_start, J_u_idx_end)


def f_sample_K_mcmc(K: int,
               Y_u: dict,
               Y_q: jnp.ndarray,
               z: jnp.ndarray,
               cutpoints: dict,
               beta: jnp.ndarray,
               phi_nn: callable,
               N: int,
               J_u: int,
               J_u_dict: dict,
               J_u_idx_start: dict,
               J_u_idx_end: dict,
               L: int,
               T: int,
               scale_term: float):
    
    # priors, cont.
    with npr.plate('J_u', J_u):
        alpha = npr.sample('alpha_' + str(K), dist.Normal().expand([T]).to_event(1))
    
    with npr.plate('L', L, dim=-2), npr.plate('T', T, dim=-1):
        if phi_nn is not None:
            concen_phi, rate_phi = phi_nn(Y_q)
            phi = npr.sample('phi_' + str(K), dist.Gamma(concen_phi.T, rate_phi.T))
        else:
            phi = npr.sample('phi_' + str(K), dist.Gamma(1.0, 1.0))
                
    # broadcasting to dim(N,J,T)
    alpha = jnp.repeat(alpha[None,:,:], repeats=N, axis=0)
    betaphi = (1/L) * jnp.repeat(jnp.expand_dims(jnp.square(beta) @ phi, 0), repeats=N, axis=0)
    z = jnp.repeat(jnp.expand_dims(z, 1), repeats=J_u, axis=1)
    
    assert alpha.shape == betaphi.shape and alpha.shape == z.shape

    # likelihood
    u = alpha + z * betaphi     # dim(N,J,T)
    for h, y in Y_u.items():
        f_sample_Y(h, N, K, T, y, u, cutpoints, J_u_dict, J_u_idx_start, J_u_idx_end)




def f_sample_Y(h: str, 
               N:int, 
               K: int, 
               T: int, 
               Y_u: jnp.ndarray,
               Y_u_star: jnp.ndarray, 
               cutpoints: dict,
               J_dict: dict, 
               idx_start: dict, 
               idx_end: dict):
    
    # heterogeneous cutoffs
    c = cutpoints[h]
    c = jnp.expand_dims(c, [0,2])
    
    with npr.plate('N', N, dim=-3), \
         npr.plate('J_' + h, J_dict[h], dim=-2), \
         npr.plate('T', T, dim=-1):
        
        npr.sample('Y_u_' + str(K) + '_' + h, 
                    dist.OrderedLogistic(predictor=Y_u_star[:,idx_start[h]:idx_end[h],:], 
                                         cutpoints=c),
                    obs=Y_u)
        
##########
# MODELS #
##########

def model_svi(Y_u_1_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_1_10: jnp.ndarray = None,
               Y_u_1_5: jnp.ndarray = None,

               Y_u_2_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_2_10: jnp.ndarray = None,
               Y_u_2_5: jnp.ndarray = None,
               
               Y_u_3_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_3_10: jnp.ndarray = None,
               Y_u_3_5: jnp.ndarray = None,
               
               Y_c_1_static: jnp.ndarray = None,       # STATIC common variables one-hot encoded, dim=(N,J_c_static,T) tensor
               Y_c_2_static: jnp.ndarray = None,       # input shape +2 for (t, k)
               Y_c_3_static: jnp.ndarray = None,
               
               Y_c_1_optim: jnp.ndarray = None,       # OPTIM common variables all continuous, dim=(N,J_c_optim,T) tensor
               Y_c_2_optim: jnp.ndarray = None,       
               Y_c_3_optim: jnp.ndarray = None,
               
               ###################################################
               # input variables below static across minibatches #
               ###################################################

               Y_q_1: jnp.ndarray = None,       # firm-level SEC 10-Q fields, dim=(T,dim(Q)) tensor
               Y_q_2: jnp.ndarray = None,
               Y_q_3: jnp.ndarray = None,
               
               J_c: int = None,                 # dim of common questions
               J_u: int = None,                 # dim uncommon questions, total
               J_u_dict: dict = None,           # dim uncommon questions, by type
               J_u_idx_start: dict = None,      # indices of uncommons, starting
               J_u_idx_end: dict = None,        # indices of uncommons, ending
               L: int = None,                   # dim of (latent) service quality
               Q: int = None,                   # dim of 10-Q fields
               T: int = None,                   # dim of quarters
               hidden_dim: int = None,
               scale_term: float = None):

    assert Y_c_1_static.shape[0] == Y_c_2_static.shape[0] and \
           Y_c_1_static.shape[0] == Y_c_3_static.shape[0]

    N = Y_c_1_static.shape[0]
        
    # concatenaate Y_c
    Y_c = jnp.concatenate(
                [   jnp.concatenate([Y_c_1_static, Y_c_2_static, Y_c_3_static], axis=0),
                    jnp.concatenate([Y_c_1_optim, Y_c_2_optim, Y_c_3_optim], axis=0)    ],
                axis=1
            )
    Y_c = jnp.moveaxis(Y_c, source=2, destination=1)
    

    # amortization NNs
    phi_nn = flax_module("phi_nn", 
                         util.PhiNN(
                                    hidden_size1=hidden_dim, 
                                    hidden_size2=hidden_dim * 2,
                                    output_size=L), 
                         input_shape=(T,Q))
    
    z_nn = flax_module("z_nn", 
                         util.IdealPointNN(
                                           hidden_size1=hidden_dim, 
                                           hidden_size2=hidden_dim * 2,
                                           output_size=1), 
                         input_shape=Y_c.shape)      
    
    # priors
    mu_z, sig_z = z_nn(Y_c) # each output should be (N*3, T)

    #############################


    with npr.plate('N_total', N*3, dim=-2), npr.plate('T', T, dim=-1):
        with npr.handlers.reparam(config={'z': reparam.TransformReparam()}):
            z = npr.sample('z', dist.TransformedDistribution(
                                    dist.Normal(),
                                    dist.transforms.AffineTransform(mu_z.squeeze(), sig_z.squeeze())
                                    ))
    
    with npr.handlers.scale(scale=scale_term):
        with npr.plate('J_u', J_u):
            beta = npr.sample('beta', dist.Normal().expand([L]).to_event(1))
        
        cutpoints = {}
        for h, j in J_u_dict.items():
            with npr.plate('J_h' + h, j):
                with npr.handlers.reparam(config={'c_' + h: reparam.TransformReparam()}):
                    cutpoints[h] = npr.sample('c_' + h, 
                            dist.TransformedDistribution(
                                    dist.Dirichlet(jnp.ones([main.H_CUTOFFS[h]+1])),
                                    dist.transforms.SimplexToOrderedTransform()
                                    ))
                    # cutpoints[h] = npr.sample("c_" + h, 
                    #                           dist.TransformedDistribution(
                    #                                 dist.Normal().expand([main.H_CUTOFFS[h]]).to_event(1),
                    #                                 dist.transforms.OrderedTransform()
                    #                                 ))
    
    # sample across 3 firms
    partialized_f_sample_K = jax.tree_util.Partial(f_sample_K,
                                cutpoints=cutpoints,
                                beta=beta,
                                phi_nn=phi_nn,
                                N=N,
                                J_u=J_u,
                                J_u_dict=J_u_dict,
                                J_u_idx_start=J_u_idx_start,
                                J_u_idx_end=J_u_idx_end,
                                L=L,
                                T=T,
                                scale_term=scale_term)
    
    # coalesce Y
    Y_1 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_1_11, Y_u_1_10, Y_u_1_5) ) }
    Y_2 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_2_11, Y_u_2_10, Y_u_2_5) ) }
    Y_3 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_3_11, Y_u_3_10, Y_u_3_5) ) }

    # split ideal-points across firms
    z_1, z_2, z_3 = jnp.split(z, 3, axis=0)

    # run sample statements
    partialized_f_sample_K(1, Y_1, Y_q_1, z_1)
    partialized_f_sample_K(2, Y_2, Y_q_2, z_2)
    partialized_f_sample_K(3, Y_3, Y_q_3, z_3)




def model_mcmc(Y_u_1_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_1_10: jnp.ndarray = None,
               Y_u_1_5: jnp.ndarray = None,

               Y_u_2_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_2_10: jnp.ndarray = None,
               Y_u_2_5: jnp.ndarray = None,
               
               Y_u_3_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_3_10: jnp.ndarray = None,
               Y_u_3_5: jnp.ndarray = None,
               
               Y_c_1_static: jnp.ndarray = None,       # STATIC common variables one-hot encoded, dim=(N,J_c_static,T) tensor
               Y_c_2_static: jnp.ndarray = None,       # input shape +2 for (t, k)
               Y_c_3_static: jnp.ndarray = None,
               
               Y_c_1_optim: jnp.ndarray = None,       # OPTIM common variables all continuous, dim=(N,J_c_optim,T) tensor
               Y_c_2_optim: jnp.ndarray = None,       
               Y_c_3_optim: jnp.ndarray = None,
               
               ###################################################
               # input variables below static across minibatches #
               ###################################################

               Y_q_1: jnp.ndarray = None,       # firm-level SEC 10-Q fields, dim=(T,dim(Q)) tensor
               Y_q_2: jnp.ndarray = None,
               Y_q_3: jnp.ndarray = None,
               
               J_c: int = None,                 # dim of common questions
               J_u: int = None,                 # dim uncommon questions, total
               J_u_dict: dict = None,           # dim uncommon questions, by type
               J_u_idx_start: dict = None,      # indices of uncommons, starting
               J_u_idx_end: dict = None,        # indices of uncommons, ending
               L: int = None,                   # dim of (latent) service quality
               Q: int = None,                   # dim of 10-Q fields
               T: int = None,                   # dim of quarters
               hidden_dim: int = None,
               scale_term: float = None):

    assert Y_c_1_static.shape[0] == Y_c_2_static.shape[0] and \
           Y_c_1_static.shape[0] == Y_c_3_static.shape[0]

    N = Y_c_1_static.shape[0]
        
    # concatenaate Y_c
    Y_c = jnp.concatenate(
                [   jnp.concatenate([Y_c_1_static, Y_c_2_static, Y_c_3_static], axis=0),
                    jnp.concatenate([Y_c_1_optim, Y_c_2_optim, Y_c_3_optim], axis=0)    ],
                axis=1
            )
    Y_c = jnp.moveaxis(Y_c, source=2, destination=1)
    
    ##############################
    # Bayesian NN:

    # Define priors for PhiNN
    # phi_nn_prior = {
    #     "layer.kernel": dist.Normal(0, 1),  # Weights
    #     "layer.bias": dist.Normal(0, 1),   # Bias
    #     "mu_layer.kernel": dist.Normal(0, 1),
    #     "mu_layer.bias": dist.Normal(0, 1),
    #     "sig_layer.kernel": dist.Normal(0, 1),
    #     "sig_layer.bias": dist.Normal(0, 1),
    # }

    # # Define priors for IdealPointNN
    # z_nn_prior = {
    #     "layer.kernel": dist.Normal(0, 1),
    #     "layer.bias": dist.Normal(0, 1),
    #     "mu_layer.kernel": dist.Normal(0, 1),
    #     "mu_layer.bias": dist.Normal(0, 1),
    #     "sig_layer.kernel": dist.Normal(0, 1),
    #     "sig_layer.bias": dist.Normal(0, 1),
    # }







    # Define priors for PhiNN
    phi_nn_prior = dist.Normal(0, 0.1)  # Weights

    # # Define priors for IdealPointNN
    z_nn_prior = dist.Normal(0, 0.1)

    # Sample the Bayesian neural networks
    phi_nn = random_flax_module(
        "phi_nn",
        util.PhiNN(hidden_size1=hidden_dim, hidden_size2=hidden_dim * 2, output_size=L),
        prior=phi_nn_prior,
        input_shape=(T, Q),
        # input_shape=(),
    )

    z_nn = random_flax_module(
        "z_nn",
        util.IdealPointNN(hidden_size1=hidden_dim, hidden_size2=hidden_dim * 2, output_size=1),
        prior=z_nn_prior,
        input_shape=Y_c.shape,
        # input_shape=jnp.expand_dims(Y_c, axis=0).shape
    )

    # Define priors for PhiNN
    # phi_nn_prior = dist.Normal(0, 0.1)  # Weights

    # # Define priors for IdealPointNN
    # z_nn_prior = dist.Normal(0, 0.1)

    # # Pure JAX definitions for PhiNN and IdealPointNN
    # phi_nn_params = util.init_ideal_point_nn(
    #     input_dim=Q,  # Assuming Q is the number of input features
    #     hidden_dim1=hidden_dim,
    #     hidden_dim2=hidden_dim * 2,
    #     output_dim=L,
    #     key=jax.random.PRNGKey(234),
    # )

    # z_nn_params = util.init_ideal_point_nn(
    #     input_dim=Y_c.shape[-1],  # Assuming last dim of Y_c is the input feature size
    #     hidden_dim1=hidden_dim,
    #     hidden_dim2=hidden_dim * 2,
    #     output_dim=1,
    #     key=jax.random.PRNGKey(567),
    # )

    # # Forward pass through Bayesian NNs
    # def phi_nn_forward(x):
    #     return forward_ideal_point_nn(x, phi_nn_params)

    # def z_nn_forward(x):
    #     return forward_ideal_point_nn(x, z_nn_params)

    # # Sample Bayesian neural networks
    # phi_nn = random_flax_module(
    #     "phi_nn",
    #     util.PhiNNFlax(params=phi_nn_params), 
    #     prior=phi_nn_prior,
    #     input_shape=(T, Q),
    # )

    # z_nn = random_flax_module(
    #     "z_nn",
    #     util.IdealPointNNFlax(params=z_nn_params),
    #     prior=z_nn_prior,
    #     input_shape=Y_c.shape,
    # )


    ##############################

    
    # Forward pass through the Bayesian NNs
    mu_z, sig_z = z_nn(Y_c)  # For IdealPointNN


    with npr.plate('N_total', N*3, dim=-2), npr.plate('T', T, dim=-1):
        with npr.handlers.reparam(config={'z': reparam.TransformReparam()}):
            z = npr.sample('z', dist.TransformedDistribution(
                                    dist.Normal(),
                                    dist.transforms.AffineTransform(mu_z.squeeze(), sig_z.squeeze())
                                    ))
    
    with npr.plate('J_u', J_u):
        beta = npr.sample('beta', dist.Normal().expand([L]).to_event(1))
    
    cutpoints = {}
    for h, j in J_u_dict.items():
        with npr.plate('J_h' + h, j):
            with npr.handlers.reparam(config={'c_' + h: reparam.TransformReparam()}):
                cutpoints[h] = npr.sample('c_' + h, 
                        dist.TransformedDistribution(
                                dist.Dirichlet(jnp.ones([main.H_CUTOFFS[h]+1])),
                                dist.transforms.SimplexToOrderedTransform()
                                ))
                # cutpoints[h] = npr.sample("c_" + h, 
                #                           dist.TransformedDistribution(
                #                                 dist.Normal().expand([main.H_CUTOFFS[h]]).to_event(1),
                #                                 dist.transforms.OrderedTransform()
                #                                 ))
    
    # sample across 3 firms
    partialized_f_sample_K = jax.tree_util.Partial(f_sample_K_mcmc,
                                cutpoints=cutpoints,
                                beta=beta,
                                phi_nn=phi_nn,
                                N=N,
                                J_u=J_u,
                                J_u_dict=J_u_dict,
                                J_u_idx_start=J_u_idx_start,
                                J_u_idx_end=J_u_idx_end,
                                L=L,
                                T=T,
                                scale_term=scale_term)
    
    # coalesce Y
    Y_1 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_1_11, Y_u_1_10, Y_u_1_5) ) }
    Y_2 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_2_11, Y_u_2_10, Y_u_2_5) ) }
    Y_3 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_3_11, Y_u_3_10, Y_u_3_5) ) }

    # split ideal-points across firms
    z_1, z_2, z_3 = jnp.split(z, 3, axis=0)

    # run sample statements
    partialized_f_sample_K(1, Y_1, Y_q_1, z_1)
    partialized_f_sample_K(2, Y_2, Y_q_2, z_2)
    partialized_f_sample_K(3, Y_3, Y_q_3, z_3)


# The following assume SVI as estimation method

def model_noPredictors(
               Y_u_1_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_1_10: jnp.ndarray = None,
               Y_u_1_5: jnp.ndarray = None,

               Y_u_2_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_2_10: jnp.ndarray = None,
               Y_u_2_5: jnp.ndarray = None,
               
               Y_u_3_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_3_10: jnp.ndarray = None,
               Y_u_3_5: jnp.ndarray = None,
               
               Y_c_1_static: jnp.ndarray = None,       # STATIC common variables one-hot encoded, dim=(N,J_c_static,T) tensor
               Y_c_2_static: jnp.ndarray = None,       # input shape +2 for (t, k)
               Y_c_3_static: jnp.ndarray = None,
               
               Y_c_1_optim: jnp.ndarray = None,       # OPTIM common variables all continuous, dim=(N,J_c_optim,T) tensor
               Y_c_2_optim: jnp.ndarray = None,       
               Y_c_3_optim: jnp.ndarray = None,
               
               ###################################################
               # input variables below static across minibatches #
               ###################################################

               Y_q_1: jnp.ndarray = None,       # firm-level SEC 10-Q fields, dim=(T,dim(Q)) tensor
               Y_q_2: jnp.ndarray = None,
               Y_q_3: jnp.ndarray = None,
               
               J_c: int = None,                 # dim of common questions
               J_u: int = None,                 # dim uncommon questions, total
               J_u_dict: dict = None,           # dim uncommon questions, by type
               J_u_idx_start: dict = None,      # indices of uncommons, starting
               J_u_idx_end: dict = None,        # indices of uncommons, ending
               L: int = None,                   # dim of (latent) service quality
               Q: int = None,                   # dim of 10-Q fields
               T: int = None,                   # dim of quarters
               hidden_dim: int = None,
               scale_term: float = None):

    assert Y_c_1_static.shape[0] == Y_c_2_static.shape[0] and \
           Y_c_1_static.shape[0] == Y_c_3_static.shape[0]

    N = Y_c_1_static.shape[0]
        
    # concatenaate Y_c
    Y_c = jnp.concatenate(
                [   jnp.concatenate([Y_c_1_static, Y_c_2_static, Y_c_3_static], axis=0),
                    jnp.concatenate([Y_c_1_optim, Y_c_2_optim, Y_c_3_optim], axis=0)    ],
                axis=1
            )
    Y_c = jnp.moveaxis(Y_c, source=2, destination=1)
    

    # amortization NNs
    phi_nn = flax_module("phi_nn", 
                         util.PhiNN(hidden_size1=hidden_dim, 
                                    hidden_size2=hidden_dim * 2,
                                    output_size=L), 
                         input_shape=(T,Q))
    

    # priors
    with npr.plate('N_total', N*3, dim=-2), npr.plate('T', T, dim=-1):
        z = npr.sample('z', dist.Normal())
    
    with npr.handlers.scale(scale=scale_term):
        with npr.plate('J_u', J_u):
            beta = npr.sample('beta', dist.Normal().expand([L]).to_event(1))
        
        cutpoints = {}
        for h, j in J_u_dict.items():
            with npr.plate('J_h' + h, j):
                with npr.handlers.reparam(config={'c_' + h: reparam.TransformReparam()}):
                    cutpoints[h] = npr.sample('c_' + h, 
                            dist.TransformedDistribution(
                                    dist.Dirichlet(jnp.ones([main.H_CUTOFFS[h]+1])),
                                    dist.transforms.SimplexToOrderedTransform()
                                    ))
                    
    # sample across 3 firms
    partialized_f_sample_K = jax.tree_util.Partial(f_sample_K,
                                cutpoints=cutpoints,
                                beta=beta,
                                phi_nn=phi_nn,
                                N=N,
                                J_u=J_u,
                                J_u_dict=J_u_dict,
                                J_u_idx_start=J_u_idx_start,
                                J_u_idx_end=J_u_idx_end,
                                L=L,
                                T=T,
                                scale_term=scale_term)
    
    # coalesce Y
    Y_1 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_1_11, Y_u_1_10, Y_u_1_5) ) }
    Y_2 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_2_11, Y_u_2_10, Y_u_2_5) ) }
    Y_3 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_3_11, Y_u_3_10, Y_u_3_5) ) }

    # split ideal-points across firms
    z_1, z_2, z_3 = jnp.split(z, 3, axis=0)

    # run sample statements
    partialized_f_sample_K(1, Y_1, Y_q_1, z_1)
    partialized_f_sample_K(2, Y_2, Y_q_2, z_2)
    partialized_f_sample_K(3, Y_3, Y_q_3, z_3)
    


def model_noKPI(Y_u_1_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_1_10: jnp.ndarray = None,
               Y_u_1_5: jnp.ndarray = None,

               Y_u_2_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_2_10: jnp.ndarray = None,
               Y_u_2_5: jnp.ndarray = None,
               
               Y_u_3_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_3_10: jnp.ndarray = None,
               Y_u_3_5: jnp.ndarray = None,
               
               Y_c_1_static: jnp.ndarray = None,       # STATIC common variables one-hot encoded, dim=(N,J_c_static,T) tensor
               Y_c_2_static: jnp.ndarray = None,       # input shape +2 for (t, k)
               Y_c_3_static: jnp.ndarray = None,
               
               Y_c_1_optim: jnp.ndarray = None,       # OPTIM common variables all continuous, dim=(N,J_c_optim,T) tensor
               Y_c_2_optim: jnp.ndarray = None,       
               Y_c_3_optim: jnp.ndarray = None,
               
               ###################################################
               # input variables below static across minibatches #
               ###################################################

               Y_q_1: jnp.ndarray = None,       # firm-level SEC 10-Q fields, dim=(T,dim(Q)) tensor
               Y_q_2: jnp.ndarray = None,
               Y_q_3: jnp.ndarray = None,
               
               J_c: int = None,                 # dim of common questions
               J_u: int = None,                 # dim uncommon questions, total
               J_u_dict: dict = None,           # dim uncommon questions, by type
               J_u_idx_start: dict = None,      # indices of uncommons, starting
               J_u_idx_end: dict = None,        # indices of uncommons, ending
               L: int = None,                   # dim of (latent) service quality
               Q: int = None,                   # dim of 10-Q fields
               T: int = None,                   # dim of quarters
               hidden_dim: int = None,
               scale_term: float = None):

    assert Y_c_1_static.shape[0] == Y_c_2_static.shape[0] and \
           Y_c_1_static.shape[0] == Y_c_3_static.shape[0]

    N = Y_c_1_static.shape[0]
        
    # concatenaate Y_c
    Y_c = jnp.concatenate(
                [   jnp.concatenate([Y_c_1_static, Y_c_2_static, Y_c_3_static], axis=0),
                    jnp.concatenate([Y_c_1_optim, Y_c_2_optim, Y_c_3_optim], axis=0)    ],
                axis=1
            )
    Y_c = jnp.moveaxis(Y_c, source=2, destination=1)
    

    # amortization NNs    
    z_nn = flax_module("z_nn", 
                         util.IdealPointNN(hidden_size1=hidden_dim, 
                                           hidden_size2=hidden_dim * 2,
                                           output_size=1), 
                         input_shape=Y_c.shape)      
    

    # priors
    mu_z, sig_z = z_nn(Y_c) # each output should be (N*3, T)
    with npr.plate('N_total', N*3, dim=-2), npr.plate('T', T, dim=-1):
        with npr.handlers.reparam(config={'z': reparam.TransformReparam()}):
            z = npr.sample('z', dist.TransformedDistribution(
                                    dist.Normal(),
                                    dist.transforms.AffineTransform(mu_z.squeeze(), sig_z.squeeze())
                                    ))
    
    with npr.handlers.scale(scale=scale_term):
        with npr.plate('J_u', J_u):
            beta = npr.sample('beta', dist.Normal().expand([L]).to_event(1))
        
        cutpoints = {}
        for h, j in J_u_dict.items():
            with npr.plate('J_h' + h, j):
                with npr.handlers.reparam(config={'c_' + h: reparam.TransformReparam()}):
                    cutpoints[h] = npr.sample('c_' + h, 
                            dist.TransformedDistribution(
                                    dist.Dirichlet(jnp.ones([main.H_CUTOFFS[h]+1])),
                                    dist.transforms.SimplexToOrderedTransform()
                                    ))
    
    # sample across 3 firms
    partialized_f_sample_K = jax.tree_util.Partial(f_sample_K,
                                cutpoints=cutpoints,
                                beta=beta,
                                phi_nn=None,
                                N=N,
                                J_u=J_u,
                                J_u_dict=J_u_dict,
                                J_u_idx_start=J_u_idx_start,
                                J_u_idx_end=J_u_idx_end,
                                L=L,
                                T=T,
                                scale_term=scale_term)
    
    # coalesce Y
    Y_1 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_1_11, Y_u_1_10, Y_u_1_5) ) }
    Y_2 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_2_11, Y_u_2_10, Y_u_2_5) ) }
    Y_3 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_3_11, Y_u_3_10, Y_u_3_5) ) }

    # split ideal-points across firms
    z_1, z_2, z_3 = jnp.split(z, 3, axis=0)

    # run sample statements
    partialized_f_sample_K(1, Y_1, Y_q_1, z_1)
    partialized_f_sample_K(2, Y_2, Y_q_2, z_2)
    partialized_f_sample_K(3, Y_3, Y_q_3, z_3)
    


def model_linear_homo(
               Y_u_1_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_1_10: jnp.ndarray = None,
               Y_u_1_5: jnp.ndarray = None,

               Y_u_2_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_2_10: jnp.ndarray = None,
               Y_u_2_5: jnp.ndarray = None,
               
               Y_u_3_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_3_10: jnp.ndarray = None,
               Y_u_3_5: jnp.ndarray = None,
               
               Y_c_1_static: jnp.ndarray = None,       # STATIC common variables one-hot encoded, dim=(N,J_c_static,T) tensor
               Y_c_2_static: jnp.ndarray = None,       # input shape +2 for (t, k)
               Y_c_3_static: jnp.ndarray = None,
               
               Y_c_1_optim: jnp.ndarray = None,       # OPTIM common variables all continuous, dim=(N,J_c_optim,T) tensor
               Y_c_2_optim: jnp.ndarray = None,       
               Y_c_3_optim: jnp.ndarray = None,
               
               ###################################################
               # input variables below static across minibatches #
               ###################################################

               Y_q_1: jnp.ndarray = None,       # firm-level SEC 10-Q fields, dim=(T,dim(Q)) tensor
               Y_q_2: jnp.ndarray = None,
               Y_q_3: jnp.ndarray = None,
               
               J_c: int = None,                 # dim of common questions
               J_u: int = None,                 # dim uncommon questions, total
               J_u_dict: dict = None,           # dim uncommon questions, by type
               J_u_idx_start: dict = None,      # indices of uncommons, starting
               J_u_idx_end: dict = None,        # indices of uncommons, ending
               L: int = None,                   # dim of (latent) service quality
               Q: int = None,                   # dim of 10-Q fields
               T: int = None,                   # dim of quarters
               hidden_dim: int = None,
               scale_term: float = None):

    assert Y_c_1_static.shape[0] == Y_c_2_static.shape[0] and \
           Y_c_1_static.shape[0] == Y_c_3_static.shape[0]

    N = Y_c_1_static.shape[0]
        
    # concatenaate Y_c
    Y_c = jnp.concatenate(
                [   jnp.concatenate([Y_c_1_static, Y_c_2_static, Y_c_3_static], axis=0),
                    jnp.concatenate([Y_c_1_optim, Y_c_2_optim, Y_c_3_optim], axis=0)    ],
                axis=1
            )
    Y_c = jnp.moveaxis(Y_c, source=2, destination=1)
    

    # amortization NNs
    phi_nn = flax_module("phi_nn", 
                         util.PhiLinear(output_size=L), 
                         input_shape=(T,Q))
    
    z_nn = flax_module("z_nn", 
                       util.IdealPointLinear(output_size=1), 
                       input_shape=Y_c.shape)      
    

    # priors
    mu_z, sig_z = z_nn(Y_c) # each output should be (N*3, T)
    with npr.plate('N_total', N*3, dim=-2), npr.plate('T', T, dim=-1):
        with npr.handlers.reparam(config={'z': reparam.TransformReparam()}):
            z = npr.sample('z', dist.TransformedDistribution(
                                    dist.Normal(),
                                    dist.transforms.AffineTransform(mu_z.squeeze(), sig_z.squeeze())
                                    ))
    
    with npr.handlers.scale(scale=scale_term):
        with npr.plate('J_u', J_u):
            beta = npr.sample('beta', dist.Normal().expand([L]).to_event(1))
        
        cutpoints = {}
        for h, j in J_u_dict.items():
            with npr.plate('J_h' + h, j):
                with npr.handlers.reparam(config={'c_' + h: reparam.TransformReparam()}):
                    cutpoints[h] = npr.sample('c_' + h, 
                            dist.TransformedDistribution(
                                    dist.Dirichlet(jnp.ones([main.H_CUTOFFS[h]+1])),
                                    dist.transforms.SimplexToOrderedTransform()
                                    ))
    
    # sample across 3 firms
    partialized_f_sample_K = jax.tree_util.Partial(f_sample_K,
                                cutpoints=cutpoints,
                                beta=beta,
                                phi_nn=phi_nn,
                                N=N,
                                J_u=J_u,
                                J_u_dict=J_u_dict,
                                J_u_idx_start=J_u_idx_start,
                                J_u_idx_end=J_u_idx_end,
                                L=L,
                                T=T,
                                scale_term=scale_term)
    
    # coalesce Y
    Y_1 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_1_11, Y_u_1_10, Y_u_1_5) ) }
    Y_2 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_2_11, Y_u_2_10, Y_u_2_5) ) }
    Y_3 = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_u_3_11, Y_u_3_10, Y_u_3_5) ) }

    # split ideal-points across firms
    z_1, z_2, z_3 = jnp.split(z, 3, axis=0)

    # run sample statements
    partialized_f_sample_K(1, Y_1, Y_q_1, z_1)
    partialized_f_sample_K(2, Y_2, Y_q_2, z_2)
    partialized_f_sample_K(3, Y_3, Y_q_3, z_3)




        


##############
# UNIT TESTS #
##############

def unit_test_model(Y_11: jnp.ndarray,      # dim=(N,J_h,T) tensors
                    Y_10: jnp.ndarray,
                    Y_5: jnp.ndarray,
                    Y_2: jnp.ndarray,
                    c_a: jnp.ndarray,       # Rossi cutoff coefficients
                    c_b: jnp.ndarray,
                    N: int,                 # num of respondents
                    L: int,                 # latent dimensions, service quality
                    T: int,                 # num of quarters
                    J_total: int,
                    J_dict: dict,           # 
                    J_idx_end: dict,
                    J_idx_start: dict):                
    
    # coalesce Y
    Y = { k:v for (k, v) in zip( main.H_CUTOFFS.keys(), (Y_11, Y_10, Y_5, Y_2) ) }
    H = len(main.H_CUTOFFS)
    

    # priors
    with npr.plate('J', J_total, dim=-2), npr.plate('T', T):
        alpha = npr.sample('alpha', dist.Normal())

    with npr.plate('J', J_total, dim=-2), npr.plate('L', L):
        beta = npr.sample('beta', dist.HalfNormal())
    
    with npr.plate('L', L, dim=-2), npr.plate('T', T):
        phi = npr.sample('phi', dist.Normal())

    with npr.plate('N', N,dim=-2), npr.plate('T', T):
        x = npr.sample('x', dist.Normal())
    
    with npr.plate('H', H):
        c_b = npr.sample('c_b', dist.HalfNormal())
        c_e = npr.sample('c_e', dist.HalfNormal())
    

    # broadcasting to dim(N,J,T)
    alpha = jnp.repeat(jnp.expand_dims(alpha, 0), 
                       repeats=N, axis=0)
    betaphi = jnp.repeat(jnp.expand_dims(jnp.matmul(beta, phi), 0), 
                         repeats=N, axis=0)
    x = jnp.repeat(jnp.expand_dims(x, 1),
                    repeats=J_total, axis=1)
    
    assert alpha.shape == betaphi.shape and alpha.shape == x.shape


    # heterogeneous cutoffs
    c_global = { c_label : c_a[i] + c_b[i]*jnp.arange(c_val) + c_e[i]*jnp.square(jnp.arange(c_val)) \
                 for i, (c_label, c_val) in enumerate(main.H_CUTOFFS.items()) }
    
    sigma = npr.param('sigma', init_value=jnp.ones([N, 1]), constraint=dist.constraints.positive)
    tau = npr.param('tau', init_value=jnp.zeros([N, 1]))
    
    c = { c_label : tau + sigma * c_val \
            for (c_label, c_val) in c_global.items() }


    # likelihood
    with npr.plate('N', N, dim=-3):
        u = alpha + x * betaphi     # dim(N,J,T)
        for h, y in Y.items():
            f_sample_Y(h, y, u, c, J_dict, T, J_idx_start, J_idx_end)



#####################
# VARIATIONAL GUIDE #
#####################

def guide_full(Y_u_1_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_1_10: jnp.ndarray = None,
               Y_u_1_5: jnp.ndarray = None,

               Y_u_2_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_2_10: jnp.ndarray = None,
               Y_u_2_5: jnp.ndarray = None,
               
               Y_u_3_11: jnp.ndarray = None,   # uncommon variables, in scale points, dim=(N,J,T) tensor
               Y_u_3_10: jnp.ndarray = None,
               Y_u_3_5: jnp.ndarray = None,
               
               Y_c_1_static: jnp.ndarray = None,       # STATIC common variables one-hot encoded, dim=(N,J_c_static,T) tensor
               Y_c_2_static: jnp.ndarray = None,       # input shape +2 for (t, k)
               Y_c_3_static: jnp.ndarray = None,
               
               Y_c_1_optim: jnp.ndarray = None,       # OPTIM common variables all continuous, dim=(N,J_c_optim,T) tensor
               Y_c_2_optim: jnp.ndarray = None,       
               Y_c_3_optim: jnp.ndarray = None,
               
               ###################################################
               # input variables below static across minibatches #
               ###################################################

               Y_q_1: jnp.ndarray = None,       # firm-level SEC 10-Q fields, dim=(T,dim(Q)) tensor
               Y_q_2: jnp.ndarray = None,
               Y_q_3: jnp.ndarray = None,
               
               J_c: int = None,                 # dim of common questions
               J_u: int = None,                 # dim uncommon questions, total
               J_u_dict: dict = None,           # dim uncommon questions, by type
               J_u_idx_start: dict = None,      # indices of uncommons, starting
               J_u_idx_end: dict = None,        # indices of uncommons, ending
               L: int = None,                   # dim of (latent) service quality
               Q: int = None,                   # dim of 10-Q fields
               T: int = None,                   # dim of quarters
               hidden_dim: int = None,
               scale_term: float = None,
               is_predictive: bool = False):
    
    # under predictive distribution, Y_u_* are missing
    if is_predictive:
        N = Y_c_1_static.shape[0]
    else:
        assert Y_u_1_11.shape[0] == Y_u_2_11.shape[0] and \
               Y_u_1_11.shape[0] == Y_u_3_11.shape[0]
        
        N = Y_u_1_11.shape[0]
    
    
    # z, consumer brand affinity
    mu_z = npr.param('mu_z', init_value=jnp.zeros([N*3, T]), constraint=dist.constraints.real)
    sig_z = npr.param('sig_z', init_value=jnp.ones([N*3, T]), constraint=dist.constraints.positive)
    with npr.plate('N_total', N*3, dim=-2), npr.plate('T', T, dim=-1):
        npr.sample('z_base', dist.TransformedDistribution(
                                    dist.Normal(),
                                    dist.transforms.AffineTransform(mu_z, sig_z)
                                    ))
    
    
    # beta, question weights
    mu_beta = npr.param('mu_beta', init_value=jnp.zeros([J_u, L]), constraint=dist.constraints.real)
    sig_beta = npr.param('sig_beta', init_value=jnp.ones([J_u, L]), constraint=dist.constraints.positive)
    with npr.plate('J_u', J_u):
        npr.sample('beta', dist.TransformedDistribution(
                                    dist.Normal().expand([L]).to_event(1),
                                    dist.transforms.AffineTransform(mu_beta, sig_beta)
                                    ))
    
    
    # ordinal cutoffs
    for h, j in J_u_dict.items():
        mu_c = npr.param('mu_c_' + h, init_value=jnp.zeros([main.H_CUTOFFS[h]]), constraint=dist.constraints.real)
        sig_c = npr.param('sig_c_' + h, init_value=jnp.ones([main.H_CUTOFFS[h]]), constraint=dist.constraints.positive)
        with npr.plate('J_h' + h, j):
            npr.sample('c_' + h + '_base', 
                       dist.TransformedDistribution(
                                dist.Normal().expand([main.H_CUTOFFS[h]+1]).to_event(1),
                                dist.transforms.AffineTransform(mu_c, sig_c)
                                ))
    
    
    # other parameters
    for K in ('1','2','3'):
        # alpha, firm-question fixed effects
        mu_alpha = npr.param('mu_alpha_' + str(K), init_value=jnp.zeros([J_u]), constraint=dist.constraints.real)
        sig_alpha = npr.param('sig_alpha_' + str(K), init_value=jnp.ones([J_u]), constraint=dist.constraints.positive)
        with npr.plate('J_u', J_u):
            npr.sample('alpha_' + str(K), 
                       dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(mu_alpha, sig_alpha)
                                ))
            
        # phi, question loadings w.r.t. performance KPIs
        concen_phi = npr.param('concen_phi_' + str(K), init_value=jnp.ones([L, T]), constraint=dist.constraints.positive)
        rate_phi = npr.param('rate_phi_' + str(K), init_value=jnp.ones([L, T]), constraint=dist.constraints.positive)
        with npr.plate('L', L, dim=-2), npr.plate('T', T, dim=-1):
            npr.sample('phi_' + str(K), dist.Gamma(concen_phi, rate_phi))
