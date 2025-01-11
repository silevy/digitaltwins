import os
import inspect
import pandas as pd
import jax
import argparse
import numpy as np
import pickle 

import jax.numpy as jnp
from flax import linen as nn
import numpyro as npr
import numpyro.distributions as dist
from numpyro import plate
from numpyro.infer import Predictive
from numpyro.distributions import transforms
from numpyro.infer.reparam import TransformReparam

from jax import random

############################
## make results directory ##
############################

HOME_PATH = os.path.dirname(inspect.getfile(lambda: None))
DATA_DIR = os.path.join(HOME_PATH, 'data')
RESULTS_DIR = os.path.join('results')
os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='parse args')
# parser.add_argument('--is-predictive', default=False, type=bool)
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--train-test', default=512, type=int)
parser.add_argument('--latent-dims', default=50, type=int)
parser.add_argument('--hidden-dims', default=512, type=int)
args = parser.parse_args()

###################
## Simulate data ##
###################

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
    hidden_size3: int
    output_size: int
    
    def setup(self):
        self.norm_layer = nn.LayerNorm()
        # self.layer1 = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        # self.layer2 = nn.Dense(self.hidden_size2, kernel_init=nn.initializers.lecun_normal())
        # self.layer3 = nn.Dense(self.hidden_size3, kernel_init=nn.initializers.lecun_normal())
        self.layer = nn.Dense(self.hidden_size1, kernel_init=nn.initializers.lecun_normal())
        self.mu_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())
        self.sig_layer = nn.Dense(self.output_size, kernel_init=nn.initializers.lecun_normal())

    @nn.compact
    def __call__(self, x):
        x = self.norm_layer(x)
        # x = nn.tanh(self.layer1(x))
        # x = nn.tanh(self.layer2(x))
        # x = nn.tanh(self.layer3(x))
        x = nn.tanh(self.layer(x))
        concentration = self.mu_layer(x)
        rate = jnp.exp(self.sig_layer(x))
        return concentration, rate
    
# Function to generate synthetic data
def generate_synthetic_data(key, Y_c, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=1, L=50, Q=5):
    assert Y_c.ndim == 3, f"Input Y_c should have 3 dimensions, but got {Y_c.ndim}"
    N, T, J = Y_c.shape

    # Initialize the PhiNN model
    z_nn = IdealPointNN(hidden_size1=hidden_size1, hidden_size2=hidden_size2, hidden_size3=hidden_size3, output_size=1)
    phi_nn = PhiNN(hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=L)
    
    dummy_input_z = jnp.ones((N, T, J))
    dummy_input_phi = jnp.ones((T, Q))
    
    key, subkey1, subkey2 = random.split(key, 3)
    model_params_z = z_nn.init(subkey1, dummy_input_z)
    model_params_phi = phi_nn.init(subkey2, dummy_input_phi)

    # Get the parameters of the latents from a NN
    mu_z, sig_z = z_nn.apply(model_params_z, Y_c)

    return mu_z, sig_z, phi_nn, model_params_phi


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
                                         cutpoints=c))

# partially sample the latent 
def f_sample_K(K: int,
               Y_u: dict,
               Y_q: jnp.ndarray,
               z: jnp.ndarray,
               cutpoints: dict,
               beta: jnp.ndarray,
               phi_nn: callable,
               model_params_phi: dict,
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
            # phi = npr.sample('phi_' + str(K), dist.Gamma(1.0, 1.0))

            if phi_nn is not None:
                concen_phi, rate_phi = phi_nn.apply(model_params_phi, Y_q)
                phi = npr.sample('phi_' + str(K), dist.Gamma(concen_phi.T, rate_phi.T))
            
            else:
                phi = npr.sample('phi_' + str(K), dist.Gamma(1.0, 1.0))
                
    # broadcasting to dim(N,J,T)
    alpha = jnp.repeat(alpha[None,:,:], repeats=N, axis=0)
    # betaphi = jnp.repeat(jnp.expand_dims(jnp.square(beta) @ phi, 0), repeats=N, axis=0)
    betaphi = (1 / L) * jnp.repeat(jnp.expand_dims(beta @ phi, 0), repeats=N, axis=0)
    z = jnp.repeat(jnp.expand_dims(z, 1), repeats=J_u, axis=1)
    
    # assert alpha.shape == betaphi.shape and alpha.shape == z.shape
    assert alpha.shape == z.shape

    # likelihood
    u = alpha + z * betaphi     # dim(N,J,T)
    npr.deterministic('u_' + str(K), u)

    for h, y in Y_u.items():
        f_sample_Y(h, N, K, T, y, u, cutpoints, J_u_dict, J_u_idx_start, J_u_idx_end)


    # # >>>>>>>>>>> REGULARIZE alpha & beta <<<<<<<<<<<
    # with npr.handlers.scale(scale=scale_term):
    #     with npr.plate('J_u', J_u):
    #         # Instead of dist.Normal(0, 1), try smaller std dev:
    #         alpha = npr.sample('alpha_' + str(K), dist.Normal(0.0, 0.1).expand([T]).to_event(1))
        
    #     with npr.plate('L', L, dim=-2), npr.plate('T', T, dim=-1):
    #         if phi_nn is not None:
    #             concen_phi, rate_phi = phi_nn.apply(model_params_phi, Y_q)
    #             phi = npr.sample('phi_' + str(K), dist.Gamma(concen_phi.T, rate_phi.T))
    #         else:
    #             # narrower Gamma prior can help too:
    #             phi = npr.sample('phi_' + str(K), dist.Gamma(2.0, 2.0))

    # # >>>>>>>>>> ADD A SCALING PARAMETER <<<<<<<<<<<
    # # One global scale factor for the linear predictor
    # predictor_scale = npr.sample(f"predictor_scale_{K}", dist.HalfNormal(1.0))

    # # Broadcasting to dim(N,J,T)
    # alpha = jnp.repeat(alpha[None, :, :], repeats=N, axis=0)
    # betaphi = jnp.repeat(jnp.expand_dims(jnp.square(beta) @ phi, 0), repeats=N, axis=0)
    # z = jnp.repeat(jnp.expand_dims(z, 1), repeats=J_u, axis=1)

    # # SCALING the predictor
    # u = predictor_scale * (alpha + z * betaphi)  # <-- scaled predictor

    # for h, y in Y_u.items():
    #     f_sample_Y(h, N, K, T, y, u, cutpoints, J_u_dict, J_u_idx_start, J_u_idx_end)


# Probabilistic model function, equations 3.8 - 3.10 in the paper
def probabilistic_model(Y_c, mu_z, sig_z, Y_1_dummy, Y_2_dummy, Y_3_dummy, 
                        Y_q_1, Y_q_2, Y_q_3,
                        J_u, J_u_dict, phi_nn, model_params_phi,
                        N, J_u_idx_start, J_u_idx_end, 
                        L, T, scale_term, H_CUTOFFS):

    # Specify the priors for z
    with npr.plate('N_total', Y_c.shape[0], dim=-2), npr.plate('T', Y_c.shape[1], dim=-1):
        with npr.handlers.reparam(config={'z': TransformReparam()}):
            z = npr.sample('z', dist.TransformedDistribution(
                dist.Normal(0, 1),
                transforms.AffineTransform(mu_z.squeeze(), sig_z.squeeze())
            ))
        
    # with npr.handlers.scale(scale=scale_term):
    #     with npr.plate('J_u', J_u):
    #         beta = npr.sample('beta', dist.Normal().expand([L]).to_event(1))
    
    with npr.handlers.scale(scale=scale_term):
        with npr.plate('J_u', J_u):
            # beta ~ N_+(0,1), truncated to be strictly positive
            beta = npr.sample(
                'beta',
                dist.TruncatedNormal(
                    low=0.0,
                    loc=jnp.zeros(L),
                    scale=jnp.ones(L)
                ).to_event(1)
            )

    cutpoints = {}
    
    for h, j in J_u_dict.items():
        with npr.plate('J_h' + h, j):
            with npr.handlers.reparam(config={'c_' + h: TransformReparam()}):
                cutpoints[h] = npr.sample('c_' + h, 
                        dist.TransformedDistribution(
                                dist.Dirichlet(5.0 * jnp.ones([H_CUTOFFS[h]+1])),
                                dist.transforms.SimplexToOrderedTransform()
                                ))
                    
    # sample across 3 firms
    partialized_f_sample_K = jax.tree_util.Partial(f_sample_K,
                                cutpoints=cutpoints,
                                beta=beta,
                                phi_nn=phi_nn,
                                model_params_phi=model_params_phi,
                                N=N,
                                J_u=J_u,
                                J_u_dict=J_u_dict,
                                J_u_idx_start=J_u_idx_start,
                                J_u_idx_end=J_u_idx_end,
                                L=L,
                                T=T,
                                scale_term=scale_term)
    

    # split ideal-points across firms
    z_1, z_2, z_3 = jnp.split(z, 3, axis=0)
    
    # run sample statements
    partialized_f_sample_K(1, Y_1_dummy, Y_q_1, z_1)
    partialized_f_sample_K(2, Y_2_dummy, Y_q_2, z_2)
    partialized_f_sample_K(3, Y_3_dummy, Y_q_3, z_3)

    return z

# Generate samples using Predictive class
def generate_samples(key, Y_c, mu_z, sig_z, Y_1_dummy, Y_2_dummy, Y_3_dummy, 
                        Y_q_1, Y_q_2, Y_q_3,
                        J_u, J_u_dict, phi_nn, model_params_phi,
                        N, J_u_idx_start, J_u_idx_end, 
                        L, T, scale_term, H_CUTOFFS, num_samples=1):
    
    predictive = Predictive(probabilistic_model, {}, num_samples=num_samples)
    
    key, subkey = random.split(key, 2)
    samples = predictive(subkey, Y_c=Y_c, mu_z=mu_z, sig_z=sig_z, 
                Y_1_dummy=Y_1_dummy, Y_2_dummy=Y_2_dummy, Y_3_dummy=Y_3_dummy, 
                Y_q_1=Y_q_1, Y_q_2=Y_q_2, Y_q_3=Y_q_3,
                J_u=J_u, J_u_dict=J_u_dict, phi_nn=phi_nn, model_params_phi=model_params_phi,
                N=N, J_u_idx_start=J_u_idx_start, J_u_idx_end=J_u_idx_end, 
                L=L, T=T, scale_term=scale_term, H_CUTOFFS=H_CUTOFFS)
    
    return samples

def save_jax_arrays_as_npy(arrays_dict, folder_name='saved_jax_arrays'):
    """
    Saves JAX arrays in a given dictionary as .npy files in the specified folder.

    Parameters:
    arrays_dict (dict): A dictionary of JAX arrays.
    folder_name (str): The name of the folder to save the arrays in.
    """
    # Ensure the folder exists or create it
    os.makedirs(folder_name, exist_ok=True)

    for key, array in arrays_dict.items():
        # Convert JAX array to a NumPy array for saving
        np_array = np.array(array)
        file_path = os.path.join(folder_name, f'{key}.npy')
        np.save(file_path, np_array)
        print(f'Saved {key} to {file_path}')
        

def save_jax_array_as_npy(jax_array, array_name, folder_name='saved_jax_arrays'):
    """
    Saves a single JAX array as a .npy file in the specified folder.

    Parameters:
    jax_array (jax.numpy.ndarray): A JAX array to save.
    array_name (str): The name to save the .npy file as.
    folder_name (str): The name of the folder to save the array in.
    """
    # Ensure the folder exists or create it
    os.makedirs(folder_name, exist_ok=True)

    # Convert the JAX array to a NumPy array for saving
    np_array = np.array(jax_array)
    file_path = os.path.join(folder_name, f'{array_name}.npy')
    np.save(file_path, np_array)
    print(f'Saved {array_name} to {file_path}')

def reshape_first_n(arrays_dict, n):
    """
    Reshapes the first n objects in the given dictionary from (1, N, J, T) to (N, J, T).

    Parameters:
    arrays_dict (dict): A dictionary of JAX arrays.

    Returns:
    dict: The updated dictionary with the first 6 objects reshaped.
    """
    reshaped_dict = {}
    keys = list(arrays_dict.keys())
    for i, key in enumerate(keys):
        array = arrays_dict[key]
        if i < n:
            reshaped_dict[key] = array.squeeze(axis=0)  # Remove the first axis
        else:
            reshaped_dict[key] = array  # Keep the original array for other keys
    return reshaped_dict

    
def main():

    key = random.PRNGKey(0) # initial key for pseudorandom number generator
    K, T = 3, 10 # no of competitors, no of time periods (quarters), 
    N1, N2, N3 = 5000, 5000, 5000 # no of customers
    N = N1 + N2 + N3
    Ju1, Ju2, Ju3,  = 4, 1, 2 # no of unique percpetion questions
    J_c, J_o = 321, 33 # number of static + optim variables
    J = J_c + J_o
    J_u_dict = {'11': 2, '10': 1, '5': 4} # number of perception questions for each scale points (ex: there are two 11-scale point questions)
    J_u_idx_start = {'11': 0, '10': 2, '5': 3} # when index starts
    J_u_idx_end = {'11': 2, '10': 3, '5': 7} # when index ends
    J_u = sum(J_u_dict.values()) # total number of perception questions (Y variables)
    L = args.latent_dims # hidden dimension of service quality
    Q = 111 # number of KPI per quarter
    H_CUTOFFS = {"11" : 10, "10": 9, "5" : 4} # number of cut points for each perception question (n-1 scale points)
    scale_term = 1.0 # only used at inference time, 1.0 at simulation time

    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    
    # # load common variables
    # Y_c_1_static = random.normal(subkey1, (N1, J_c, T, K)) 
    # Y_c_2_static = random.normal(subkey2, (N2, J_c, T, K)) 
    # Y_c_3_static = random.normal(subkey3, (N3, J_c, T, K)) 
    
    # key, subkey1, subkey2, subkey3 = random.split(key, 4)

    # Y_c_1_optim = random.normal(subkey1, (N1, J_o, T, K)) 
    # Y_c_2_optim = random.normal(subkey2, (N2, J_o, T, K)) 
    # Y_c_3_optim = random.normal(subkey3, (N3, J_o, T, K)) 
    

    # FOR STATIC INPUTS, input dim=2 add (k, t) (N,J+2,T)
    ## load only T periods of 10-Q data
    key, subkey1, subkey2, subkey3 = random.split(key, 4)

    Y_q_1 = random.normal(subkey1, (T, Q)) 
    Y_q_2 = random.normal(subkey2, (T, Q)) 
    Y_q_3 = random.normal(subkey3, (T, Q)) 
    
    # get dim related to 10-Q
    assert Y_q_1.shape == Y_q_2.shape and Y_q_1.shape == Y_q_3.shape    
    
    J_u = sum(J_u_dict.values())
    
    # Example usage
    hidden_dim = args.hidden_dims
    key, subkey = random.split(key, 2)
    Y_c = jax.random.normal(subkey, (N, T, J))    

    # generate dummies for correct shape taking
    Y_u_1_11_dummy = jnp.ones((N1, Ju1, T))
    Y_u_1_10_dummy = jnp.ones((N1, Ju2, T))
    Y_u_1_5_dummy = jnp.ones((N1, Ju3, T))
    Y_u_2_11_dummy = jnp.ones((N2, Ju1, T))
    Y_u_2_10_dummy = jnp.ones((N2, Ju2, T))
    Y_u_2_5_dummy = jnp.ones((N2, Ju3, T))
    Y_u_3_11_dummy = jnp.ones((N3, Ju1, T))
    Y_u_3_10_dummy = jnp.ones((N3, Ju2, T))
    Y_u_3_5_dummy = jnp.ones((N3, Ju3, T))
     
    # coalesce Y
    Y_1_dummy = { k:v for (k, v) in zip( H_CUTOFFS.keys(), (Y_u_1_11_dummy, Y_u_1_10_dummy, Y_u_1_5_dummy) ) }
    Y_2_dummy = { k:v for (k, v) in zip( H_CUTOFFS.keys(), (Y_u_2_11_dummy, Y_u_2_10_dummy, Y_u_2_5_dummy) ) }
    Y_3_dummy = { k:v for (k, v) in zip( H_CUTOFFS.keys(), (Y_u_3_11_dummy, Y_u_3_10_dummy, Y_u_3_5_dummy) ) }

    # generate mu_z, sig_z, which are parameters for the latent variables
    mu_z, sig_z, phi_nn, model_params_phi = generate_synthetic_data(subkey,
                                            Y_c, 
                                            hidden_size1=hidden_dim,
                                            hidden_size2=hidden_dim*2, 
                                            output_size=1,
                                            L=L,
                                            Q=Q)
    
    print(f"mu shape: {mu_z.shape}, sig shape: {sig_z.shape}")

    key, subkey = random.split(key, 2)

    # phi_nn = None
    # generates latent variables + dependent variables from the probabilistic choice model
    samples = generate_samples(subkey, Y_c, mu_z, sig_z, Y_1_dummy, Y_2_dummy, Y_3_dummy, 
                        Y_q_1, Y_q_2, Y_q_3,
                        J_u, J_u_dict, phi_nn, model_params_phi,
                        N1, J_u_idx_start, J_u_idx_end, 
                        L, T, scale_term, H_CUTOFFS, num_samples=1)
    
    samples = reshape_first_n(samples, K * len(J_u_dict))


    Y_c_1, Y_c_2, Y_c_3 = jnp.split(Y_c, 3)
    split_index = J_c
    Y_c_1_static, Y_c_1_optim = Y_c_1[..., :split_index], Y_c_1[..., split_index:]
    Y_c_2_static, Y_c_2_optim = Y_c_2[..., :split_index], Y_c_2[..., split_index:]
    Y_c_3_static, Y_c_3_optim = Y_c_3[..., :split_index], Y_c_3[..., split_index:]

    Y_c_1_static = jnp.moveaxis(Y_c_1_static, source=2, destination=1)
    Y_c_2_static = jnp.moveaxis(Y_c_2_static, source=2, destination=1)
    Y_c_3_static = jnp.moveaxis(Y_c_3_static, source=2, destination=1)
    Y_c_1_optim = jnp.moveaxis(Y_c_1_optim, source=2, destination=1)
    Y_c_2_optim = jnp.moveaxis(Y_c_2_optim, source=2, destination=1)
    Y_c_3_optim = jnp.moveaxis(Y_c_3_optim, source=2, destination=1)

    assert Y_c_1_static.shape[0] == Y_c_1_optim.shape[0] and \
            Y_c_2_static.shape[0] == Y_c_2_optim.shape[0] and \
            Y_c_3_static.shape[0] == Y_c_3_optim.shape[0]
    
    assert Y_c_1_static.shape[1:] == Y_c_2_static.shape[1:] and \
            Y_c_2_static.shape[1:] == Y_c_3_static.shape[1:]
    
    assert Y_c_1_optim.shape[1:] == Y_c_2_optim.shape[1:] and \
            Y_c_2_optim.shape[1:] == Y_c_3_optim.shape[1:]
    
    assert Y_c_1_static.shape[2] == Y_c_1_optim.shape[2] and \
            Y_c_2_static.shape[2] == Y_c_2_optim.shape[2] and \
            Y_c_3_static.shape[2] == Y_c_3_optim.shape[2]
        
    # save dictionary into .npy
    save_jax_arrays_as_npy(samples, "simulated_data")
    # save other data into .npy
    save_jax_array_as_npy(Y_c, "Y_c", "simulated_data")
    save_jax_array_as_npy(Y_q_1, "Y_q_1", "simulated_data")
    save_jax_array_as_npy(Y_q_2, "Y_q_2", "simulated_data")
    save_jax_array_as_npy(Y_q_3, "Y_q_3", "simulated_data")
    save_jax_array_as_npy(Y_c_1_static, "Y_c_1_static", "simulated_data")
    save_jax_array_as_npy(Y_c_2_static, "Y_c_2_static", "simulated_data")
    save_jax_array_as_npy(Y_c_3_static, "Y_c_3_static", "simulated_data")
    save_jax_array_as_npy(Y_c_1_optim, "Y_c_1_optim", "simulated_data")
    save_jax_array_as_npy(Y_c_2_optim, "Y_c_2_optim", "simulated_data")
    save_jax_array_as_npy(Y_c_3_optim, "Y_c_3_optim", "simulated_data")
    
    save_jax_array_as_npy(J_c, "J_c", "simulated_data") 
    save_jax_array_as_npy(J_u, "J_u", "simulated_data")
    with open('simulated_data/J_u_dict.pkl', 'wb') as f:
        pickle.dump(J_u_dict, f)
    with open('simulated_data/J_u_idx_start.pkl', 'wb') as f:
        pickle.dump(J_u_idx_start, f)
    with open('simulated_data/J_u_idx_end.pkl', 'wb') as f:
        pickle.dump(J_u_idx_end, f)
    # save_jax_array_as_npy(J_u_dict, "J_u_dict", "simulated_data")
    # save_jax_array_as_npy(J_u_idx_start, "J_u_idx_start", "simulated_data")
    # save_jax_array_as_npy(J_u_idx_end, "J_u_idx_end", "simulated_data")
    save_jax_array_as_npy(Q, "Q", "simulated_data")
    save_jax_array_as_npy(T, "T", "simulated_data")


#########
## run ##
#########

if __name__ == '__main__':
    main()
    