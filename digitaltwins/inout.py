import os
import inspect
import numpy
import pandas as pd
import jax
import pickle
import numpy as np

from jax import random, lax, numpy as jnp

from . import main


############################
## make results directory ##
############################

HOME_PATH = os.path.dirname(inspect.getfile(lambda: None))
DATA_DIR = os.path.join(HOME_PATH, 'data')
RESULTS_DIR = os.path.join('results')
os.makedirs(RESULTS_DIR, exist_ok=True)


#########################
## read and parse data ##
#########################

def etl_simulation():

    # read files
    get_npy = lambda f : numpy.load(os.path.join("simulated_data", f), allow_pickle=True)
    
    # load common variables
    Y_c_1_static = get_npy('Y_c_1_static.npy')
    Y_c_2_static = get_npy('Y_c_2_static.npy')
    Y_c_3_static = get_npy('Y_c_3_static.npy')
    
    Y_c_1_optim = get_npy('Y_c_1_optim.npy')
    Y_c_2_optim = get_npy('Y_c_2_optim.npy')
    Y_c_3_optim = get_npy('Y_c_3_optim.npy')
    
    Y_u_1_5 = get_npy('Y_u_1_5.npy')
    Y_u_1_10 = get_npy('Y_u_1_10.npy')
    Y_u_1_11 = get_npy('Y_u_1_11.npy')
    
    Y_u_2_5 = get_npy('Y_u_2_5.npy')
    Y_u_2_10 = get_npy('Y_u_2_10.npy')
    Y_u_2_11 = get_npy('Y_u_2_11.npy')
    
    Y_u_3_5 = get_npy('Y_u_3_5.npy')
    Y_u_3_10 = get_npy('Y_u_3_10.npy')
    Y_u_3_11 = get_npy('Y_u_3_11.npy')

    Y_q_1 = get_npy('Y_q_1.npy')
    Y_q_2 = get_npy('Y_q_2.npy')
    Y_q_3 = get_npy('Y_q_3.npy')

    J_c = get_npy('J_c.npy')
    J_u = get_npy('J_u.npy')
    # J_u_dict = get_npy('J_u_dict.npy')
    # J_u_idx_start = get_npy('J_u_idx_start.npy')
    # J_u_idx_end = get_npy('J_u_idx_end.npy')
    with open('simulated_data/J_u_dict.pkl', 'rb') as f:
        J_u_dict = pickle.load(f)
    with open('simulated_data/J_u_idx_start.pkl', 'rb') as f:
        J_u_idx_start = pickle.load(f)
    with open('simulated_data/J_u_idx_end.pkl', 'rb') as f:
        J_u_idx_end = pickle.load(f)
    
    Q = get_npy('Q.npy')
    T = get_npy('T.npy')

    return J_c.item(), J_u.item(), J_u_dict, J_u_idx_start, J_u_idx_end, Q.item(), T.item(), \
            Y_u_1_11, Y_u_1_10, Y_u_1_5, \
            Y_u_2_11, Y_u_2_10, Y_u_2_5, \
            Y_u_3_11, Y_u_3_10, Y_u_3_5, \
            Y_c_1_static, Y_c_2_static, Y_c_3_static, \
            Y_c_1_optim, Y_c_2_optim, Y_c_3_optim, \
            Y_q_1, Y_q_2, Y_q_3
            
def etl():
        
    # read files
    get_csv = lambda f : pd.read_csv(os.path.join(DATA_DIR, f), header=0).values
    get_npy = lambda f : numpy.load(os.path.join(DATA_DIR, f), allow_pickle=True)
    
    
    # load common variables
    Y_c_1_static = get_npy('Y_c_1_static.npy')
    Y_c_2_static = get_npy('Y_c_2_static.npy')
    Y_c_3_static = get_npy('Y_c_3_static.npy')
    
    Y_c_1_optim = get_npy('Y_c_1_optim.npy')
    Y_c_2_optim = get_npy('Y_c_2_optim.npy')
    Y_c_3_optim = get_npy('Y_c_3_optim.npy')
    
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
    
    J_c = Y_c_1_static.shape[1]
    T = Y_c_1_static.shape[2]
    
    # FOR STATIC INPUTS, input dim=2 add (k, t) (N,J+2,T)
    def aug_t_k(y, k):
        return numpy.concatenate([y, 
                                numpy.ones([y.shape[0], 1, y.shape[2]]) * k,
                                numpy.tile(numpy.expand_dims(numpy.arange(T), (0,1)), reps=(y.shape[0],1,1))], 
                                axis=1)
    
    Y_c_1_static = jnp.asarray(aug_t_k(Y_c_1_static, 1), dtype=jnp.float64)
    Y_c_2_static = jnp.asarray(aug_t_k(Y_c_2_static, 2), dtype=jnp.float64)
    Y_c_3_static = jnp.asarray(aug_t_k(Y_c_3_static, 3), dtype=jnp.float64)

    
    ## load only T periods of 10-Q data
    Y_q_1 = jnp.asarray(get_csv('Y_q_1.csv')[:,:T], dtype=jnp.float64).T
    Y_q_2 = jnp.asarray(get_csv('Y_q_2.csv')[:,:T], dtype=jnp.float64).T
    Y_q_3 = jnp.asarray(get_csv('Y_q_3.csv')[:,:T], dtype=jnp.float64).T
    
    # get dim related to 10-Q
    assert Y_q_1.shape == Y_q_2.shape and Y_q_1.shape == Y_q_3.shape
    Q = Y_q_1.shape[1]
    
    
    # load uncommon variables
    def load_Y_u(path_str):
        a = get_npy(path_str)
        a = a[~(numpy.isnan(a) | (a<0)).any(axis=(1,2)), :, :]
        a -= a.min()
        return jnp.asarray(a, dtype=jnp.int32)

    Y_u_1_5 = load_Y_u('Y_u_1_5.npy')
    Y_u_1_10 = load_Y_u('Y_u_1_10.npy')
    Y_u_1_11 = load_Y_u('Y_u_1_11.npy')
    
    Y_u_2_5 = load_Y_u('Y_u_2_5.npy')
    Y_u_2_10 = load_Y_u('Y_u_2_10.npy')
    Y_u_2_11 = load_Y_u('Y_u_2_11.npy')
    
    Y_u_3_5 = load_Y_u('Y_u_3_5.npy')
    Y_u_3_10 = load_Y_u('Y_u_3_10.npy')
    Y_u_3_11 = load_Y_u('Y_u_3_11.npy')

    
    # form J-related position indices
    J_u_dict = { k : v for (k, v) in \
        zip(main.H_CUTOFFS.keys(), [Y_u_1_11.shape[1], Y_u_1_10.shape[1], Y_u_1_5.shape[1]]) } #Y_u_1_2.shape[1]
    
    J_u_idx_end = { k : i for (k, i) in \
        zip( main.H_CUTOFFS.keys(), numpy.cumsum(list(J_u_dict.values()), dtype=numpy.int32) ) }
    
    J_u_idx_start = { k : i for (k, i) in \
                        zip(main.H_CUTOFFS.keys(), 
                            numpy.concatenate([numpy.zeros([1], dtype=numpy.int32), 
                                               numpy.cumsum(list(J_u_dict.values()), dtype=numpy.int32)]
                                              ) 
                            ) 
                    }
    
    J_u = sum(J_u_dict.values())
    
    
    return J_c, J_u, J_u_dict, J_u_idx_start, J_u_idx_end, Q, T, \
            Y_u_1_11, Y_u_1_10, Y_u_1_5, \
            Y_u_2_11, Y_u_2_10, Y_u_2_5, \
            Y_u_3_11, Y_u_3_10, Y_u_3_5, \
            Y_c_1_static, Y_c_2_static, Y_c_3_static, \
            Y_c_1_optim, Y_c_2_optim, Y_c_3_optim, \
            Y_q_1, Y_q_2, Y_q_3
            
            

###########################
## main public interface ##
###########################

def load_dataset(rng_key, batch_size, N_split):
    (   J_c, J_u, J_u_dict, J_u_idx_start, J_u_idx_end, Q, T,
        Y_u_1_11, Y_u_1_10, Y_u_1_5,
        Y_u_2_11, Y_u_2_10, Y_u_2_5,
        Y_u_3_11, Y_u_3_10, Y_u_3_5,
        Y_c_1_static, Y_c_2_static, Y_c_3_static,
        Y_c_1_optim, Y_c_2_optim, Y_c_3_optim,
        Y_q_1, Y_q_2, Y_q_3     ) = etl_simulation()
    Q = np.int32(str(Q))
    J_c = np.int32(str(J_c))
    J_u =  np.int32(str(J_u))
    T = np.int32(str(T))
    
    # batch sizes
    N_arr = [Y_c_1_static.shape[0], Y_c_2_static.shape[0], Y_c_3_static.shape[0]]
    N_max = numpy.max(N_arr)
    # batch_num_train = (N_max-N_split) // batch_size + 1
    # batch_num_test = N_split // batch_size + 1
    batch_num_train = (N_max - N_split + batch_size - 1) // batch_size
    batch_num_test  = (N_split + batch_size - 1) // batch_size

    # create train vs. test randomized indices
    rng_key, rng_key_perm = random.split(rng_key, 2)
    rand_idx_arr = [random.permutation(rng_key_perm, n) for n in N_arr]
    rand_train_arr = [i[N_split:] for i in rand_idx_arr]
    rand_test_arr = [i[:N_split] for i in rand_idx_arr]

    # callable, partializable fetch function
    def fetch_all_u(i):
        # return all observations, equalized in dim=0 length
        ret_idx = lax.dynamic_slice_in_dim(numpy.arange(N_max), i * batch_size, batch_size)

        return {    'Y_u_1_11' : lax.index_take(Y_u_1_11, (ret_idx,), axes=(0,)),
                    'Y_u_1_10' : lax.index_take(Y_u_1_10, (ret_idx,), axes=(0,)),
                    'Y_u_1_5' : lax.index_take(Y_u_1_5, (ret_idx,), axes=(0,)),
                    'Y_u_2_11' : lax.index_take(Y_u_2_11, (ret_idx,), axes=(0,)),
                    'Y_u_2_10' : lax.index_take(Y_u_2_10, (ret_idx,), axes=(0,)),
                    'Y_u_2_5' : lax.index_take(Y_u_2_5, (ret_idx,), axes=(0,)),
                    'Y_u_3_11' : lax.index_take(Y_u_3_11, (ret_idx,), axes=(0,)),
                    'Y_u_3_10' : lax.index_take(Y_u_3_10, (ret_idx,), axes=(0,)),
                    'Y_u_3_5' : lax.index_take(Y_u_3_5, (ret_idx,), axes=(0,))      }
    
    def fetch_all_c(i):
        # return all observations, equalized in dim=0 length
        ret_idx = lax.dynamic_slice_in_dim(numpy.arange(N_max), i * batch_size, batch_size)

        return  {   'Y_c_1_static' : lax.index_take(Y_c_1_static, (ret_idx,), axes=(0,)),
                    'Y_c_2_static' : lax.index_take(Y_c_2_static, (ret_idx,), axes=(0,)),
                    'Y_c_3_static' : lax.index_take(Y_c_3_static, (ret_idx,), axes=(0,)),
                    'Y_c_1_optim' : lax.index_take(Y_c_1_optim, (ret_idx,), axes=(0,)),
                    'Y_c_2_optim' : lax.index_take(Y_c_2_optim, (ret_idx,), axes=(0,)),
                    'Y_c_3_optim' : lax.index_take(Y_c_3_optim, (ret_idx,), axes=(0,))      }
    
    def fetch_(i, idx_arr):
        ret_idx_1 = lax.dynamic_slice_in_dim(idx_arr[0], i * batch_size, batch_size)
        ret_idx_2 = lax.dynamic_slice_in_dim(idx_arr[1], i * batch_size, batch_size)
        ret_idx_3 = lax.dynamic_slice_in_dim(idx_arr[2], i * batch_size, batch_size)

        return lax.index_take(Y_u_1_11, (ret_idx_1,), axes=(0,)), \
                lax.index_take(Y_u_1_10, (ret_idx_1,), axes=(0,)), \
                lax.index_take(Y_u_1_5, (ret_idx_1,), axes=(0,)), \
                lax.index_take(Y_u_2_11, (ret_idx_2,), axes=(0,)), \
                lax.index_take(Y_u_2_10, (ret_idx_2,), axes=(0,)), \
                lax.index_take(Y_u_2_5, (ret_idx_2,), axes=(0,)), \
                lax.index_take(Y_u_3_11, (ret_idx_3,), axes=(0,)), \
                lax.index_take(Y_u_3_10, (ret_idx_3,), axes=(0,)), \
                lax.index_take(Y_u_3_5, (ret_idx_3,), axes=(0,)), \
                lax.index_take(Y_c_1_static, (ret_idx_1,), axes=(0,)), \
                lax.index_take(Y_c_2_static, (ret_idx_2,), axes=(0,)), \
                lax.index_take(Y_c_3_static, (ret_idx_3,), axes=(0,)), \
                lax.index_take(Y_c_1_optim, (ret_idx_1,), axes=(0,)), \
                lax.index_take(Y_c_2_optim, (ret_idx_2,), axes=(0,)), \
                lax.index_take(Y_c_3_optim, (ret_idx_3,), axes=(0,))
    
    fetch_train = jax.tree_util.Partial(fetch_, idx_arr=rand_train_arr)
    fetch_test = jax.tree_util.Partial(fetch_, idx_arr=rand_test_arr)

    return J_c, J_u, J_u_dict, J_u_idx_start, J_u_idx_end, Q, T, \
            Y_q_1, Y_q_2, Y_q_3, \
            batch_num_train, batch_num_test, \
            fetch_train, fetch_test, fetch_all_u, fetch_all_c