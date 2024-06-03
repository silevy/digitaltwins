import os
import pickle
import numpy
import jax

from jax import random, numpy as jnp
from numpyro import infer

from . import inout



###############
# GRID SEARCH #
###############

def post_grid_search(rng_key,
                     args,
                     model,
                     guide,
                     # optional arguments w/ preset values #
                     pkl_file_name: str = "param_10k_newdata.pkl",
                     grid_size: float = 0.1, 
                     range_down: float = 2.1, 
                     range_up: float = 2.1,
                     is_digital_twins: bool = False,
                     digital_twins_k_idx: int = 1):

    # load params #
    with open(os.path.join(inout.RESULTS_DIR, pkl_file_name), 'rb') as f:
        params = pickle.load(f)


    # load data #
    rng_key, rng_etl = random.split(rng_key, 2)
    (   J_c, J_u, J_u_dict, J_u_idx_start, J_u_idx_end, Q, T,
        Y_q_1, Y_q_2, Y_q_3, batch_num_train, batch_num_test,
        fetch_train, fetch_test, fetch_all_u, fetch_all_c
    ) = inout.load_dataset(rng_key=rng_etl,
                     batch_size=args.batch_size, 
                     N_split=args.train_test)
    
    
    # assert and set dim J, N_batch #
    Y_c_optim = fetch_all_c(0)
    assert Y_c_optim['Y_c_1_optim'].shape == Y_c_optim['Y_c_2_optim'].shape and \
        Y_c_optim['Y_c_1_optim'].shape == Y_c_optim['Y_c_3_optim'].shape
                
    J = Y_c_optim['Y_c_1_optim'].shape[1]
    N_batch = args.batch_post
    
    
    # get static kwargs #
    static_kwargs = {   'Y_q_1' : Y_q_1,
                        'Y_q_2' : Y_q_2,
                        'Y_q_3' : Y_q_3,
                        'J_u_dict' : J_u_dict,
                        'J_u_idx_start' : J_u_idx_start,
                        'J_u_idx_end' : J_u_idx_end,
                        'J_c' : J_c, 'J_u' : J_u, 'Q' : Q, 'T' : T,
                        'L' : args.latent_dims, 
                        'hidden_dim' : args.hidden_dims,
                        'scale_term' : 1.0 / batch_num_train } # ,
                        # 'is_predictive' : False }
    
    
    # posterior predictive function #
    post_pred_dist = infer.Predictive(model=model, 
                                      guide=guide, 
                                      params=params, 
                                      num_samples=10,
                                      parallel=True,
                                      return_sites=['z'])
    
    
    # form grid #
    grid = jnp.linspace(-range_down, 
                        range_up, 
                        int((range_up + range_down) / grid_size + 1),
                        dtype=jnp.float32)
    grid_length = grid.shape[0]


    ##########################
    # helper functions BEGIN #
    ##########################
    
    def fetch_static_c_i(i: int):
        Y_c_dict = fetch_all_c(i)
        return {k: Y_c_dict[k] for k in ('Y_c_1_static', 'Y_c_2_static', 'Y_c_3_static')}
    
    
    def fetch_optim_c_i(i:int, j:int):
        Y_c_dict = fetch_all_c(i)
        Y_c_optim = list(map(lambda k: jnp.tile(Y_c_dict[k][None, :, :, :], (grid_length,1,1,1)), 
                            ['Y_c_1_optim', 'Y_c_2_optim', 'Y_c_3_optim']))
        return [ y.at[:,:,j,:].add(grid[:, None, None]) for y in Y_c_optim ]
    
    
    def batch_ppd_empirical(rng_key, i: int, j: int):
        rng_key_grid = jnp.tile(rng_key, (grid_length,1))
        
        Y_u_dict = fetch_all_u(i)
        Y_c_static_dict = fetch_static_c_i(i)
        Y_c_optim_increment = fetch_optim_c_i(i, j)
        
        # vmap post predictive, partialize all non-vmap arguments    
        def partialized_ppd(rng_key, Y_c_1_optim, Y_c_2_optim, Y_c_3_optim):
            return post_pred_dist(rng_key, 
                                Y_c_1_optim=Y_c_1_optim, 
                                Y_c_2_optim=Y_c_2_optim, 
                                Y_c_3_optim=Y_c_3_optim, 
                                **Y_u_dict, **Y_c_static_dict, **static_kwargs)    
        
        vmap_ppd = jax.vmap(partialized_ppd, in_axes=(0,0,0,0))
        post_samples = vmap_ppd(rng_key_grid, *Y_c_optim_increment)
        
        return post_samples['z']
    
    
    def batch_ppd_twins(rng_key, 
                        i: int,     # index of batch
                        j: int,     # index of question
                        k: int):    # index of firm to propagate twins
        
        # assert k is in [1, 2, 3]
        klist = [1, 2, 3]
        assert k in klist
        
        # STEP 1 - fetch
        Y_u_dict = fetch_all_u(i)
        Y_c_static_dict = fetch_static_c_i(i)
        Y_c_optim_increment = fetch_optim_c_i(i, j)

        # STEP 2 - propagate twins by selected firm, in each quarter
        # note: this version DOES NOT create twins across quarters
        def aug_t_k(y, k):
            y = y[:,:-2,:]
            return jnp.concatenate([y, 
                                    jnp.ones([y.shape[0], 1, y.shape[2]]) * k,
                                    jnp.tile(jnp.expand_dims(jnp.arange(T), (0,1)), reps=(y.shape[0],1,1))], 
                                    axis=1)
        
        Y_c_static_target = Y_c_static_dict[f"Y_c_{k}_static"]
        Y_c_static_keys = list(Y_c_static_dict.keys())
        Y_c_static_dict = { Y_c_static_keys[i-1] : aug_t_k(Y_c_static_target, i) for i in klist}
        Y_c_optim_increment = [Y_c_optim_increment[k-1] for _ in range(3)]
        
        # STEP 3 (OPTIONAL) - twins over time, based on t=0
        Y_c_static_dict = { key : jnp.tile(val[:,:,0,None], reps=(1,1,T)) for key, val in Y_c_static_dict.items() }
        Y_c_optim_increment = [ jnp.tile(y[:,:,:,0,None], reps=(1,1,1,T)) for y in Y_c_optim_increment ]
        
        # vmap post predictive, partialize all non-vmap arguments    
        def partialized_ppd(rng_key, Y_c_1_optim, Y_c_2_optim, Y_c_3_optim):
            return post_pred_dist(rng_key, 
                                Y_c_1_optim=Y_c_1_optim, 
                                Y_c_2_optim=Y_c_2_optim, 
                                Y_c_3_optim=Y_c_3_optim, 
                                **Y_u_dict, **Y_c_static_dict, **static_kwargs)
        
        # run
        rng_key_grid = jnp.tile(rng_key, (grid_length,1))
        vmap_ppd = jax.vmap(partialized_ppd, in_axes=(0,0,0,0))
        post_samples = vmap_ppd(rng_key_grid, *Y_c_optim_increment)
        
        return post_samples['z']
    
    ########################
    # helper functions END #
    ########################
    
    
    # run grid search sequentially over J questions #
    if is_digital_twins:
        batch_ppdf_fn = jax.tree_util.Partial(batch_ppd_twins, k=digital_twins_k_idx)
    else:
        batch_ppdf_fn = batch_ppd_empirical

    j = 0
    rng_key_list = random.split(rng_key, N_batch + 1)
    rng_key, rng_key_grid = rng_key_list[0], rng_key_list[1:]
    
    # run vmapped over batches
    batch_post_pred_j = jax.tree_util.Partial(batch_ppdf_fn, j=j)
    vmap_post_pred = jax.vmap(batch_post_pred_j, in_axes=(0, 0), out_axes=0)
    post_samples_z = vmap_post_pred(rng_key_grid, jnp.arange(N_batch))
    
    # save post samples to file
    if is_digital_twins:
        str_file_name = f"postgrid_twins_Y_c_k={digital_twins_k_idx}_optim_{j}.npy" 
    else:
        str_file_name = f"postgrid_Y_c_k_optim_{j}.npy"
        
    numpy.save(os.path.join(inout.RESULTS_DIR, str_file_name), post_samples_z) 

    # for j in range(J):
    #     rng_key_list = random.split(rng_key, N_batch + 1)
    #     rng_key, rng_key_grid = rng_key_list[0], rng_key_list[1:]
        
    #     # run vmapped over batches
    #     batch_post_pred_j = jax.tree_util.Partial(batch_ppdf_fn, j=j)
    #     vmap_post_pred = jax.vmap(batch_post_pred_j, in_axes=(0, 0), out_axes=0)
    #     post_samples_z = vmap_post_pred(rng_key_grid, jnp.arange(N_batch))
        
    #     # save post samples to file
    #     if is_digital_twins:
    #         str_file_name = f"postgrid_twins_Y_c_k={digital_twins_k_idx}_optim_{j}.npy" 
    #     else:
    #         str_file_name = f"postgrid_Y_c_k_optim_{j}.npy"
            
    #     numpy.save(os.path.join(inout.RESULTS_DIR, str_file_name), post_samples_z) 