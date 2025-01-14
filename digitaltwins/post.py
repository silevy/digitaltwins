import os
import pickle
import numpy

from jax import random, numpy as jnp
from numpyro import infer

from . import inout
            


def reconstruct(rng_key, model, guide, params, fetch_all_u, fetch_all_c, N_batch, static_kwargs):
    
    # I/O, parameters
    with open(os.path.join(inout.RESULTS_DIR, "param.pkl"), 'wb') as f:
        pickle.dump(params, f)

    post_pred_dist = infer.Predictive(model=model, 
                                      guide=guide, 
                                      params=params, 
                                      num_samples=100,
                                      parallel=True)
    
    # posterior predictive
    rng_key, rng_key_post = random.split(rng_key, 2)
    Y_u_sites = list(fetch_all_u(0))
    Y_u_orig, Y_u_post = {}, {}
        
    for i in range(N_batch):
        model_args_post = {**fetch_all_c(i), **static_kwargs}
        post_samples = post_pred_dist(rng_key_post, **model_args_post)

        for site in Y_u_sites:
            if site in Y_u_post:
                Y_u_orig[site] = jnp.concatenate([Y_u_orig[site], fetch_all_u(i)[site]], axis=0)
                Y_u_post[site] = jnp.concatenate([Y_u_post[site], post_samples[site]], axis=1)
            else:
                Y_u_orig[site] = fetch_all_u(i)[site]
                Y_u_post[site] = post_samples[site]
     
    f_mae = lambda x : numpy.mean(numpy.abs(x), axis=(0,2))
    mae = { site : f_mae(Y_u_orig[site] - Y_u_post[site].mean(axis=0)) for site in Y_u_sites }

    # save to file
    # for site in Y_u_sites:
    #     numpy.save(os.path.join(inout.RESULTS_DIR, f"original_{site}.npy"), Y_u_orig[site])
    #     numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{site}.npy"), Y_u_post[site])
        
    return mae


def reconstruct_all(rng_key, model, guide, params, fetch_all_u, fetch_all_c, batch_num_train, batch_num_test, static_kwargs):
    
    # I/O, parameters
    with open(os.path.join(inout.RESULTS_DIR, "param.pkl"), 'wb') as f:
        pickle.dump(params, f)

    post_pred_dist = infer.Predictive(model=model, 
                                      guide=guide, 
                                      params=params, 
                                      num_samples=100,
                                      parallel=True)
    
    # posterior predictive
    rng_key, rng_key_post = random.split(rng_key, 2)
    Y_u_sites = list(fetch_all_u(0))
    Y_u_orig, Y_u_post = {}, {}
        
    batch_num_total = batch_num_train + batch_num_test
    for i in range(batch_num_total):
        print(f'{i} out of {batch_num_total}')
        model_args_post = {**fetch_all_c(i), **static_kwargs}
        post_samples = post_pred_dist(rng_key_post, **model_args_post)

        for site in Y_u_sites:
            if site in Y_u_post:
                Y_u_orig[site] = jnp.concatenate([Y_u_orig[site], fetch_all_u(i)[site]], axis=0)
                Y_u_post[site] = jnp.concatenate([Y_u_post[site], post_samples[site]], axis=1)
            else:
                Y_u_orig[site] = fetch_all_u(i)[site]
                Y_u_post[site] = post_samples[site]
     
    # f_mae = lambda x : numpy.mean(numpy.abs(x), axis=(0,2))
    # mae = { site : f_mae(Y_u_orig[site] - Y_u_post[site].mean(axis=0)) for site in Y_u_sites }

    # save to file
    for site in Y_u_sites:
        numpy.save(os.path.join(inout.RESULTS_DIR, f"original_{site}.npy"), Y_u_orig[site])
        numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{site}.npy"), Y_u_post[site])
        
    # return mae



def reconstruct_mcmc(rng_key, model, mcmc_samples, fetch_all_u, fetch_all_c, N_batch, static_kwargs):

    """
    Generate posterior predictive samples using MCMC draws (mcmc_samples) 
    and save them to disk for posterior checks.

    Args:
        rng_key: jax.random.PRNGKey
        model: the same model function used in MCMC
        mcmc_samples: dictionary of posterior samples from MCMC
        fetch_all_u, fetch_all_c: data loader functions
        static_kwargs: dictionary of additional data needed for model
    """

    # 1) Create a Predictive object using MCMC posterior samples
    post_pred_dist = infer.Predictive(
        model=model,
        posterior_samples=mcmc_samples,  # <--- key difference vs SVI
        num_samples=100,                 # how many draws from each chain
        parallel=True
    )

    # No need to save "params.pkl" for MCMC, unless you want to store
    # chain draws in a .pkl, but that usually happens in the main script.

    rng_key, rng_key_post = random.split(rng_key, 2)

    # all_u = {}
    # all_c = {}
    # # total number of batches is (batch_num_train + batch_num_test)
    # for i in range(N_batch):
    #     # 1) Grab data from batch i
    #     u_dict = fetch_all_u(i)  # e.g. {'Y_u_1_5': ..., 'Y_u_1_10': ...}
    #     c_dict = fetch_all_c(i)  # e.g. {'Y_c_1_static': ..., 'Y_c_1_optim': ...}

    #     # 2) Append each array to the corresponding list in all_u, all_c
    #     for k, v in u_dict.items():
    #         if k not in all_u:
    #             all_u[k] = []
    #         all_u[k].append(v)

    #     for k, v in c_dict.items():
    #         if k not in all_c:
    #             all_c[k] = []
    #         all_c[k].append(v)

    # for k in all_u:
    #     all_u[k] = jnp.concatenate(all_u[k], axis=0)  # merges along N dimension

    # for k in all_c:
    #     all_c[k] = jnp.concatenate(all_c[k], axis=0)

    Y_u_sites = list(fetch_all_u(0))
    Y_u_orig, Y_u_post = {}, {}
    

    # model_args_post = {**fetch_all_c(i), **static_kwargs}
    # # posterior predictive draws
    # post_samples = post_pred_dist(rng_key_post, **model_args_post)

    # for site in Y_u_sites:
    #     # store original and predicted
    #     if site in Y_u_post:
    #         Y_u_orig[site] = jnp.concatenate([Y_u_orig[site], fetch_all_u(i)[site]], axis=0)
    #         Y_u_post[site] = jnp.concatenate([Y_u_post[site], post_samples[site]], axis=1)
    #     else:
    #         Y_u_orig[site] = fetch_all_u(i)[site]
    #         Y_u_post[site] = post_samples[site]

    # 2) Loop over each data batch
    for i in range(N_batch):
        model_args_post = {**fetch_all_c(i), **static_kwargs}
        # posterior predictive draws
        post_samples = post_pred_dist(rng_key_post, **model_args_post)

        for site in Y_u_sites:
            # store original and predicted
            if site in Y_u_post:
                Y_u_orig[site] = jnp.concatenate([Y_u_orig[site], fetch_all_u(i)[site]], axis=0)
                Y_u_post[site] = jnp.concatenate([Y_u_post[site], post_samples[site]], axis=1)
            else:
                Y_u_orig[site] = fetch_all_u(i)[site]
                Y_u_post[site] = post_samples[site]

    # 3) Compute MAE (or any other metric) and save .npy files
    f_mae = lambda x: jnp.mean(jnp.abs(x), axis=(0,2))
    mae = {site: f_mae(Y_u_orig[site] - Y_u_post[site].mean(axis=0)) for site in Y_u_sites}

    for site in Y_u_sites:
        numpy.save(os.path.join(inout.RESULTS_DIR, f"original_{site}.npy"), Y_u_orig[site])
        numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{site}.npy"), Y_u_post[site])
    
    return mae


def reconstruct_mcmc_all(rng_key, model, mcmc_samples, fetch_all_u, fetch_all_c, batch_num_train, batch_num_test, static_kwargs):

    post_pred_dist = infer.Predictive(
        model=model,
        posterior_samples=mcmc_samples,  # <--- key difference vs SVI
        num_samples=100,                 # how many draws from each chain
        parallel=True
    )
    
    # posterior predictive
    rng_key, rng_key_post = random.split(rng_key, 2)
    Y_u_sites = list(fetch_all_u(0))
    Y_u_orig, Y_u_post = {}, {}
        
    batch_num_total = batch_num_train + batch_num_test
    for i in range(batch_num_total):
        print(f'{i} out of {batch_num_total}')
        model_args_post = {**fetch_all_c(i), **static_kwargs}
        post_samples = post_pred_dist(rng_key_post, **model_args_post)

        for site in Y_u_sites:
            if site in Y_u_post:
                Y_u_orig[site] = jnp.concatenate([Y_u_orig[site], fetch_all_u(i)[site]], axis=0)
                Y_u_post[site] = jnp.concatenate([Y_u_post[site], post_samples[site]], axis=1)
            else:
                Y_u_orig[site] = fetch_all_u(i)[site]
                Y_u_post[site] = post_samples[site]
     
    f_mae = lambda x : numpy.mean(numpy.abs(x), axis=(0,2))
    mae = { site : f_mae(Y_u_orig[site] - Y_u_post[site].mean(axis=0)) for site in Y_u_sites }

    # save to file
    for site in Y_u_sites:
        numpy.save(os.path.join(inout.RESULTS_DIR, f"original_{site}.npy"), Y_u_orig[site])
        numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{site}.npy"), Y_u_post[site])
        
    return mae



def post_latent_sites(rng_key, 
                      args,
                      model, 
                      guide,
                      K: int = 3, # three firms
                      pkl_file_name :str = "param.pkl"):

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
    return_sites = ['beta',
                    *map(lambda k : 'alpha_' + str(k), range(1, K + 1)),
                    *map(lambda k : 'phi_' + str(k), range(1, K + 1))]
    
    post_pred_dist = infer.Predictive(model=model, 
                                      guide=guide, 
                                      params=params, 
                                      num_samples=1000,
                                      parallel=True,
                                      return_sites=return_sites)
    
    # run posterior predictive #
    # note: 0 index because any batch will do, as alpha, beta, phi are not dependent on Y_c or Y_u
    rng_key, rng_key_post = random.split(rng_key, 2)
    model_args_post = {**fetch_all_u(0), **fetch_all_c(0), **static_kwargs}
    post_samples = post_pred_dist(rng_key_post, **model_args_post)
    
    # save to file #
    for sites in return_sites:
        numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{sites}.npy"), post_samples[sites])



def post_Y_predictive(rng_key, 
                      args,
                      model, 
                      guide,
                      pkl_file_name :str = "param.pkl"):

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
                        # 'is_predictive' : True }
    
    
    # posterior predictive function #
    Y_u_sites = list(fetch_all_u(0))
    post_pred_dist = infer.Predictive(model=model, 
                                      guide=guide, 
                                      params=params, 
                                      num_samples=1000,
                                      parallel=True,
                                      return_sites=Y_u_sites)
    
    # run posterior predictive #
    post_samples_dict = {}
    N_batch = args.batch_post
    
    for i in range(N_batch):
        # step 1 - draw post samples per batch, which is a dictionary itself
        rng_key, rng_key_post = random.split(rng_key, 2)
        
        # Numpyro bug: need to initialize all sites in the model, otherwise it will throw an error
        model_args_post = {**fetch_all_u(i), **fetch_all_c(i), **static_kwargs}
        _ = post_pred_dist(rng_key_post, **model_args_post)
        # now re-run without Y_u observations
        model_args_post = {**fetch_all_c(i), **static_kwargs}
        post_samples = post_pred_dist(rng_key_post, **model_args_post)
        
        # step 2 - concatenate to post_samples_dict
        for site in Y_u_sites:
            if site in post_samples_dict:
                post_samples_dict[site] = jnp.concatenate([post_samples_dict[site], post_samples[site]], axis=1)
            else:
                post_samples_dict[site] = post_samples[site]
        

    # # save to file #
    for site, tensor in post_samples_dict.items():
        numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{site}.npy"), tensor)