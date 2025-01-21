import os
import pickle
import numpy
import numpyro as npr
from .args import get_parser # Import the parser logic
from jax import random, numpy as jnp
from numpyro import infer
from jax import vmap
from numpyro import handlers
import pandas as pd
import numpy as np
import jax

from . import inout
from . import model, inout, post, optim, util
        
from numpyro.diagnostics import summary

import matplotlib.pyplot as plt

# **Function to Display Parameters and Shapes:**
def display_parameter_shapes(samples):
    """
    Prints the keys and shapes of parameters in the given (MCMC) samples.

    Args:
        mcmc_samples (dict): Dictionary containing (MCMC) samples, with keys as parameter names
                             and values as sampled arrays.
    """
    print("Parameter Shapes:")
    for key, value in samples.items():
        print(f"{key}: {value.shape}")


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



# # helper function for prediction
# def predict_mcmc(model, rng_key, samples, Y_u_sites, model_args_post):
#     model = handlers.substitute(handlers.seed(model, rng_key), samples)
#     # note that Y will be sampled in the model because we pass Y=None here
#     model_trace = handlers.trace(model).get_trace(**model_args_post)
#     return model_trace[Y_u_sites]["value"]

# def predict_mcmc(model, rng_key, samples, Y_u_sites, model_args_post):
#     results = []
#     size_post_samples = samples["z"].shape[1]
#     display_parameter_shapes(samples)
#     # Iterate over posterior samples
#     for i in range(samples["z"].shape[1]):  # Assuming shape (1, 100, ...)
#         print(f"{i+1} out of {size_post_samples}")
#         single_sample = {k: v[:, i, ...] for k, v in samples.items()}
#         model_single = handlers.substitute(handlers.seed(model, rng_key), single_sample)
#         trace = handlers.trace(model_single).get_trace(**model_args_post)
#         results.append(trace[Y_u_sites]["value"])
#     return jnp.stack(results, axis=1)  # Combine along sample dimension


def remove_first_dim(samples_dict):
    """
    Returns a copy of samples_dict where each array's
    first dimension has been removed (assuming size=1).
    """
    new_dict = {}
    for k, v in samples_dict.items():
        # Check that the first dim is actually 1
        if v.shape[0] != 1:
            raise ValueError(f"Expected first dimension to be 1, but got {v.shape} for key '{k}'.")
        
        # Remove the first dimension
        new_dict[k] = jnp.squeeze(v, axis=0)
    return new_dict


def reconstruct_mcmc(rng_key, model, mcmc_samples, fetch_all_u, fetch_all_c, N_batch, static_kwargs, args):

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
    
    mcmc_samples2 = remove_first_dim(mcmc_samples)

    # 1) Create a Predictive object using MCMC posterior samples
    post_pred_dist = infer.Predictive(
        model=model,
        posterior_samples=mcmc_samples2,  # <--- key difference vs SVI
        # num_samples=100,                 # how many draws from each chain
        parallel=True,
        return_sites=Y_u_sites,
        batch_ndims=1
    )

    # 2) Loop over each data batch
    # for i in range(N_batch):
    i = 0
    model_args_gpu = {**fetch_all_c(i), **static_kwargs}
    # posterior predictive draws

    # posterior predictives
    cpu_device = jax.devices("cpu")[0]
    jax.config.update("jax_default_device", cpu_device)
    # model_args_cpu = [ jax.device_put(arg, cpu_device) if isinstance(arg, jax.Array) else arg for arg in model_args_gpu]
    model_args_cpu = {
        k: jax.device_put(v, cpu_device) if isinstance(v, jax.Array) else v
        for k, v in model_args_gpu.items()
    }
    post_samples = post_pred_dist(rng_key_post, **model_args_cpu)
    
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


# def reconstruct_mcmc_all(rng_key, model, mcmc_samples, fetch_all_u, fetch_all_c, batch_num_train, batch_num_test, static_kwargs):

#     post_pred_dist = infer.Predictive(
#         model=model,
#         posterior_samples=mcmc_samples,  # <--- key difference vs SVI
#         num_samples=100,                 # how many draws from each chain
#         parallel=True
#     )
    
#     # posterior predictive
#     rng_key, rng_key_post = random.split(rng_key, 2)
#     Y_u_sites = list(fetch_all_u(0))
#     Y_u_orig, Y_u_post = {}, {}
        
#     batch_num_total = batch_num_train + batch_num_test
#     for i in range(batch_num_total):
#         print(f'{i} out of {batch_num_total}')
#         model_args_post = {**fetch_all_c(i), **static_kwargs}
#         post_samples = post_pred_dist(rng_key_post, **model_args_post)

#         for site in Y_u_sites:
#             if site in Y_u_post:
#                 Y_u_orig[site] = jnp.concatenate([Y_u_orig[site], fetch_all_u(i)[site]], axis=0)
#                 Y_u_post[site] = jnp.concatenate([Y_u_post[site], post_samples[site]], axis=1)
#             else:
#                 Y_u_orig[site] = fetch_all_u(i)[site]
#                 Y_u_post[site] = post_samples[site]
     
#     f_mae = lambda x : numpy.mean(numpy.abs(x), axis=(0,2))
#     mae = { site : f_mae(Y_u_orig[site] - Y_u_post[site].mean(axis=0)) for site in Y_u_sites }

#     # save to file
#     for site in Y_u_sites:
#         numpy.save(os.path.join(inout.RESULTS_DIR, f"original_{site}.npy"), Y_u_orig[site])
#         numpy.save(os.path.join(inout.RESULTS_DIR, f"post_samples_{site}.npy"), Y_u_post[site])
        
#     return mae



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


def save_posterior_summaries(mcmc_samples, output_dir="summaries"):
    """
    For each parameter in mcmc_samples, compute basic summaries across the
    first axis (assumed to be the sample dimension) and save to CSV.

    The CSV columns:
        mean, std, median, 5.0%, 95.0%

    If a parameter has shape (num_samples, d1, d2, ...),
    each flattened element (d1, d2, ...) gets its own row in the CSV,
    using a multi-index that records the sub-dimension index.
    """
    os.makedirs(output_dir, exist_ok=True)

    for param_name, values in mcmc_samples.items():
        # values shape: (num_samples, d1, d2, ...)
        if values.ndim < 1:
            # Shouldn't happen in typical MCMC samples, but just in case
            continue

        # Convert to numpy (if it's JAX array) so we can use np.ndindex, etc.
        values_np = np.asarray(values)

        # Flatten the sample dimension vs. the rest
        num_samples = values_np.shape[0]
        trailing_shape = values_np.shape[1:]  # (d1, d2, ...)

        # We'll create a list of all trailing indices, e.g. (i, j, ...)
        all_indices = list(np.ndindex(trailing_shape))  # e.g. [(0,0), (0,1), ..., (d1-1, d2-1)]

        # Reshape to (num_samples, -1) so each element across trailing dims is a column
        flattened = values_np.reshape(num_samples, -1)  # shape: (num_samples, d1*d2*...)

        # Compute summary stats for each "column" (each param element)
        means = np.mean(flattened, axis=0)
        sds = np.std(flattened, axis=0)
        medians = np.median(flattened, axis=0)
        p5 = np.percentile(flattened, 5, axis=0)
        p95 = np.percentile(flattened, 95, axis=0)

        # Build a DataFrame: one row per element in trailing dims
        df = pd.DataFrame({
            "mean": means,
            "std": sds,
            "median": medians,
            "5.0%": p5,
            "95.0%": p95,
        })

        # If the parameter is scalar per sample, then trailing_shape=().
        # Otherwise, we can use a MultiIndex to indicate which element each row refers to.
        if len(trailing_shape) > 0:
            # Create a multi-index from all_indices
            df.index = pd.MultiIndex.from_tuples(
                all_indices,
                names=[f"dim_{i}" for i in range(len(trailing_shape))]
            )
        else:
            # No trailing dims, just name the index
            df.index.name = "element_idx"

        # Save to CSV
        out_path = os.path.join(output_dir, f"{param_name.replace('/', '_')}.csv")
        df.to_csv(out_path)
        print(f"Saved summary for '{param_name}' -> {out_path}")

    print("All parameter summaries saved.")





def plot_parameter_estimates(true_params, posterior_means, out_folder="parameter_sim_plots"):
    """
    For each common key in true_params and posterior_means, create a scatter plot of:
        x = true_params[key]
        y = posterior_means[key]
    against the 45-degree line. Save plots to out_folder.

    Parameters
    ----------
    true_params : dict
        e.g. {param_name: 1D ndarray of "true" parameter values}
    posterior_means : dict
        e.g. {param_name: 1D ndarray of posterior means of the same shape}
    out_folder : str
        Folder where all plots are saved.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Find the intersection of keys
    common_keys = set(true_params.keys()).intersection(set(posterior_means.keys()))
    if not common_keys:
        print("No common parameter names found between true_params and posterior_means.")
        return

    for key in sorted(common_keys):
        # x => true values, y => posterior means
        x = np.asarray(true_params[key])
        y = np.asarray(posterior_means[key])

        if x.shape != y.shape:
            print(f"Warning: shapes differ for key '{key}' (true={x.shape}, post={y.shape}). Skipping.")
            continue

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, alpha=0.7, edgecolors='k', s=40)
        plt.title(f"Parameter: {key}\nPosterior Mean vs. True")

        # 45-degree line (identity)
        min_xy = min(x.min(), y.min())
        max_xy = max(x.max(), y.max())
        plt.plot([min_xy, max_xy], [min_xy, max_xy], '--r', label='45-degree line')

        plt.xlabel("True Values")
        plt.ylabel("Posterior Mean")
        plt.legend(loc='best')
        plt.tight_layout()

        # Save figure
        out_path = os.path.join(out_folder, f"{key}_true_vs_posterior_mean.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved {out_path}")



H_CUTOFFS = {"11" : 10, "10": 9, "5" : 4}

def main():

    parser = get_parser()  # Get the parser
    args = parser.parse_args()  # Parse the arguments

    #initialization
    npr.enable_x64()
    npr.enable_validation()
    rng_key = random.PRNGKey(args.seed)

    # target_model = model.model_svi
    if args.method == 'svi':
        target_model = model.model_svi
    if args.method == 'mcmc':
        target_model = model.model_mcmc
    
    target_guide = infer.autoguide.AutoDiagonalNormal(model=target_model,
                                               init_loc_fn=infer.init_to_feasible())

    # load data
    rng_key, rng_etl = random.split(rng_key, 2)
    (   J_c, J_u, J_u_dict, J_u_idx_start, J_u_idx_end, Q, T,
        Y_q_1, Y_q_2, Y_q_3, batch_num_train, batch_num_test,
        fetch_train, fetch_test, fetch_all_u, fetch_all_c
    ) = inout.load_dataset(rng_key=rng_etl,
                           batch_size=args.batch_size, 
                           N_split=args.train_test)
    
    static_kwargs = {   'Y_q_1' : Y_q_1, 
                        'Y_q_2' : Y_q_2, 
                        'Y_q_3' : Y_q_3,
                        'J_u_dict' : J_u_dict, 
                        'J_u_idx_start' : J_u_idx_start, 
                        'J_u_idx_end' : J_u_idx_end,
                        'J_c' : J_c, 'J_u' : J_u, 'Q' : Q, 'T' : T,  
                        'L' : args.latent_dims, 
                        'hidden_dim' : args.hidden_dims,
                        'scale_term' : 1.0 / batch_num_train    }      

    # 1) Create "mcmc" subfolder if not already present
    mcmc_folder = os.path.join('results', 'mcmc')
    os.makedirs(mcmc_folder, exist_ok=True)
    # Construct a path for your CSV file
    csv_path = os.path.join(mcmc_folder, 'mcmc_diagnostics.csv')

    # Path to the pickle file you created in main_hmc.py
    pkl_path = os.path.join(mcmc_folder, 'mcmc_samples.pkl')  

    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)

    print("Loaded MCMC samples:", samples.keys())

    # remove the first dimension (chain)
    samples2 = remove_first_dim(samples)
    
    # save posterior summaries
    save_posterior_summaries(samples2)
    
    # Load the "true" parameters
    sim_data_folder = "simulated_data"
    true_params, true_params_original_shape = util.load_true_params(sim_data_folder)
    print("True parameters, original shape:")
    for k, v in true_params_original_shape.items():
        print(f"{k}: shape={v.shape}")
    print("True parameters loaded and flattened:")
    for k, v in true_params.items():
        print(f"{k}: shape={v.shape}")

    #    Suppose you already have a `samples` dictionary from Numpyro:
    #    e.g. samples['alpha_1'].shape = (1000,) or (1000, 10, 5) etc.
    #    We'll flatten them except the first dimension:
    # samples2 = ...  # from your MCMC or loaded from disk
    flattened_samples = util.flatten_posterior_samples(samples2)
    print("Posterior samples, first dimension kept, rest flattened:")
    for k, v in flattened_samples.items():
        print(f"{k}: shape={v.shape}")

    # Then you can do something like:
    posterior_means = {k: v.mean(axis=0) for k, v in flattened_samples.items()}
    # Compare posterior_means[k] vs. true_params[k] (for matching shapes, etc.)
    plot_parameter_estimates(true_params, posterior_means, out_folder="parameter_sim_plots")


    # ----------- 6) Optional Posterior Summaries -----------
    # summary = npr.diagnostics.summary(samples)
    # print("MCMC Summary:\n", summary)

    # Suppose we want to run the posterior predictive on N_batch of data:
    rng_key, rng_key_ppc = random.split(rng_key, 2)
    post.display_parameter_shapes(samples)
    mae = post.reconstruct_mcmc(
        rng_key_ppc,
        model=target_model,
        mcmc_samples=samples,
        fetch_all_u=fetch_all_u,
        fetch_all_c=fetch_all_c,
        static_kwargs=static_kwargs,
        N_batch=args.batch_post,
        args=args
    )
    print("MAE from MCMC-based posterior predictive:", mae)


#########
## run ##
#########


if __name__ == '__main__':
    main()

