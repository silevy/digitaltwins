import argparse
from .args import get_parser  # Import the parser logic
import time
import pickle
import os
import csv
import datetime
import matplotlib.pyplot as plt

import jax
import optax
import numpyro as npr
import numpyro.infer as infer
from jax import (jit, lax, random)
import jax.numpy as jnp
from numpyro.diagnostics import summary

from . import model, inout, post, optim

# keys = number of question scale points
# values = number of cutoffs (one less than scale points)
H_CUTOFFS = {"11" : 10, "10": 9, "5" : 4}

#######
# RUN #
#######
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
        # target_model = model.model_mcmc
    if args.method == 'mcmc':
        target_model = model.model_mcmc
    
    target_guide = infer.autoguide.AutoDiagonalNormal(model=target_model,
                                               init_loc_fn=infer.init_to_feasible())
    # target_guide = infer.autoguide.AutoDelta(model.model_svi)
    # target_guide = infer.autoguide.AutoIAFNormal(model=model.model_svi,
    #                                       num_flows=args.num_flows,
    #                                       hidden_dims=args.hidden_dims,
    #                                       init_loc_fn=infer.init_to_feasible())
    
    if args.is_predictive:
        # jax.config.update("jax_default_device", jax.devices("cpu")[0])
        predictive(rng_key, target_model, target_guide, args)
    else:
        if args.method == 'svi':
            estimate_svi(rng_key, target_model, target_guide, args)
        if args.method == 'mcmc':
            estimate_mcmc(rng_key, target_model, args)

def predictive(rng_key, target_model, target_guide, args):
    rng_key, rng_predictive = random.split(rng_key, 2)
    # # step 1 - run grid search
    # optim.post_grid_search(rng_predictive, args, target_model, target_guide, pkl_file_name = "param.pkl", is_digital_twins = True, digital_twins_k_idx = 1)
    # optim.post_grid_search(rng_predictive, args, target_model, target_guide, pkl_file_name = "param.pkl", is_digital_twins = True, digital_twins_k_idx = 2)
    # optim.post_grid_search(rng_predictive, args, target_model, target_guide, pkl_file_name = "param.pkl", is_digital_twins = True, digital_twins_k_idx = 3)
    # step 2 - generate posterior samples on alpha, beta, phi
    # post.post_latent_sites(rng_predictive, args, target_model, target_guide)
    # step 3 - generate posterior predictive samples on Y_u
    post.post_Y_predictive(rng_predictive, args, target_model, target_guide)



def estimate_svi(rng_key, target_model, target_guide, args):
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
    
    # instantiations    
    optimizer = optax.adam(optax.exponential_decay(init_value=args.learning_rate,
                                                   transition_steps=10000,
                                                   decay_rate=args.decay_rate))

    # optimizer = optax.adam(args.learning_rate)

    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adam(optax.exponential_decay(
    #         init_value=args.learning_rate,
    #         transition_steps=args.num_epochs,
    #         decay_rate=args.decay_rate
    #     ))
    # )


    svi = infer.SVI(model=target_model,
                    guide=target_guide,
                    optim=optimizer,
                    loss=infer.Trace_ELBO(num_particles=1),
                    **static_kwargs)

    # initialization
    rng_key, rng_key_rnd, rng_key_init = random.split(rng_key, 3)
    idx_init = random.randint(rng_key_rnd, (1,), minval=0, maxval=batch_num_train)[0]
    svi_state = svi.init(rng_key_init, *fetch_train(idx_init))
    
    # run
    @jit
    def epoch_train(svi_state):
        def body_fn(i, val):
            loss_sum, svi_state = val
            batch = fetch_train(i)
            svi_state, loss = svi.update(svi_state, *batch)
            loss_sum += loss
            return loss_sum, svi_state

        return lax.fori_loop(0, batch_num_train, body_fn, init_val=(0.0, svi_state))
    
    @jit
    def epoch_test(svi_state):
        def body_fn(i, val):
            loss_sum = val
            batch = fetch_test(i)
            loss = svi.evaluate(svi_state, *batch)
            loss_sum += loss
            return loss_sum

        return lax.fori_loop(0, batch_num_test, body_fn, init_val=(0.0))

    # add two lists to store loss history
    train_losses = []
    test_losses = []

    # Get current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

    # Construct file name
    file_name_epochs = f"epochs_model_{args.num_epochs}_{args.batch_size}_{date_time}.txt"
    file_name_mae = f"mae_model_{args.num_epochs}_{args.batch_size}_{date_time}.txt"

    # Check if directory exists, if not, create it
    if not os.path.exists('fit_results'):
        os.makedirs('fit_results')

    # Open the file in append mode
    f_epochs = open(os.path.join('fit_results', file_name_epochs), 'a')
    f_mae = open(os.path.join('fit_results', file_name_mae), 'a')
 
    t_start = time.time()

    total_train_number_customers =  ( 3 * (5000 - args.train_test) * 10 ) 
    total_test_number_customers = ( 3 * args.train_test * 10 ) 

    for i in range(args.num_epochs):
        loss_train, svi_state = epoch_train(svi_state)
        loss_test = epoch_test(svi_state)

        # normalize losses by the number of batches in each set
        norm_loss_train = loss_train  / total_train_number_customers
        norm_loss_test = loss_test / total_test_number_customers

        # store normalized losses
        train_losses.append(norm_loss_train)
        test_losses.append(norm_loss_test)

        print(
            "Epoch {} ({:.2f} s.): train loss = {:.2f} | test loss = {:.2f} ".format(
                i, time.time() - t_start, norm_loss_train, norm_loss_test
            )
        )
        epoch_info = f"Epoch {i} ({time.time() - t_start} s.): train loss = {norm_loss_train} | test loss = {norm_loss_test} \n"
        # print(epoch_info)
        f_epochs.write(epoch_info)
        
        # reconstruction
        if (i > 0) and (i % args.epoch_save == 0):
            rng_key, rng_key_recon = random.split(rng_key, 2)
            params = svi.get_params(svi_state)  
            mae = post.reconstruct(rng_key_recon, target_model, target_guide, params, 
                                   fetch_all_u, fetch_all_c, args.batch_post, static_kwargs)

            print("MAE:")
            mae_info = f"Epoch {i} ({time.time() - t_start} s.) \n"
            f_mae.write(mae_info)
            for k, v in mae.items():
                print(f"{k} : {v}")
                f_mae.write(f"{k} : {v}\n")

            # for k, v in mae.items():
                # print("{} : {:.2f}".format(k, v))
                
    # At the end, close the files
    f_epochs.close()
    f_mae.close()
    
    rng_key, rng_key_recon_all = random.split(rng_key, 2)

    post.reconstruct_all(rng_key_recon_all, target_model, target_guide, params, 
                                   fetch_all_u, fetch_all_c, batch_num_train, batch_num_test, static_kwargs)

    # Plotting the loss history
    plottitle = f"model_{args.num_epochs}_{args.batch_size}_{date_time}_losshistory.pdf"
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Training Loss", zorder=2, alpha=0.8)  # higher z-order and semi-transparent
    plt.plot(test_losses, label="Test Loss", zorder=1, alpha=0.5)  # lower z-order and more transparent
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Average Loss history, for Training and Test Sets")
    plt.grid(True)
    plt.savefig(os.path.join('fit_results', plottitle))  # Save the figure as PDF
    plt.close()

    # #print
    param_post = svi.get_params(svi_state)
    print("AutoGuide - locations: ")
    print(param_post['auto_loc'])
    print("AutoGuide - scales: ")
    print(param_post['auto_scale'])

    # I/O, parameters
    with open(os.path.join(inout.RESULTS_DIR, f"param.pkl"), 'wb') as f:
        pickle.dump(param_post, f)


def estimate_mcmc(rng_key, target_model, args):

    """Runs the Digital Twins model via MCMC (HMC/NUTS)."""
    # Enable double precision and turn on validation
    npr.enable_x64()
    npr.enable_validation()

    # Random seed
    rng_key = random.PRNGKey(args.seed)

    # ----------- 1) Load Data -----------
    rng_key, rng_data = random.split(rng_key, 2)
    (   J_c, J_u, J_u_dict, J_u_idx_start, J_u_idx_end, Q, T,
        Y_q_1, Y_q_2, Y_q_3,
        batch_num_train, batch_num_test,
        fetch_train, fetch_test,
        fetch_all_u, fetch_all_c
    ) = inout.load_dataset(rng_key=rng_data,
                           batch_size=args.batch_size,
                           N_split=args.train_test)

    # We only need a single “full” dataset to pass to NUTS,
    # but if data is large, you may want to subsample or map 
    # across mini-batches. For now, we'll pass everything at once.
    # Typically, you’d combine them the same way your model expects.

    # Example if the model expects:
    #   Y_u_1_11, Y_u_1_10, Y_u_1_5, ...
    # You can fetch the entire dataset with `fetch_all_u(0)`, etc.

    # data_train = {}
    data_u = {}
    data_c = {}
    # total number of batches is (batch_num_train + batch_num_test)

    for i in range(batch_num_train + batch_num_test):
        # 1) Grab data from batch i
        u_dict = fetch_all_u(i)  # e.g. {'Y_u_1_5': ..., 'Y_u_1_10': ...}
        c_dict = fetch_all_c(i)  # e.g. {'Y_c_1_static': ..., 'Y_c_1_optim': ...}

        # 2) Append each array to the corresponding list in all_u, all_c
        for k, v in u_dict.items():
            if k not in data_u:
                data_u[k] = []
            data_u[k].append(v)

        for k, v in c_dict.items():
            if k not in data_c:
                data_c[k] = []
            data_c[k].append(v)

    for k in data_u:
        data_u[k] = jnp.concatenate(data_u[k], axis=0)  # merges along N dimension

    for k in data_c:
        data_c[k] = jnp.concatenate(data_c[k], axis=0)

    data_u2 = fetch_all_u(0)
    data_c2 = fetch_all_c(0)

    # Combine all relevant arrays into a single dict
    # that we can pass to MCMC's .run(...) as **kwargs:
    model_kwargs = {
        **data_u2,
        **data_c2,
        'J_c': J_c,
        'J_u': J_u,
        'J_u_dict': J_u_dict,
        'J_u_idx_start': J_u_idx_start,
        'J_u_idx_end': J_u_idx_end,
        'Q': Q,
        'T': T,
        'L': 50,               # default latent_dims
        'hidden_dim': 512,     # default hidden_dims
        'scale_term': 1.0 / batch_num_train,
        # ... if the model expects Y_q_1, Y_q_2, Y_q_3:
        'Y_q_1': Y_q_1,
        'Y_q_2': Y_q_2,
        'Y_q_3': Y_q_3,
    }

    # ----------- 3) Define NUTS / MCMC -----------
    init_strategy=infer.initialization.init_to_feasible()
    nuts_kernel = infer.NUTS(target_model, init_strategy=init_strategy)
    
    mcmc = infer.MCMC(
        nuts_kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method="parallel"  # or "sequential" if GPU memory is low
    )

    # ----------- 4) Run MCMC -----------
    print(f"Starting NUTS with {args.num_warmup} warmup steps, "
          f"{args.num_samples} samples, {args.num_chains} chain(s).")
    t_start = time.time()
    mcmc.run(rng_key, **model_kwargs)
    t_end = time.time()
    print(f"MCMC completed in {t_end - t_start:.2f} seconds.")

    # ----------- 5) Extract and save samples -----------
    # samples = mcmc.get_samples(group_by_chain=False)
    # If you want chain-by-chain: 
    samples = mcmc.get_samples(group_by_chain=True)


    # Suppose you have something like:
    # mcmc.run(rng_key, **model_kwargs)
    # samples = mcmc.get_samples(group_by_chain=True)
    diagnostics = summary(samples)  
    # diagnostics is a dictionary: {param_name: {"mean": ..., "std": ..., "r_hat": ..., "n_eff": ...}, ...}


    # 1) Create "mcmc" subfolder if not already present
    mcmc_folder = os.path.join('results', 'mcmc')
    os.makedirs(mcmc_folder, exist_ok=True)


    # Construct a path for your CSV file
    csv_path = os.path.join(mcmc_folder, 'mcmc_diagnostics.csv')

    # We'll assume every parameter has the same keys, e.g., 'mean', 'std', 'n_eff', 'r_hat', etc.
    # Just gather them from the first parameter's dictionary (or define them explicitly).
    some_param = next(iter(diagnostics))  # get one param name
    fieldnames = ['param'] + list(diagnostics[some_param].keys())

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for param_name, stats in diagnostics.items():
            # stats is something like {'mean': 0.5, 'std': 0.2, 'n_eff': 950, 'r_hat': 1.01, ...}
            row = {'param': param_name}
            # Merge stats into the row
            row.update(stats)
            writer.writerow(row)

    print(f"Saved CSV diagnostics to {csv_path}")

    # Optionally, save to a file
    # 2) Save MCMC samples to "mcmc" subfolder
    pkl_path_save = os.path.join(mcmc_folder, args.mcmc_output)  # e.g., "mcmc_samples.pkl"
    with open(pkl_path_save, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved MCMC samples to {pkl_path_save}")

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


    # Path to the pickle file you created in main_hmc.py
    pkl_path = os.path.join(mcmc_folder, 'mcmc_samples.pkl')  

    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)

    print("Loaded MCMC samples:", samples.keys())

    # ----------- 6) Optional Posterior Summaries -----------
    # summary = npr.diagnostics.summary(samples)
    # print("MCMC Summary:\n", summary)

    # Suppose we want to run the posterior predictive on N_batch of data:
    rng_key, rng_key_ppc = random.split(rng_key, 2)
    mae = post.reconstruct_mcmc(
        rng_key_ppc,
        model=target_model,
        mcmc_samples=samples,
        fetch_all_u=fetch_all_u,
        fetch_all_c=fetch_all_c,
        static_kwargs=static_kwargs,
        N_batch=args.batch_post
    )
    print("MAE from MCMC-based posterior predictive:", mae)



#########
## run ##
#########

if __name__ == '__main__':
    main()