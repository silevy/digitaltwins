import argparse
import time
import pickle
import os
import datetime
import matplotlib.pyplot as plt

import jax
import optax
import numpyro as npr
import numpyro.infer as infer
from jax import (jit, lax, random)

from . import model, inout, post, optim

####################
# GLOBAL VARIABLES #
####################

# keys = number of question scale points
# values = number of cutoffs (one less than scale points)
H_CUTOFFS = {"11" : 10, "10": 9, "5" : 4}

parser = argparse.ArgumentParser(description='parse args')
# parser.add_argument('--is-predictive', default=False, type=bool)
parser.add_argument('--is-predictive', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--num-epochs', default=2001, type=int)
parser.add_argument('--epoch-save', default=500, type=int)
parser.add_argument('--batch-post', default=5, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-test', default=1024, type=int)
parser.add_argument('--num-flows', default=6, type=int)
parser.add_argument('--latent-dims', default=50, type=int)
parser.add_argument('--hidden-dims', default=512, type=int)
parser.add_argument('--learning-rate', default=1e-5, type=float)
parser.add_argument('--decay-rate', default=0.1, type=float)
args = parser.parse_args()

#######
# RUN #
#######
def main():
    #initialization
    npr.enable_x64()
    npr.enable_validation()
    rng_key = random.PRNGKey(args.seed)

    # target_model = model.model_full
    target_model = model.model_full
    target_guide = infer.autoguide.AutoDiagonalNormal(model=target_model,
                                               init_loc_fn=infer.init_to_feasible())
    # target_guide = infer.autoguide.AutoDelta(model.model_full)
    # target_guide = infer.autoguide.AutoIAFNormal(model=model.model_full,
    #                                       num_flows=args.num_flows,
    #                                       hidden_dims=args.hidden_dims,
    #                                       init_loc_fn=infer.init_to_feasible())
    
    if args.is_predictive:
        # jax.config.update("jax_default_device", jax.devices("cpu")[0])
        predictive(rng_key, target_model, target_guide, args)
    else:
        estimate(rng_key, target_model, target_guide, args)

def predictive(rng_key, target_model, target_guide, args):
    rng_key, rng_predictive = random.split(rng_key, 2)
    # # step 1 - run grid search
    optim.post_grid_search(rng_predictive, args, target_model, target_guide, pkl_file_name = "param.pkl", is_digital_twins = True, digital_twins_k_idx = 1)
    optim.post_grid_search(rng_predictive, args, target_model, target_guide, pkl_file_name = "param.pkl", is_digital_twins = True, digital_twins_k_idx = 2)
    optim.post_grid_search(rng_predictive, args, target_model, target_guide, pkl_file_name = "param.pkl", is_digital_twins = True, digital_twins_k_idx = 3)
    # step 2 - generate posterior samples on alpha, beta, phi
    post.post_latent_sites(rng_predictive, args, target_model, target_guide)
    # step 3 - generate posterior predictive samples on Y_u
    post.post_Y_predictive(rng_predictive, args, target_model, target_guide)



def estimate(rng_key, target_model, target_guide, args):
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

    
#########
## run ##
#########

if __name__ == '__main__':
    main()