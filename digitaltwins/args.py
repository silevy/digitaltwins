import argparse

####################
# GLOBAL VARIABLES #
####################

def get_parser():
    parser = argparse.ArgumentParser(description='parse args')
    # parser.add_argument('--is-predictive', default=False, type=bool)
    parser.add_argument('--is-predictive', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--method', default='svi', choices=['svi', 'mcmc'], 
                        help='Select which inference method to use: svi or mcmc (default: svi)'
    )
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--num-epochs', default=5001, type=int)
    parser.add_argument('--epoch-save', default=1000, type=int)
    parser.add_argument('--batch-post', default=5, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--train-test', default=1024, type=int)
    parser.add_argument('--num-flows', default=6, type=int)
    parser.add_argument('--latent-dims', default=50, type=int)
    parser.add_argument('--hidden-dims', default=32, type=int)
    parser.add_argument('--learning-rate', default=1e-5, type=float)
    parser.add_argument('--decay-rate', default=0.95, type=float)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')

    # For MCMC:
    parser.add_argument('--num-warmup', default=200, type=int, help='Number of warm-up steps for NUTS')
    parser.add_argument('--num-samples', default=200, type=int, help='Number of MCMC samples')
    parser.add_argument('--num-chains', default=1, type=int, help='Number of chains (parallel or sequential)')
    parser.add_argument('--mcmc-output', default='mcmc_samples.pkl', type=str,
                        help='Filename to save MCMC samples (Pickle)')
    # For Simulation
    parser.add_argument('--cust-per-firm', default=5000, type=int,  help='Number of simulated customers per firm')

    return parser
