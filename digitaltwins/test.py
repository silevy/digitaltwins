import argparse

parser = argparse.ArgumentParser(description='Simulation script arguments')
parser.add_argument('--seed', default=2, type=int, help='Random seed')
parser.add_argument('--train-test', default=512, type=int, help='Train test size')
parser.add_argument('--latent-dims', default=50, type=int, help='Latent dimensions')
parser.add_argument('--hidden-dims', default=512, type=int, help='Hidden dimensions')
args = parser.parse_args()
    
def main():
    # Your existing main function code
    print(f"Simulation script started with args: {args}")

if __name__ == "__main__":
    main()
