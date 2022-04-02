import argparse

from config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='NeuMF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='WSDream', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    # configurations initialization
    config = Config(model=args.model, dataset=args.dataset)


