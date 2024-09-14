
import os
import configparser
from argparse import ArgumentParser

from src.train.train import train
from src.train.evaluate import test


def main(config_file, labels_dict):

    config = configparser.ConfigParser()
    config.read(config_file)

    args = parse_args(config)

    # Variables
    labels = labels_dict
    weights_dir = args.train_dir + '/logs/best_model.pt'

    # Create directories
    if not os.path.exists(args.proj_dir):
        os.mkdir(args.proj_dir)

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)

    train(weights_dir, labels, args)
    test(labels, weights_dir, args)


def parse_args(config):
    parser = ArgumentParser()

    # Directory args
    parser.add_argument("--proj_dir", type=str, default=config['DIR']['PROJ_DIR'])
    parser.add_argument("--train_dir", type=str, default=config['DIR']['TRAIN_DIR'])
    parser.add_argument("--test_dir", type=str, default=config['DIR']['TEST_DIR'])

    # Data args
    parser.add_argument("--train_data_dir", type=str, default=config['DIR']['TRAIN_DATA_DIR'])
    parser.add_argument("--test_data_dir", type=str, default=config['DIR']['TEST_DATA_DIR'])
    parser.add_argument("--train_data_mode", type=str, help="if to use generated/synthetic images or original to train. Either orig or gen",
    default=config['DIR']['TRAIN_DATA_MODE'])

    # Training args
    parser.add_argument("--epochs", type=int, default=config['VAR']['EPOCHS'])
    parser.add_argument("--batch_size", type=int, default=config['VAR']['BATCH_SIZE'])
    parser.add_argument("--init_lr", type=float, default=config['VAR']['INIT_LR'])
    parser.add_argument("--lr_drop", type=float, default=config['VAR']['LR_DROP'])
    parser.add_argument("--lr_patience", type=int, default=config['VAR']['LR_PATIENCE'])
    parser.add_argument("--early_stop_patience", type=int, default=config['VAR']['EARLY_STOP_PATIENCE'])

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    labels_dict = {'NC': 0, 'PD': 1} # Dictionary of classes you want to classify
    config_file = 'config.ini' # Add your .ini config file

    main(config_file, labels_dict)
