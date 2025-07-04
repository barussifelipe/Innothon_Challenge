import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=96) # MODIFIED
    parser.add_argument('--input_c', type=int, default=4)   # MODIFIED
    parser.add_argument('--output_c', type=int, default=4)  # MODIFIED
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='Enel') # MODIFIED
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Tests/Data/10_days.csv') # MODIFIED - ADJUST PATH
    parser.add_argument('--model_save_path', type=str, default='/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Tests/Save')
    parser.add_argument('--anomaly_ratio', type=float, default=4.00) # Typo in original: 'anormly' should be 'anomaly'

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)