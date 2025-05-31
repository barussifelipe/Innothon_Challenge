import os
import argparse

from torch.backends import cudnn
from Anomaly_Transformer.utils.utils import *

from Anomaly_Transformer.solver import Solver


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

class Config():
    def __init__(self, lr: float = 1e-4,
           num_epochs: int = 10,
           k: int = 3,
           win_size: int = 10,
           input_c: int = 95,
           output_c: int = 95,
           batch_size: int = 1024,  
           pretrained_model: str = 20,
           dataset: str = 'Consumption',
           mode: str = 'test',
           data_path: str = r'C:\Users\Vanessa Castanho\Documents\Programming\Innothon - Challenge 2',
           model_save_path: str = 'Setup_Anomaly/src/Anomaly_Transformer/checkpoints',
           anormly_ratio: float = 4.00):
        
        self.lr = lr
        self.num_epochs = num_epochs
        self.k = k
        self.win_size = win_size
        self.input_c = input_c
        self.output_c = output_c
        self.batch_size = batch_size
        self.pretrained_model = pretrained_model
        self.dataset = dataset
        self.mode = mode
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.anormly_ratio = anormly_ratio


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--num_epochs', type=int, default=10)
    # parser.add_argument('--k', type=int, default=3)
    # parser.add_argument('--win_size', type=int, default=100)
    # parser.add_argument('--input_c', type=int, default=38)
    # parser.add_argument('--output_c', type=int, default=38)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--pretrained_model', type=str, default=None)
    # parser.add_argument('--dataset', type=str, default='credit')
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    # parser.add_argument('--model_save_path', type=str, default='checkpoints')
    # parser.add_argument('--anormly_ratio', type=float, default=4.00)

    # config = parser.parse_args()

    args = vars(Config())
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(Config())
