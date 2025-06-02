import os
import argparse

from torch.backends import cudnn
from Anomaly_Transformer.utils.utils import *

from Anomaly_Transformer.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config, test_labels = None):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

#For each dataset, I want to train a model on it, then, I want to test and see if at least one point is given as anomalous, if so, we classify this dataset as anomalous itself. Then, I want to build a label that has these classifications to compare with the labels given. 

#However, the way that this code is implemented is to give the accurracy for each one of the datasets and not in the end, so I have to run both of the methods and then get the result of them, store on this list and then after, give the prediction result outside this method, in the main. 
#So on the test method I have to return the classification made, 0 or 1, and append. 

#Then I need to do a for loop that acess each of these data, puts then on train and then on test, append the result of the test to this list and calculate the metrics later. 

#Then we don't need the accuracy metrics on the test method. 
#We will implement these at the end. 

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        test_labels.append(solver.test())
       

    return solver

class Config():
    def __init__(self, lr: float = 1e-4,
           num_epochs: int = 10,
           k: int = 3,
           win_size: int = 10,
           input_c: int = 96,
           output_c: int = 96,
           batch_size: int = 128,  
           pretrained_model: str = None,
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


    # from sklearn.metrics import precision_recall_fscore_support
    # from sklearn.metrics import accuracy_score
    # accuracy = accuracy_score(gt, pred)
    # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
    #                                                                         average='binary')
    # print(
    #     "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
    #         accuracy, precision,
    #         recall, f_score))

    test_labels = []
    #We will now open the directory where the datasets are stored and run the model for each one of them, appending the result to the test_labels list.
    datasets_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data'
    for dataset_file in os.listdir(datasets_path):
        config = Config()
        if dataset_file.endswith('.csv') and dataset_file == 'SUPPLY034.csv':
            dataset_path = os.path.join(datasets_path, dataset_file)
            config.data_path = dataset_path
            config.mode = 'train'
            config.pretrained_model = None  # Assuming no pretrained model for testing
            print(f"Training on dataset: {dataset_file}")
            main(config)
            config.mode = "test"
            config.pretrained_model = 20
            main(config, test_labels)
            print(f"Testing on dataset: {dataset_file}, label: {test_labels[-1]}")
    print("Test labels:", test_labels)
