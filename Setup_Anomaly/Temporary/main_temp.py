import os
import argparse

from torch.backends import cudnn
from Anomaly_Transformer.utils.utils import *

from Anomaly_Transformer.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config): # Removed test_labels from signature, as it's global
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)

    # When initializing solver, pass the full config vars, including supply_id
    solver = Solver(vars(config)) 

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        # Solver.test() now needs to be able to access config.supply_id
        # You already initialized solver with vars(config), so you can just call it
        # The test method will handle saving and returning the 0/1 prediction
        return solver.test() # Return the 0/1 prediction directly from test

    return None # Return None if not in test mode (or handle differently)

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
           anormly_ratio: float = 4.00,
           supply_id: str = None): # <--- ADD THIS LINE

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
        self.supply_id = supply_id # <--- AND THIS LINE


if __name__ == '__main__':
    # ... (initial args/config setup) ...

    test_predictions_per_supply = [] # This list will collect (supply_id, 0/1 prediction)

    datasets_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data'
    all_supply_files = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

    # Example to limit for testing (remove this for full run)
    # all_supply_files = ['SUPPLY009.csv'] 

    for dataset_file in all_supply_files:
        config = Config() # Create a fresh config for each supply
        supply_id = dataset_file.replace('.csv', '')

        # Update config parameters for the current supply
        config.data_path = os.path.join(datasets_path, dataset_file)
        config.supply_id = supply_id # <--- IMPORTANT: SET THE SUPPLY_ID IN CONFIG

        # --- Train Mode ---
        config.mode = 'train'
        config.pretrained_model = None # Ensure no pretrained model for training
        print(f"\n--- Training on supply: {supply_id} ---")
        main(config) # No return value needed for train mode

        # --- Test Mode ---
        config.mode = "test"
        # config.pretrained_model = 20 # This line is not needed. Solver loads checkpoint
        print(f"--- Testing on supply: {supply_id} and saving anomaly scores ---")

        # main(config) will now return the 0/1 prediction from solver.test()
        single_supply_prediction = main(config) 
        test_predictions_per_supply.append((supply_id, single_supply_prediction))

        print(f"Finished processing {supply_id}. Predicted as anomalous: {single_supply_prediction}")

    print("\n--- All Supplies Processed ---")
    print("Supply-level predictions (ID, Anomalous?):")
    for s_id, pred_val in test_predictions_per_supply:
        print(f"  {s_id}: {pred_val}")
