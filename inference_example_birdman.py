import argparse
import os
import os.path as op
import sys
sys.path.insert(0, os.path.abspath('../.')) #I might have to adjust this

import numpy as np
from scipy.optimize import curve_fit
import torch
from statsmodels.tsa.stattools import acf

from models.predictive_coding_ista import DynPredNet
import models.data_loader as data_loader
import utils
from evaluation import record

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/gpfs01/bartels/user/abeyer/my_dynamic_predictive_coding/data/birdman') #data/birdman might still be empty atm but will be filled 
parser.add_argument('--model_dir', default='/gpfs01/bartels/user/abeyer/my_dynamic_predictive_coding/experiments/two_forest') #I can keep this path as I use pre-trained weights

if __name__ == '__main__':

    args = parser.parse_args()
    # load data and model
    fpath = args.model_dir
    data_dir = args.data_dir
    params = utils.Params(op.join(fpath, 'params.json'))
    params.cuda = torch.cuda.is_available()
    # keep order the same
    params.shuffle = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DynPredNet(params, device).to(device)
    ckpt = utils.load_checkpoint(op.join(fpath, 'last.pth.tar'), model)
    model.eval()
    save_dir = "birdman_results" #Saving results directly to a new birdman_results directory
    if not op.exists(save_dir): #New directory is automatically created
        os.makedirs(save_dir)
    
    save_dir = "birdman_results/inf_example/"
    if not op.exists(save_dir):
        os.makedirs(save_dir)
        
    batch_size = params.batch_size
    dataloaders = data_loader.fetch_dataloader(args.data_dir, params)
    dl = dataloaders #Need to check dataloader!!!

    #X = dl.dataset.data[:params.batch_size].to(device) #Come back if problems arise

    # Process all 703 batches
    num_batches = 720  # Hardcoded for simplicity
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, dl.dataset.data.shape[0])
        X = dl.dataset.data[batch_start:batch_end].to(device)

        save_batch_dir = op.join(save_dir, f"batch_{batch_idx}") # Save results for each batch
        if not op.exists(save_batch_dir):
            os.makedirs(save_batch_dir)

        # inference example
        result_dict = record(model, X, input_dim=params.input_dim, mixture=True, turnoff=10)
        np.savez(op.join(save_batch_dir, f"result_dict_birdman.npz"), **result_dict)
    
        # long term prediction
        result_dict = record(model, X, input_dim=params.input_dim, mixture=True, turnoff=3)
        np.savez(op.join(save_batch_dir, f"result_dict_long_pred_birdman.npz"), **result_dict)

        print(f"Processed batch {batch_idx + 1}/{num_batches}")

    print("Inference complete. Results saved in:", save_dir)



