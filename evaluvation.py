import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import copy

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.lib.query import Q

import os


def load_algorithm(file_path):

    d = torch.load(file_path)
    args = d["args"]
    print('Saved Args:')
    for k in args:
        print('\t{}: {}'.format(k, args[k]))

    hparams=d["model_hparams"]
    hparams["data_augmentation"] = False
    print('Saved HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
 
    algo = algorithms.get_algorithm_class(d["args"]["algorithm"])
    model = algo(
        input_shape=d["model_input_shape"], 
        num_classes=d["model_num_classes"], 
        num_domains=d["model_num_domains"], 
        hparams=d["model_hparams"]
    )
    for i in d["model_dict"]:
        if i in "rand_conv":
            del d["model_dict"][i]
    for j in model.state_dict():
        if j in "rand_conv":
            del model.state_dict()[j]
    model.load_state_dict(d["model_dict"])
    model.eval()

    return model, args, hparams, d

def load_eval_dataset(dataset_name, data_dir, hparms):

    test_envs = list(range(datasets.num_environments(dataset_name)))
    if dataset_name in vars(datasets):
        dataset = vars(datasets)[dataset_name](data_dir,
                                               test_envs, hparams)
    else:
        raise NotImplementedError
    print("Loading dataset")
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for _, env in tqdm(enumerate(dataset))]
    eval_weights = [None for _, weights in enumerate(dataset)]
    eval_loader_names = ['env_{}'.format(i)
                         for i in dataset.ENVIRONMENTS]
    return eval_loaders, eval_weights, eval_loader_names

def validation_accuracy(model, loader, weights, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = model.predict(x)
            
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:
               
                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                # print('p hai ye', p.size(1))
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            break
    return correct / total        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="PACS")
    parser.add_argument('--output_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--saved_model', type=str, default="")
    parser.add_argument('--saved_model_evaluvator', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('device:', device)
    print ('Current cuda device ', torch.cuda.current_device())

    model, model_args, hparams, model_data = load_algorithm(args.saved_model)
    model.to(device)
    
    eval_loaders, eval_weights, eval_loader_names = load_eval_dataset(args.dataset, args.data_dir, hparams)
    evals = zip(eval_loader_names, eval_loaders, eval_weights)

    results = dict()

    for name, loader, weights in evals:
        acc = validation_accuracy(model, loader, weights, device)
        print(name + '_acc',acc)
        results[name + '_acc'] = acc
    
    results_keys = sorted(results.keys())

    results.update({
                'hparams': hparams,
                # 'args': vars(model_args)
            })

    epochs_path = os.path.join(args.output_dir, 'results.jsonl')
    with open(epochs_path, 'a') as f:
        f.write(json.dumps(results, sort_keys=True) + "\n")


    