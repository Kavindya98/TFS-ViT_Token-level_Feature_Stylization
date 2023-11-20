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
import torch.nn as nn
import copy

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.lib.query import Q
from timm.models import create_model
from domainbed.networks import ResNet 
import torchvision.models

from timm.models import create_model

import os


def load_algorithm(file_path):

    d = torch.load(file_path)
    args = d["args"]
    print('Saved Args:')
    for k in args:
        print('\t{}: {}'.format(k, args[k]))

    hparams=d["model_hparams"]
    hparams["data_augmentation"] = False
    hparams['empty_head']=False
    hparams["eval"]=True
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
        if "rand_conv" in i:
            del d["model_dict"][i]
    for j in model.state_dict():
        if "rand_conv" in j:
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

def validation_accuracy(model, loader, weights, device, dataset, conversion_array=None):
    correct = 0
    total = 0
    model.eval()
    #model = nn.DataParallel(model,device_ids=[0])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            #y = y.to(device)
            p = model.predict(x)
            #if algorithm == None:
                #p = model.predict(x)
            #else:    
                #p = model(x)
            
            _, pred = p.topk(1, 1, True, True)
            pred = pred.cuda().detach()[:, 0]
            pred_list = list(pred.cpu().numpy())
            pred = torch.LongTensor([conversion_array[str(x)] for x in pred_list])
            correct += (pred==y).sum().item()
            
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)

            #batch_weights = batch_weights.to(device)
            # # print(p.shape)
            # if len(p.shape)==1:
            #     p = p.reshape(1,-1)
            # if p.size(1) == 1:
               
            #     # if p.size(1) == 1:
            #     correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            # else:
            #     # print('p hai ye', p.size(1))

            #     if dataset == "ImageNet_9" or dataset == "Cue_conflicts":
            #         correct += (imageNet_9_conversion(p.argmax(1), conversion_array).eq(y).float() * batch_weights).sum().item()
            #     else:
            #         correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            
            
    return correct / total   

def imageNet_9_conversion(output, conversion_array):
    result = output
    for j in range(len(output)):
        result[j]=conversion_array[str(output[j].item())]
    return result

def get_algorithm(algorithm):

    hparams = dict()
    hparams['data_augmentation']=False
    hparams['resnet18']=False
    hparams['empty_fc']=False
    hparams['resnet_dropout']=0 
    hparams['empty_head']=False
    hparams['fixed_featurizer']=False
    hparams["lr"]=0
    hparams['weight_decay']=0

    algo = None
    if algorithm == "ResNet50":
        algo = algorithms.get_algorithm_class('ERM')
    elif algorithm == "ResNet18":
        hparams['resnet18'] =True
        algo = algorithms.get_algorithm_class('ERM')
    elif algorithm == "DeiTBase":
        hparams['backbone'] = "DeiTBase"
        algo = algorithms.get_algorithm_class('ERM_ViT')
    elif algorithm == "ViTBase":
        hparams['backbone'] = "ViTBase"
        algo = algorithms.get_algorithm_class('ERM_ViT')
    else:
        print("Not defined backbone")

    model = algo(
        input_shape=(3, 224, 224,), 
        num_classes=1000, 
        num_domains=None, 
        hparams=hparams
    )

    return model, hparams

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--saved_model', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda:2")
    parser.add_argument('--saved_model_evaluvator', action='store_true')
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print('device:', device)
    print ('Current cuda device ', torch.cuda.current_device())

    if args.saved_model_evaluvator:
        model, hparams = get_algorithm(args.algorithm)
    else:
        model, model_args, hparams, model_data = load_algorithm(args.saved_model)
    # hparams = dict()
    # hparams['data_augmentation']=False
    # hparams['resnet18']=False
    # hparams['empty_fc']=False
    # hparams['resnet_dropout']=0
    
    eval_loaders, eval_weights, eval_loader_names = load_eval_dataset(args.dataset, args.data_dir, hparams)
    evals = zip(eval_loader_names, eval_loaders, eval_weights)

    
    # model = ResNet((3, 224, 224,),hparams)
    # model = torchvision.models.resnet50(pretrained=False)
    # #model = create_model('resnet50',drop_path_rate=0)
    # chk=torch.load("/home/kavindya/data/Model/deit/Results/Resnet50_1/best_checkpoint.pth", map_location='cpu')
    # model.load_state_dict(chk['model'])
    model.to(device)

    results = dict()

    conversion_array = {}
    if args.dataset == "ImageNet_9":
        with open('in_to_in9.json', 'r') as f:
            conversion_array.update(json.load(f))
    elif args.dataset == "Cue_conflicts":
        with open('cue_conflicts.json', 'r') as f:
            conversion_array.update(json.load(f))
    


    for name, loader, weights in evals:
        acc = validation_accuracy(model, loader, weights, device, args.dataset, conversion_array)
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


    