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
import torchvision.transforms as transforms
import torchvision.datasets as data
import torch.nn.functional as F

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
    hparams["eval"]=True
    # hparams['empty_head']=False
    hparams['unfreeze_train_bn']=False
    hparams['scheduler']=False
    hparams['nesterov']=False
    hparams["mean_std"] = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
    #d["args"]["algorithm"] ="AugMix_CNN"
    

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

def load_eval_dataset(dataset_name, data_dir, hparams):
    print(" Inside load_eval_dataset Normalization is set to ",hparams['normalization'])

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
            y = y.to(device)
            p = model.predict(x)
            #if algorithm == None:
                #p = model.predict(x)
            #else:    
                #p = model(x)
            
            # _, pred = p.topk(1, 1, True, True)
            # pred = pred.cuda().detach()[:, 0]
            # pred_list = list(pred.cpu().numpy())
            # pred = torch.LongTensor([conversion_array[str(x)] for x in pred_list])
            # correct += (pred==y).sum().item()
            
            # if weights is None:
            #     batch_weights = torch.ones(len(x))
            # else:
            #     batch_weights = weights[weights_offset: weights_offset + len(x)]
            #     weights_offset += len(x)

            # batch_weights = batch_weights.to(device)
            # print(p.shape)
            if len(p.shape)==1:
                p = p.reshape(1,-1)
            if p.size(1) == 1:
               
                # if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item() #* batch_weights.view(-1, 1)
            else:
                # print('p hai ye', p.size(1))

                if dataset == "ImageNet_9" or dataset == "Cue_conflicts":
                    correct += (imageNet_9_conversion(p.argmax(1), conversion_array).eq(y).float()).sum().item() #* batch_weights
                else:
                    correct += (p.argmax(1).eq(y).float()).sum().item() #* batch_weights
            total += torch.ones(len(x)).sum().item()
         
            
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
    hparams["eval"]=False
    hparams['unfreeze_train_bn']=False
    hparams['scheduler']=False
    hparams['nesterov']=False
    hparams['normalization']=True
    hparams['val_augmentation']=False


    algo = None
    if algorithm == "ResNet50":
        hparams['backbone']=None
        algo = algorithms.get_algorithm_class('ERM')
    elif algorithm == "ResNet18":
        hparams['resnet18'] =True
        algo = algorithms.get_algorithm_class('ERM')
    elif algorithm == "DeiTBase":
        hparams['backbone'] = "DeiTBase"
        algo = algorithms.get_algorithm_class('ERM_ViT')
    elif algorithm == "DeitSmall":
        hparams['backbone'] = "DeitSmall"
        algo = algorithms.get_algorithm_class('ERM_ViT')
    elif algorithm == "T2T14":
        hparams['backbone'] = "T2T14"
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

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net.predict(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  for corruption in CORRUPTIONS:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs) 

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
        os.makedirs(args.output_dir, exist_ok=True)

    #print('device:', device)
    #print ('Current cuda device ', torch.cuda.current_device())

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
        with open('/home/kavindya/data/Model/backgrounds_challenge/in_to_in9.json', 'r') as f:
            conversion_array.update(json.load(f))
    elif args.dataset == "Cue_conflicts":
        with open('cue_conflicts.json', 'r') as f:
            conversion_array.update(json.load(f))
    
    # test_transform = transforms.Compose(
    #   [transforms.ToTensor(),
    #    transforms.Normalize([0.5] * 3, [0.5] * 3)])

    # test_data = data.CIFAR10(
    #     '/media/SSD2/Dataset/cifar', train=False, transform=test_transform, download=True)
    # base_c_path = '/media/SSD2/Dataset/CIFAR-10-C/'

    # test_c_acc = test_c(model, test_data, base_c_path)
    # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    error =[]
    print("Algorithm ",args.algorithm)
    print("Dataset ",args.dataset)
    for name, loader, weights in evals:
        acc = validation_accuracy(model, loader, weights, device, args.dataset, conversion_array)
        #print()
        results[name + '_acc'] = round(acc,4)
        results[name + '_error'] = round(1-acc,4)
        error.append(round(1-acc,4))
        print(name + '_acc',round(acc,4),"  ",name + '_error',(round(1-acc,4)))

    print('Mean Corruption Error: {:.3f}'.format(100. * np.mean(error)))
    
    results_keys = sorted(results.keys())

    results.update({
                'hparams': hparams,
                # 'args': vars(model_args)
            })
    
    epochs_path = os.path.join(args.output_dir, 'results.jsonl')
    with open(epochs_path, 'a') as f:
        f.write(json.dumps(results, sort_keys=True) + "\n")


    