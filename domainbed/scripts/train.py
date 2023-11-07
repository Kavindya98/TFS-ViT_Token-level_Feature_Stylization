# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="ViT_RB_small")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--holdout_fraction', type=float, default=0.001)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    
    args = parser.parse_args()
    args.save_best_model = True


    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        js = json.loads(args.hparams)
        js["test_env"] = args.test_envs
        # print(args.hparams)
        hparams.update(js)

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print('device:', device)
    print ('Current cuda device ', torch.cuda.current_device())

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    ### DEBUGGING    
    #     print(dataset)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selection method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discarded at training.

    in_splits = []
    out_splits = []
    uda_splits = []
    assigned = False
    for env_i, env in enumerate(dataset):  # env is a domain
        uda = []
        if hparams['custom_train_val'] and not (hparams['custom_val'] == env_i or hparams['custom_train'] == env_i):
            out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))
            
        elif  hparams['custom_train_val'] and (hparams['custom_val'] == env_i or hparams['custom_train'] == env_i):
            if not assigned:
                for env_j, env_ in enumerate(dataset):
                
                    if  hparams['custom_val'] == env_j:
                        out = env_
                    elif  hparams['custom_train'] == env_j:
                        in_= env_
                assigned = True
                print(len(out)," length of val ",len(in_)," length of train") 
            else:
                continue  
        else:
            out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))
            


        # if not assigned and hparams['custom_train_val']:
        #     for env_j, env_ in enumerate(dataset):
        #         print("in loop env no",env_j,"len env",len(env_))
        #         if (not env_j in args.test_envs) and hparams['custom_val'] == env_j:
        #             out = env_
        #         elif (not env_j in args.test_envs) and hparams['custom_train'] == env_j:
        #             in_= env_
        #     assigned = True
        #     print("train val captured ",env_i)
            
        # elif assigned and hparams['custom_train_val']:
        #     if not (hparams['custom_val'] == env_i or hparams['custom_train'] == env_i):
        #         print("test captured ",env_i)
        #         out, in_ = misc.split_dataset(env,
        #                               int(len(env) * args.holdout_fraction),
        #                               misc.seed_hash(args.trial_seed, env_i))
        #     else:
        #         print("train val captured after assign",env_i)
        #         continue    
        # else:
        #     out, in_ = misc.split_dataset(env,
        #                               int(len(env) * args.holdout_fraction),
        #                               misc.seed_hash(args.trial_seed, env_i))
        # print("came out ",env_i)
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    envs_d = datasets.get_dataset_class(args.dataset).ENVIRONMENTS
    for i in range(len(in_splits)):
        print("env ",envs_d[i]," in ",len(in_splits[i][0])," out ",len(out_splits[i][0]))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    print(f"+ checkpoint_freq: {checkpoint_freq}")


    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    def save_checkpoint_best(filename, algo):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algo.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))



    last_results_keys = None
    best_val_acc = 0
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                          for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        # print("Training done")
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            temp_acc = 0
            temp_count = 0
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                if args.save_best_model:
                    if int(name[3]) not in args.test_envs and "out" in name:
                        temp_acc += acc
                        temp_count += 1
                results[name + '_acc'] = acc
            # print("Validation done")    
            if args.save_best_model:
                val_acc = temp_acc / (temp_count * 1.0)
                if val_acc >= best_val_acc:
                    # model_save = algorithm.detach().clone()  # clone
                    # model_save = copy.deepcopy(algorithm)  # clone
                    if (args.save_best_model):
                        save_checkpoint('IID_best.pkl')
                        algorithm.to(device)
                    best_val_acc = val_acc
                    print("Best model upto now")
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            # records = []
            # with open(epochs_path, 'r') as f:
            #     for line in f:
            #         records.append(json.loads(line[:-1]))
            # records = Q(records)
            # scores = records.map(model_selection.IIDAccuracySelectionMethod._step_acc)
            # if scores[-1] == scores.argmax('val_acc'):
            #     save_checkpoint('IID_best.pkl')
            #     algorithm.to(device)

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
        # print("One iteration done")

    save_checkpoint('model.pkl')
    # if (args.save_best_model):
    #     save_checkpoint_best('IID_best.pkl', model_save)
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
