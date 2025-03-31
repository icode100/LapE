# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _thread
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from LapE import *
 
query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())  # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']


class Args:
    def __init__(self, **kwargs):
        # Basic settings
        self.cuda = kwargs.get('cuda', True)
        self.do_train = kwargs.get('do_train', True)
        self.do_valid = kwargs.get('do_valid', True)
        self.do_test = kwargs.get('do_test', True)

        # Data path and dataset parameters
        self.data_path = kwargs.get('data_path', '/path/to/data')
        self.negative_sample_size = kwargs.get('negative_sample_size', 128)
        self.nentity = kwargs.get('nentity', 0)  # Will be set later from stats.txt
        self.nrelation = kwargs.get('nrelation', 0)  # Will be set later from stats.txt

        # Embedding and model parameters
        self.hidden_dim = kwargs.get('hidden_dim', 800)
        self.gamma = kwargs.get('gamma', 60.0)  # Margin for ranking loss
        self.geo = kwargs.get('geo', 'gamma')   # For LapE, you can set this to 'laplace' or leave as is for compatibility

        # Batch and learning parameters
        self.batch_size = kwargs.get('batch_size', 512)
        self.test_batch_size = kwargs.get('test_batch_size', 4)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.cpu_num = kwargs.get('cpu_num', 3)

        # Checkpoint and save settings
        self.save_path = kwargs.get('save_path', './checkpoints')
        self.max_steps = kwargs.get('max_steps', 450001)
        self.warm_up_steps = kwargs.get('warm_up_steps', None)
        self.drop = kwargs.get('drop', 0.1)
        self.save_checkpoint_steps = kwargs.get('save_checkpoint_steps', 50000)
        self.valid_steps = kwargs.get('valid_steps', 30000)
        self.log_steps = kwargs.get('log_steps', 100)
        self.test_log_steps = kwargs.get('test_log_steps', 1000)

        # Query and task settings
        self.tasks = kwargs.get('tasks', '1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up')
        self.evaluate_union = kwargs.get('evaluate_union', "DNF")
        self.print_on_screen = kwargs.get('print_on_screen', True)
        
        # Seed and mode settings
        self.seed = kwargs.get('seed', 42)
        self.beta_mode = kwargs.get('beta_mode', "(1600,2)")
        self.gamma_mode = kwargs.get('gamma_mode', "(1600,4)")
        self.box_mode = kwargs.get('box_mode', "(none,0.02)")
        self.prefix = kwargs.get('prefix', None)
        self.checkpoint_path = kwargs.get('checkpoint_path', None)
        
def save_model(model, optimizer, save_variable_list, args):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate.
    (Works for LapE model)
    """
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(args.save_path, 'checkpoint'))


def set_logger(args):
    """
    Configure logging to output to both console and a log file.
    (Works for LapE model)
    """
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs.
    (This function is generic and works for LapE as well.)
    """
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in the dataloader for the LapE model.
    This function computes metrics (MRR, Hits@K, etc.) by calling the model's test_step,
    and then logs and aggregates the results.
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]),
                              metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)

    return all_metrics

def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks.
    This function is independent of the embedding distribution (Gamma, Laplace, etc.)
    and can be used as-is for the LapE model.
    '''
    logging.info("loading data")
    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))

    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers

def main(args):
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    # For union evaluation, our LapE model uses the same scheme as GammaE
    if args.evaluate_union == 'DM':
        assert args.geo == 'gamma', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    prefix = args.prefix if args.prefix is not None else 'logs'

    print("overwriting args.save_path")
    args.save_path = os.path.join(prefix, os.path.basename(args.data_path), args.tasks, args.geo)
    if args.geo in ['box']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
    elif args.geo in ['vec']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo == 'beta':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
    elif args.geo == 'gamma':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.gamma_mode)
    # For our LapE model, we typically set args.geo to 'laplace'
    else: tmp_str = "g-{}-mode-{}".format(args.gamma, args.geo)  # Uses args.geo directly

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("logging to", args.save_path)
    writer = SummaryWriter(args.save_path) if args.do_train else SummaryWriter('./logs-debug/unused-tb')
    set_logger(args)

    with open(os.path.join(args.data_path, 'stats.txt')) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('-------------------------------' * 3)
    logging.info('Geo: %s' % args.geo)
    logging.info('seed: %d' % args.seed)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unions using: %s' % args.evaluate_union)

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers = load_data(args, tasks)
    # Merge union queries from valid into train queries
    train_queries[(('e', ('r',)), ('e', ('r',)), ('u',))].update(valid_queries[(('e', ('r',)), ('e', ('r',)), ('u',))])
    train_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))].update(
        valid_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))])
    # for key in valid_easy_answers.keys():
    #     valid_easy_answers[key] = valid_easy_answers[key].union(valid_hard_answers[key])
    for key, valid_set in valid_hard_answers.items():
        if key in train_answers:
            train_answers[key] = train_answers[key].union(valid_set)
        else:
            train_answers[key] = valid_set
    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, nentity, nrelation, args.negative_sample_size, train_answers),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, nentity, nrelation, args.negative_sample_size, train_answers),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(valid_queries, args.nentity, args.nrelation),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(test_queries, args.nentity, args.nrelation),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    # Initialize our LapE model (KGReasoningLapE) instead of GammaE
    model = KGReasoningLapE(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        use_cuda=args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode=eval_tuple(args.beta_mode),
        gamma_mode=eval_tuple(args.gamma_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict=query_name_dict,
        drop=args.drop
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()

    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2

    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step
    # For LapE, we can log the mode as 'laplace'
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    elif args.geo == 'gamma':
        logging.info('gamma mode = %s' % args.gamma_mode)
    else:
        logging.info('Using Laplace-based embeddings for KG reasoning.')
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []
        # Training Loop
        for step in range(init_step, args.max_steps):
            if step == 2 * args.max_steps // 3:
                args.valid_steps *= 4

            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_' + metric, log[metric], step)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_' + metric, log[metric], step)
                log = model.train_step(model, optimizer, train_path_iterator, args, step)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader,
                                                 query_name_dict, 'Valid', step, writer)
                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                                query_name_dict, 'Test', step, writer)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)

    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict,
                                    'Test', step, writer)

    logging.info("Training finished!!")
