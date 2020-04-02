import torch
import numpy as np
from torchtext import data
import logging
import random
import argparse

from data.data import DocField, DocDataset, DocIter
from data.lazy_iterator import BucketIterator, Iterator
import time

from model.seq2seq import train, decode
from pathlib import Path
import json


def parse_args():
    parser = argparse.ArgumentParser(description='order')

    # dataset settings
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--lang', type=str, nargs='+', help="the suffix of the corpus, translation language")
    parser.add_argument('--valid', type=str)

    parser.add_argument('--writetrans', type=str, help='write translations for to a file')
    parser.add_argument('--ref', type=str, help='references, word unit')

    parser.add_argument('--vocab', type=str)
    parser.add_argument('--vocab_size', type=int, default=40000)

    parser.add_argument('--load_vocab', action='store_true', help='load a pre-computed vocabulary')
    parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')
    parser.add_argument('--pool', type=int, default=100, help='shuffle batches in the pool')

    # model name
    parser.add_argument('--model', type=str, default='[time]')

    # network settings
    parser.add_argument('--n_layers', type=int, default=5, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=2, help='number of heads')

    parser.add_argument('--d_emb', type=int, default=278, help='dimention size for hidden states')
    parser.add_argument('--d_rnn', type=int, default=507, help='dimention size for FFN')
    parser.add_argument('--d_mlp', type=int, default=507, help='dimention size for FFN')
    parser.add_argument('--senenc', default='bow', help='sentence encoder')

    parser.add_argument('--d_pair', type=int, default=512, help='dimention size for pair representaiton')
    parser.add_argument('--d_label', type=int, default=128, help='dimention size for label embedding')

    parser.add_argument('--gnnl', default=2, type=int, help='stacked layer number')
    parser.add_argument('--attdp', default=0, type=float, help='self-att dropout')

    parser.add_argument('--initnn', default='standard', help='parameter init')
    parser.add_argument('--early_stop', type=int, default=0)

    parser.add_argument('--lamb_rela', type=float, default=0, help='lambda of relative order loss')

    # running setting
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test',
                                 'pwr', 'pair'])
    parser.add_argument('--seed', type=int, default=1234, help='seed for randomness')

    parser.add_argument('--keep_cpts', type=int, default=1, help='save n checkpoints, when 1 save best model only')

    # training
    parser.add_argument('--eval_every', type=int, default=100, help='validate every * step')
    parser.add_argument('--save_every', type=int, default=-1, help='save model every * step (5000)')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--delay', type=int, default=1)

    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')

    # lr decay
    parser.add_argument('--lrdecay', type=float, default=0, help='learning rate decay')
    parser.add_argument('--patience', type=int, default=0, help='learning rate decay 0.5')

    parser.add_argument('--maximum_steps', type=int, default=5000000, help='maximum steps you take to train a model')
    parser.add_argument('--drop_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--input_drop_ratio', type=float, default=0.1, help='dropout ratio only for inputs')

    parser.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping')

    # decoding
    parser.add_argument('--beam_size', type=int, default=1,
                        help='beam-size used in Beamsearch, default using greedy decoding')

    parser.add_argument('--test', type=str, default=None, help='test src file')
    parser.add_argument('--test_order', type=str, default=None, help='test order file')

    # model saving/reloading, output translations
    parser.add_argument('--load_from', nargs='+', default=None, help='load from 1.modelname, 2.lastnumber, 3.number')

    parser.add_argument('--resume', action='store_true',
                        help='when resume, need other things besides parameters')
    # save path
    parser.add_argument('--main_path', type=str, default="./")
    parser.add_argument('--model_path', type=str, default="models")
    parser.add_argument('--decoding_path', type=str, default="decoding")

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def curtime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def override(args, load_dict, except_name):
    for k in args.__dict__:
        if k not in except_name:
            args.__dict__[k] = load_dict[k]


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        if args.load_from is not None and len(args.load_from) == 1:
            load_from = args.load_from[0]
            print('{} load the checkpoint from {} for initilize or resume'.
                  format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            checkpoint = None

        # if not resume(initilize), only need model parameters
        if args.resume:
            print('update args from checkpoint')
            load_dict = checkpoint['args'].__dict__
            except_name = ['mode', 'resume', 'maximum_steps']
            override(args, load_dict, tuple(except_name))

        main_path = Path(args.main_path)
        model_path = main_path / args.model_path
        decoding_path = main_path / args.decoding_path

        for path in [model_path, decoding_path]:
            path.mkdir(parents=True, exist_ok=True)

        args.model_path = str(model_path)
        args.decoding_path = str(decoding_path)

        if args.model == '[time]':
            args.model = time.strftime("%m.%d_%H.%M.", time.gmtime())

        # setup random seeds
        set_seeds(args.seed)

        # special process, shuffle each document
        # DOC = DocField(batch_first=True, include_lengths=True, eos_token='<eos>', init_token='<bos>')
        DOC = DocField(batch_first=True, include_lengths=True)
        ORDER = data.Field(batch_first=True, include_lengths=True, pad_token=0, use_vocab=False,
                           sequential=True)

        train_data = DocDataset(path=args.corpus, text_field=DOC, order_field=ORDER)

        dev_data = DocDataset(path=args.valid, text_field=DOC, order_field=ORDER)

        DOC.vocab = torch.load(args.vocab)
        print('vocab {} loaded'.format(args.vocab))
        args.__dict__.update({'doc_vocab': len(DOC.vocab)})

        train_flag = True
        train_real = DocIter(train_data, args.batch_size, device='cuda',
                             train=train_flag,
                             shuffle=train_flag,
                             sort_key=lambda x: len(x.doc))
        setattr(train_real, 'sforder', True)

        dev_real = DocIter(dev_data, 1, device='cuda', batch_size_fn=None,
                           train=False, repeat=False, shuffle=False, sort=False)
        setattr(dev_real, 'sforder', True)

        args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
        print(args_str)

        test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, order=args.test_order)
        test_real = DocIter(test_data, 1, device='cuda', batch_size_fn=None,
                            train=False, repeat=False, shuffle=False, sort=False)
        setattr(test_real, 'sforder', False)

        print('{} Start training'.format(curtime()))
        train(args, train_real, dev_real, test_real, (DOC, ORDER), checkpoint)
    else:
        if len(args.load_from) == 1:
            load_from = '{}.best.pt'.format(args.load_from[0])
            print('{} load the best checkpoint from {}'.format(curtime(), load_from))
            checkpoint = torch.load(load_from, map_location='cpu')
        else:
            raise RuntimeError('must load model')

        # when translate load_dict update args except some
        print('update args from checkpoint')
        load_dict = checkpoint['args'].__dict__
        if args.mode == 'pwr':
            # except_name = ['mode', 'vocab', 'load_from', 'test', 'batch_size']
            except_name = ['mode', 'load_from', 'test', 'batch_size']
        else:
            except_name = ['mode', 'load_from', 'test', 'writetrans', 'beam_size', 'batch_size', 'test_order']

        override(args, load_dict, tuple(except_name))

        print('{} Load test set'.format(curtime()))

        DOC = DocField(batch_first=True, include_lengths=True)
        ORDER = data.Field(batch_first=True, include_lengths=True, pad_token=0, use_vocab=False,
                           sequential=True)

        DOC.vocab = torch.load(args.vocab)
        print('vocab {} loaded'.format(args.vocab))
        args.__dict__.update({'doc_vocab': len(DOC.vocab)})

        args_str = json.dumps(args.__dict__, indent=4, sort_keys=True)
        print(args_str)

        test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, order=args.test_order)
        test_real = DocIter(test_data, 1, device='cuda', batch_size_fn=None,
                            train=False, repeat=False, shuffle=False, sort=False)
        setattr(test_real, 'sforder', False)

        print('{} Load data done'.format(curtime()))
        start = time.time()
        decode(args, test_real, (DOC, ORDER), checkpoint)
        print('{} Decode done, time {} mins'.format(curtime(), (time.time() - start) / 60))
