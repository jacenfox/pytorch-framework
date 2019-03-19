import sys
import os

imsize = [128, 256]
device = None

ABCNet = None
DatasetABC = None
DatasetCombined = None
'''
Imports
'''
from datetime import datetime

import numpy as np
import argparse
import colorama
from shutil import copyfile, SameFileError

'''
Torch
'''
import torch
from torch.utils import data as torchdata

'''
Network dependents
'''
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
# import from parent folder
from deepnet_training import TrainDeepNet
from torchsummary import summary
from utils import time_elapse_parser
sys.path.remove(os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

from callbacks_training import CallbackSaveSamples, CallbackLossCurves, CallbackGradientCurves


def main(debug, subset, domain, module_name, resume, batch_size, lr, max_epochs, tag=''):
    timenow = datetime.now().strftime('%m%d-%H%M')
    '''
    =====================================================================================
    Prepare folder for training
    ====================================================================================='''
    modelzoo_dir = os.path.join('../../result/some-tags')
    if resume:
        model_path = os.path.join(modelzoo_dir, resume)
        sys.path.append(os.path.join(model_path))
        print(colorama.Back.CYAN + 'Resume from: %s' % (model_path) + colorama.Style.RESET_ALL)
        from net_bkp import NET as ABCNet
        from dataset_bkp import DatasetABC, DatasetCombined
    else:
        from net import ABCNet
        from dataset import DatasetABC, DatasetCombined
        model_path = os.path.join(modelzoo_dir, timenow + tag)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not resume:
        try:
            copyfile(src=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'net.py'), dst=os.path.join(model_path, 'net_bkp.py'))
            copyfile(src=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.py'), dst=os.path.join(model_path, 'train_bkp.py'))
            copyfile(src=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset.py'), dst=os.path.join(model_path, 'dataset_bkp.py'))
        except SameFileError:
            pass
    with open(os.path.join(model_path, 'cmd-args.txt'), 'w') as f:
        f.write(' '.join(sys.argv[0:]))

    '''
    =====================================================================================
    Callback, training loop setup
    ====================================================================================='''
    model = ABCNet(imsize=imsize, domain_adapt=domain, module_name=module_name, device=device).to(device)
    summary(model, (imsize[0], imsize[1], 3), output_fname=os.path.join(model_path, 'network-structure.txt'), device=device)

    callbacks = [
                CallbackSaveSamples(call_me_every_n_epoch=1),
                CallbackLossCurves(),
                CallbackGradientCurves(),
                ]
    deepnet_training_obj = TrainDeepNet(model_path=model_path,
                                        model=model,
                                        domain_adapt=domain,
                                        learning_rate=lr,
                                        device=device,
                                        model_name='ABC-NET-or-other-tag',
                                        callbacks=callbacks)

    if resume:
        deepnet_training_obj.resume_training()

    '''
    =====================================================================================
    Datasets and loaders
    ====================================================================================='''
    data_base_path = '/path/to/data/'
    augmentation_para = {'doFlip': True}

    # carla
    dataset_tr = DatasetABC(os.path.join(data_base_path, 'some-tags'), size=(imsize[0], imsize[1]), typeIn='train', augmentation_para=augmentation_para)
    dataset_vld = DatasetABC(os.path.join(data_base_path, 'some-tags'), size=(imsize[0], imsize[1]), typeIn='validation', augmentation_para=augmentation_para)
    dataset_test = DatasetABC(os.path.join(data_base_path, 'some-tags'), size=(imsize[0], imsize[1]), typeIn='test')

    datasets_to_use_tr = [dataset_tr]
    datasets_to_use_vld = [dataset_vld]
    dataset_sample_ratios = [1]

    if subset:
        print(colorama.Back.CYAN + '[Info] use subset to train' + colorama.Style.RESET_ALL)
        for _dataset in datasets_to_use_tr:
            _dataset.df = _dataset.df.iloc[0:int(len(_dataset) / 5)]
        for _dataset in datasets_to_use_vld:
            _dataset.df = _dataset.df.iloc[0:int(len(_dataset) / 5)]

    if debug:
        for _dataset in datasets_to_use_tr:
            _dataset.df = _dataset.df.iloc[0:batch_size * 2]
        for _dataset in datasets_to_use_vld:
            _dataset.df = _dataset.df.iloc[0:batch_size * 2]

    assert np.round(np.sum(dataset_sample_ratios)) == 1.0, 'sum of ratio should be 1, now: %.2f' % (np.sum(dataset_sample_ratios))
    '''
    data loader
    '''
    dataset_tr = dataset_tr
    dataset_vld = dataset_vld

    _paras_loader = {'batch_size': batch_size,
                     'num_workers': 12 if not debug else 8,
                     'shuffle': True if not debug else False,
                     'pin_memory': True,
                     'drop_last': True}
    train_loader = torchdata.DataLoader(dataset_tr, **_paras_loader)
    _paras_loader['shuffle'] = False
    _paras_loader['drop_last'] = False  # in case of small vld dataset
    vld_loader = torchdata.DataLoader(dataset_vld, **_paras_loader) if not debug else train_loader
    print('[Dataloader][bs*%d] train: #%04d\tvalidation: #%04d' % (batch_size, len(train_loader), len(vld_loader)))
    print(colorama.Back.CYAN + '=' * 50 + colorama.Style.RESET_ALL)

    '''
    =====================================================================================
    Start the training loop
    ====================================================================================='''
    deepnet_training_obj.loop(train_loader=train_loader, test_loader=vld_loader, max_epochs=max_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pytorch network')
    parser.add_argument('--tag', default='', help="tag add to the name. disabled in debug", required=False)
    parser.add_argument('--epochs', default=1000, help="max epochs to train", required=False)
    parser.add_argument('--bs', default=50, help="batch size; recommend value: [xxx]", required=False)
    parser.add_argument('--lr', default=0.0001, help="learning rate", required=False)
    parser.add_argument('--resume', default=None, help='resume from a given folder')
    parser.add_argument('--module', default='multiscale', help='which module to use. Choose from [abc, def]')
    parser.add_argument('--subset', action='store_true', help='use subset (less images) to train the network')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--debug', action='store_true', help='debug, small minibatch, fixed subset, w/o random')
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    domain = not args.nodomain
    domain = False
    args.tag = '.debug' if args.debug else args.tag

    if args.debug:
        args.bs = 2
        print(colorama.Back.RED + '======== Debug ========' + colorama.Style.RESET_ALL)

    main(debug=args.debug, subset=args.subset, domain=domain, module_name=args.module, resume=args.resume, 
         batch_size=int(args.bs), lr=float(args.lr), max_epochs=int(args.epochs), tag=args.tag)
    print('done')