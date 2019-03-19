import sys
import os

'''
Imports
'''
import numpy as np
import argparse
from glob import glob
import cv2
import warnings
warnings.filterwarnings("ignore")

'''
Torch
'''
import torch
from torch.utils import data as torchdata
from torch.utils.data.dataset import Dataset

'''
Network dependents
'''
from callbacks_test import *

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
# import from parent folder
from deepnet_test import TestDeepNet
from utils import time_elapse_parser
sys.path.remove(os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

imread = lambda fname: cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

imsize = (128, 256)
device = None

NetStructure = None
DatasetABC = None
DatasetCombined = None


class DatasetFnames(Dataset):
    '''
    dataset from fnames
    '''
    def __init__(self, fnames, size=[128, 256], args=None):
        self.fnames = fnames
        self.height = imsize[0]
        self.width = imsize[1]

    def __getitem__(self, idx):
        rgb = cv2.resize(imread(self.fnames[idx]), (self.width, self.height)) / 255.0
        name = os.path.split(self.fnames[idx])[1][0:-4]
        sample = {'data': {'name': name, 'rgb': rgb},
                  'target': {'gt': np.zeros([1])}
                  }
        return sample

    def __len__(self):
        return len(self.fnames)


def main(debug, model_dir, epoch_to_load, output_dir_tag, dataset, module, domain):
    '''
    Datasets and loaders, path setup
    '''
    output_dir = os.path.join(model_dir, output_dir_tag)

    '''
    =====================================================================================
    Prepare dataset
    dataset_test: can be either
        1. single image - dict{'data':{}, 'target':{}}
        2. torchdata.DataLoader(pytorch_dataset, **parameters)
    ====================================================================================='''
    print('[Info] Preparing dataset')
    dataset_test = None
    dataloader_param = {'batch_size': 20, 'num_workers': 6, 'shuffle': False, 'drop_last': False, 'pin_memory': False}

    # task x: load data from fnames
    fnames_rgb = sorted(glob('/path/to/*.jpg'))
    dataset_test = torchdata.DataLoader(DatasetFnames(fnames_rgb, size=(imsize[0], imsize[1]), sun_uv=sun_uv),
                                        **dataloader_param)
    '''
    =====================================================================================
    Model: load the pre-trained model
    =====================================================================================
    '''
    sys.path.insert(0, os.path.normpath(model_dir))
    from net_bkp import ABCNET as NetStructure
    sys.path.remove(os.path.normpath(model_dir))

    model = NetStructure(imsize=imsize, domain_adapt=domain, module_name=module, device=device).to(device)
    deep_net = TestDeepNet(model, model_dir=model_dir, domain=domain, epoch_to_load=epoch_to_load, model_name='name_your_model', device=device)

    '''
    =====================================================================================
    Callback setup
    =====================================================================================
    '''
    callbacks = [
                CallbackSavePrediction(net=deep_net.model, output_dir=output_dir),
                # you could add more calbacks?
                ]

    '''
    Run deep CNNs
    '''
    print('Start computing...')
    deep_net.forward(dataset_test, callbacks)
    print('Output: %s' % output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('-i', '--input', default='model.abcd', help="input dir <0717-2130.tag>")
    parser.add_argument('-o', '--output', default='output', help="output dir tag: model_dir/output_tag")
    parser.add_argument('-d', '--dataset', default='abc', help="pre-defined dataset to use, choose from [aaa, bbb]")
    parser.add_argument('--module', default='abc', help='which module to use. Choose from [abc, def]')
    parser.add_argument('--epoch', default='last', help='str, which epoch to load')
    parser.add_argument('--debug', action='store_true', help='debug, small minibatch, w/o random')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()
    device = 'cpu' if args.cpu else 'cuda'

    model_base_dir = os.path.abspath(os.path.join('/path/to/the/results/folder'))

    main(debug=args.debug,
         model_dir=os.path.join(model_base_dir, args.input),
         epoch_to_load=args.epoch,
         output_dir_tag=args.output,
         dataset=args.dataset,
         module=args.module,
         domain=False,
         )
    print('done')
