import sys
import os
from glob import glob
import torch
import time
import numpy as np


class TestDeepNet():
    '''
    Perform forward for a net and loader
    '''

    def __init__(self, model, model_dir, domain, epoch_to_load=None, model_name="", device='cuda'):
        sys.path.append(model_dir)
        self.MODEL_NAME=model_name
        print('model dir: %s' % (model_dir))
        if not os.path.exists(model_dir):
            print('NOT Found ERROR: model dir: %s' % (model_dir))
            raise FileNotFoundError
        '''
        =====================================================================================
        Network setup: load the parameters
        =====================================================================================
        '''
        if epoch_to_load is None:
            ckpt_filename = sorted(glob(os.path.join(model_dir, 'log', '.%s.model.*.ckpt' % (model_name))))[-1]
        else:
            ckpt_filename = os.path.join(model_dir, 'log', '.%s.model.%s.ckpt' % (model_name, epoch_to_load))

        model_loader = TorchModelLoader(model, device).load_torch_model(ckpt_filename)
        self.model = model_loader.net
        self.model.eval()
        self.device = model_loader.device
        self.epoch = model_loader.epoch

    def map_data_to_device(self, data, is_training):
        '''
            map dataloader data to torch device (cpu, gpu)
            data: list or dict
        '''
        if type(data) is list:
            data = [d.to(self.device) for d in data]
        if type(data) is dict:
            for key in data.keys():
                if type(data[key]) is torch.Tensor:
                    data[key] = data[key].to(self.device)
                    if is_training:
                        data[key].requires_grad = True
                    if data[key].dtype is torch.float64:
                        data[key] = data[key].type(torch.float32)
                else:  # string, fname
                    data[key] = data[key]
        return data

    def forward(self, loader, callbacks):
        if type(loader) == torch.utils.data.dataloader.DataLoader:
            self.forward_dataset(loader, callbacks)
        elif type(loader) == dict:
            self.forward_images(loader, callbacks)
        else:
            print('[ERROR] Not implemented with this type of loader: %s' % (type(loader)))
            raise NotImplementedError

    def forward_dataset(self, loader, callbacks):
        # In test phase, no gradients (for memory efficiency)
        self.model.eval()
        with torch.no_grad():
            _avg_batch_time = []
            _avg_callback_time = []
            for i, samples in enumerate(loader):
                _avg_batch_time_start = time.time()
                target = self.map_data_to_device(samples['target'], is_training=False)
                data = self.map_data_to_device(samples['data'], is_training=False)
                network_output = self.model(data, epoch=self.epoch, max_epochs=500)
                _avg_batch_time.append(time.time() - _avg_batch_time_start)

                _avg_callback_time_start = time.time()
                for callback in callbacks:
                    callback.batch(network_output, target, data)
                _avg_callback_time.append(time.time() - _avg_callback_time_start)
                if (i) % np.maximum(1, int(len(loader)/5)) == 0:
                    print('Inference [%03d/%03d], avg time: inference[%.1f]s, callbacks[%.1f]s' %
                        (i, len(loader), np.mean(_avg_batch_time), np.mean(_avg_callback_time)))
                    _avg_batch_time = []
                    _avg_callback_time = []
            for callback in callbacks:
                callback.epoch(network_output, target, data)

    def forward_images(self, samples, callbacks=[]):
        # In test phase, no gradients (for memory efficiency)
        self.model.eval()
        with torch.no_grad():
            target = self.map_data_to_device(samples['target'], is_training=False)
            data = self.map_data_to_device(samples['data'], is_training=False)
            network_output = self.model(data, epoch=self.epoch, max_epochs=500)
            for callback in callbacks:
                callback.batch(network_output, target, data)
            for callback in callbacks:
                callback.epoch(network_output, target, data)


class TorchModelLoader():
    '''
    Input: net, model definition
           model_path, saved weights
    Init the pytorch network with pre-trained weights from `model_path`
    '''

    def __init__(self, net, device):
        device = device if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.net = net.to(self.device)
        self.epoch = None

    def load_torch_model(self, ckpt_fname='/full/path/to/*.ckpt'):
        if os.path.exists(ckpt_fname):
            state_dict, epoch = self._load_checkpoint(ckpt_fname)
            self.net.load_state_dict(state_dict)
            self.epoch = epoch
        else:
            raise RuntimeError("Can't load model {}. FileNotFoundError.".format(ckpt_fname))
        return self

    def _load_checkpoint(self, fname_model):
        print("Loading model: %s" % fname_model)
        state = torch.load(fname_model, map_location=self.device)
        state_dict = state['state_dict']
        epoch = state['epoch']
        return state_dict, epoch
