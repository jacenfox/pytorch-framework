import torch
import os
import sys
import numpy as np
import time
import glob
from utils import time_elapse_parser
import colorama

class TrainDeepNet(object):
    """docstring for TrainDeepNet"""

    def __init__(self, model_path, model, domain_adapt, learning_rate, callbacks, model_name='LDR2HDR', device='cuda'):
        self.model_path = model_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print('[Output Path] %s' % (model_path))
        self.callbacks = callbacks
        self.device = device

        self.model = model
        self.MODEL_NAME=model_name
        self.do_domain_adapt = domain_adapt
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0, eps=1e-8)
        self.epoch = 0
        self.epoch_ckpt = 0
        self._best_loss = None

    def map_data_to_device(self, data, is_training):
        '''
            map dataloader data to torch device (cpu, gpu)
            data: list or dict
        '''
        if type(data) is list:
            data = [d.to(self.device) for d in data]
        if type(data) is dict:
            for key in data.keys():
                try:
                    if type(data[key]) is torch.Tensor:
                        data[key] = data[key].to(self.device)
                        if is_training:
                            data[key].requires_grad = True
                        else:
                            data[key].requires_grad = False
                        if data[key].dtype is torch.float64:
                            data[key] = data[key].type(torch.float32)
                    else:  # string, fname
                        data[key] = data[key]
                except TypeError:
                    print('Type Error in processing: ', key, type(data[key]), data[key].shape, data[key].dtype)
                    raise TypeError
        return data

    def loop(self, train_loader, test_loader, max_epochs):
        _start_time = time.time()
        self.max_epochs = max_epochs
        for i in range(self.epoch_ckpt, max_epochs):
            _epoch_start_time = time.time()
            self.epoch = i
            self.train(train_loader)
            self.test(test_loader)
            print('Epoch [%03d/%03d] Epoch time [%s] Running time [%s]' % (self.epoch, self.max_epochs,
                                                                           time_elapse_parser(time.time() - _epoch_start_time),
                                                                           time_elapse_parser(time.time() - _start_time)))
            if (i) % 5 == 0:
                print(colorama.Fore.GREEN + '[Runing CMD] %s' % ' '.join(sys.argv[0:]) + colorama.Style.RESET_ALL)
                print(colorama.Fore.GREEN + '[Output dir] %s' % self.model_path + colorama.Style.RESET_ALL)
                print(colorama.Back.CYAN + '='*50 + colorama.Style.RESET_ALL)

    def train(self, train_loader):
        # Train the model
        self.model.train()
        epoch = self.epoch
        total_step = len(train_loader)
        print_list = {'loss':[], 'GPU': [], 'Loading': [], 'CBs': []}
        _last_batch_end_time = time.time()
        for i, samples in enumerate(train_loader):
            print_list['Loading'].append(time.time() - _last_batch_end_time)
            _avg_gpu_time_start = time.time()
            # map data to device
            target = self.map_data_to_device(samples['target'], is_training=True)
            data = self.map_data_to_device(samples['data'], is_training=True)
            # forward
            network_output = self.model(data, epoch, self.max_epochs)
            loss = self.model.loss(network_output, target)
            print_list['loss'].append(loss.data.cpu().numpy())
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print_list['GPU'].append(time.time() - _avg_gpu_time_start)
            # callbacks
            _avg_callback_time_start = time.time()            
            self.after_batch_callbacks(network_output, target, data, loss_dict=self.model.loss_dict, is_training=True)
            print_list['CBs'].append(time.time() - _avg_callback_time_start)
            # additional process
            _last_batch_end_time = time.time()
            if (i) % np.maximum(1, int(total_step/5)) == 0:
                print('  Step[%03d/%03d] Loss: [%.4f] Time(s): CPU[%.2f] GPU[%.1f] CBs[%.1f] per batch' %
                      (i, total_step, np.mean(print_list['loss']), np.mean(print_list['Loading']), 
                                      np.mean(print_list['GPU']), np.mean(print_list['CBs'])))
                print_list = {'loss':[], 'GPU': [], 'Loading': [], 'CBs': []}

        self.save_checkpoint(loss.data.cpu().numpy())
        self.after_epoch_callbacks(network_output, target, data, loss_dict=self.model.loss_dict, is_training=True)
        self.model.ibatch = 0

    def test(self, test_loader):
        # Test the model
        self.model.eval()
        # In test phase, no gradients (for memory efficiency)
        print('  Testing', end =" ")
        _test_time_start = time.time()
        with torch.no_grad():
            for i, samples in enumerate(test_loader):
                target = self.map_data_to_device(samples['target'], is_training=False)
                data = self.map_data_to_device(samples['data'], is_training=False)
                network_output = self.model(data, self.epoch, self.max_epochs)
                self.after_batch_callbacks(network_output, target, data, loss_dict=self.model.loss_dict, is_training=False)
            self.after_epoch_callbacks(network_output, target, data, loss_dict=self.model.loss_dict, is_training=False)
        print('time [%.2f]' % (time.time()-_test_time_start))
        self.model.ibatch = 0

    def after_batch_callbacks(self, network_output, target, data, loss_dict, is_training):
        callbacks = self.callbacks
        for callback_fun in callbacks:
            callback_fun.batch(self, network_output, target, data, loss_dict, is_training)

    def after_epoch_callbacks(self, network_output, target, data, loss_dict, is_training):
        callbacks = self.callbacks
        for callback_fun in callbacks:
            callback_fun.epoch(self, network_output, target, data, loss_dict, is_training)

    def save_checkpoint(self, loss):
        # Save the model checkpoint
        log_path = os.path.join(self.model_path, 'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        ckpt_data = {'epoch': self.epoch,
                     'state_dict': self.model.state_dict(),
                     'loss': loss,
                    }
        torch.save(ckpt_data, os.path.join(log_path, '.%s.model.last.ckpt'%(self.MODEL_NAME)))

        if self.epoch % 50 == 0 or self.epoch >= (self.max_epochs-5):
            torch.save(ckpt_data, os.path.join(log_path, '.%s.model.%03d.ckpt'%(self.MODEL_NAME, self.epoch)))
        if (self._best_loss is None) or (self._best_loss > loss):
            self._best_loss = loss
            torch.save(ckpt_data, os.path.join(log_path, '.%s.model.best.ckpt'%(self.MODEL_NAME)))


    def load_checkpoint(self, model_path, epoch=-1):
        """
        :return:
        """
        if (epoch == -1) or (epoch == 'last'):
            filename = sorted(glob.glob(os.path.join(model_path, 'log', '.%s.model.last.ckpt'%(self.MODEL_NAME))))[-1]
        elif epoch == 'best':
            filename = sorted(glob.glob(os.path.join(model_path, 'log', '.%s.model.best.ckpt'%(self.MODEL_NAME))))[-1]
        else:
            filename = os.path.join(model_path, 'log', '.%s.model.%03d.ckpt'%(self.MODEL_NAME) % (epoch))
        if os.path.exists(filename):
            print("Loading model from %s" % (filename))
        else:
            print("Cannot load model from %s" % (filename))
            return
        ckpt_data = torch.load(filename)
        self.model.load_state_dict(ckpt_data['state_dict'])
        self.epoch_ckpt = ckpt_data['epoch']
        self._best_loss = ckpt_data['loss'] if 'loss' in ckpt_data.keys() else None

    def resume_training(self):
        self.load_checkpoint(self.model_path, epoch=-1)
