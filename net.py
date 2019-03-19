import torch
from torch import nn


class ABCNet(nn.Module):

    def __init__(self, imsize, domain_adapt, module_name, device='cuda'):
        super(ABCNet, self).__init__()
        self.imsize = imsize
        self.device = device
        self.do_domain_adapt = domain_adapt
        self.model = "Define you CNN network as nn.Module, with a forward function inside"
        '''Other parameters'''
        pass

    def forward(self, data, epoch, max_epochs):
        self.ibatch += 1
        network_output = {}
        self.epoch = epoch
        ldr = data['rgb']

        x = ldr.permute([0, 3, 1, 2])
        out = self.model(x)

        '''outputs'''
        network_output['out'] = out
        return network_output

    def loss(self, network_output, target):
        '''
        Define different losses
        Pick a final loss to train the network
        '''
        loss_dict = {}

        '''
        compute the loss
        '''
        pass

        # define the final loss
        loss_dict['final'] = 0
        loss_dict['final'] += loss_dict['any-other-loss']

        self.loss_dict = loss_dict
        return loss_dict['final']


class GradientScale(torch.autograd.Function):

    def __init__(self, name=None):
        self.lambdar = 0
        self.name = name

    def forward(self, x, lambdar):
        self.lambdar = lambdar
        return x.view_as(x)

    def backward(self, grad_output):
        if self.name:  # for debug
            print('%s: [min, mean, max]=[%.15f %.15f %.15f]' % (self.name, grad_output.min(), grad_output.mean(), grad_output.max()))
        return grad_output * self.lambdar, None


class GradientRecordHook(torch.autograd.Function):
    '''
    Simple way to record gradient
    '''

    def __init__(self):
        self.lambdar = 0
        self.gradients = []
        self.mag = None
        self.std = None

    def forward(self, x):
        '''
        Do Nothing
        '''
        return x.view_as(x)

    def backward(self, grad_output):
        # only record the magnitude, return the original gradient
        self.mag = torch.mean(torch.abs(grad_output.data))
        self.std = torch.std(grad_output.data)
        return grad_output
