'''https://github.com/sksq96/pytorch-summary'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
from contextlib import redirect_stdout


def summary(model, input_size, output_fname=None, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, dict):
                for key in output.keys():
                    summary[m_key]["output_shape"] = [
                        [-1] + list(output[key].size())[1:]]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # batch_size of 2 for batchnorm
    x = torch.rand([1] + list(input_size)).type(dtype)
    print('[Summary]', x.shape)
    data = {'rgb':x, 'rgbnormal':x, 'rgbd':x}
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(data=data, epoch=1, max_epochs=1)

    # remove these hooks
    for h in hooks:
        h.remove()

    # forward output to a file
    with open(output_fname, 'w') as f:
        with redirect_stdout(f):
            print_summary(summary, input_size, batch_size)

def print_summary(summary, input_size, batch_size):
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15} {:>15}".format("Layer (type)", "Output Shape", "Tensor Size (MB)", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        total_params += summary[layer]["nb_params"]
        layer_tensor_size = 0
        if type(summary[layer]["output_shape"][0]) is int:
            layer_tensor_size += np.prod(summary[layer]["output_shape"])
        elif  type(summary[layer]["output_shape"][0]) is list:
            for shape in summary[layer]["output_shape"]:
                layer_tensor_size += np.prod(shape)
        else:
            import ipdb; ipdb.set_trace()
            pass
        
        total_output += layer_tensor_size
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "%.2f" % (abs(layer_tensor_size * 4. / (1024 ** 2.))),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: #{0:,}".format(total_params))
    print("Trainable params: #{0:,}".format(trainable_params))
    print("Non-trainable params: #{0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB) including gradients: %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
