from torch.autograd import Variable


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += n * val
        self.avg = self.sum / self.count

def accuracy(output,label):
    _,pred=output.topk(1)
    pred=pred.t()
    # print(pred)
    # print(label)
    correct=pred.eq(label.unsqueeze(0).view_as(pred))
    acc=correct.data.sum()/correct.size()[1]
    return acc

def tensor2variable(x=None,device=None,requires_grad=False):
    x=x.to(device)
    return Variable(x,requires_grad=requires_grad)



import torch
from collections import OrderedDict
from torch.nn.parameter import Parameter

def state_dict(model, destination=None, prefix='', keep_vars=False):
    own_state = model.module if isinstance(model, torch.nn.DataParallel) \
        else model
    if destination is None:
        destination = OrderedDict()
    for name, param in own_state._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.data
    for name, buf in own_state._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf
    for name, module in own_state._modules.items():
        if module is not None:
            state_dict(module, destination, prefix + name + '.', keep_vars=keep_vars)
    return destination

def load_state_dict(model, state_dict, strict=True):
    own_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
        else model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                    'whose dimensions in the model are {} and '
                                    'whose dimensions in the checkpoint are {}.'
                                    .format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                            .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))
