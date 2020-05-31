import torch
import cv2

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_dims(sample):
    norm_dims = 3
    curr_dims = sample.shape[1]

    if curr_dims < norm_dims:
        chs = [ sample[:1,i-1,:,:] for i in range(curr_dims) ]
        for _ in range(curr_dims, norm_dims):
            chs.append(sample[:1,0,:,:])
        return torch.cat(tuple(chs), dim=0).unsqueeze(0)

    if curr_dims > norm_dims:
        return sample[:1,:3,:,:]

    return sample


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y


class AverageMeter(object):
    '''
        Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    avg = 0.0

    for i in range(len(outputs)):
        avg += torch.mean(torch.abs(outputs[i] - targets[i]))

    return float(avg / len(outputs))
