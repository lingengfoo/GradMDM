import os
import logging
import json

class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""

    def __init__(self):
        self.len = -1  # set up the maximum length

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        if self.len == -1:
            self.len = len(vals)
            self.reset()
            
        len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += vals[i].item() * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""

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

def more_config(args):
    """config save path and logging"""
    args.save_path = os.path.join(args.save_folder, args.model_type)
    os.makedirs(args.save_path, exist_ok=True)

    args.logger_file = os.path.join(args.save_path,
                                    'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    # save training parameters to the folder
    save_args(args)

def save_args(args):
    args.args_file = args_file = os.path.join(args.save_path, 'train_args.json')
    with open(args_file, 'w') as f:
        args_dict = {
            k: v for k, v in args._get_kwargs()}
        json.dump(args_dict, f)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res