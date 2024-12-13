"""
Code forked from https://github.com/erkil1452/touch/tree/master/classification/main.py
Modified to run as a function
"""

import sys; sys.path.insert(0, '.')
import numpy as np
import sys, os, re, time, shutil, math, random, datetime, argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from .ObjectClusterDataset import ObjectClusterDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


batch_size = 32
workers = 0

doFilter = True

class Trainer(object):

    def __init__(self, project_path, nFrames):
        # super(Trainer, self).__init__()
        self.nFrames=nFrames
        self.metaFile = f'{project_path}/../grasp_trainer/metadata.mat'
        self.snapshotDir = f"{project_path}/../grasp_trainer/weights"
        self.init()

    def init(self):
        # Init model
        self.val_loader = self.loadDatasets('test', False, False)

        self.initModel()

    def loadDatasets(self, split='train', shuffle=True, useClusterSampling=False):
        return torch.utils.data.DataLoader(
            ObjectClusterDataset(split=split, doAugment=(split=='train'), doFilter = doFilter, sequenceLength=self.nFrames, metaFile=self.metaFile, useClusters=useClusterSampling),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=workers)

    def run(self, epochs):
        print('[Trainer] Starting...')

        self.counters = {
            'train': 0,
            'test': 0,
        }

        val_loader = self.val_loader
        train_loader = self.loadDatasets('train', True, False)

        for epoch in range(epochs):
            print('Epoch %d/%d....' % (epoch, epochs))
            self.model.updateLearningRate(epoch)

            self.step(train_loader, self.model.epoch, isTrain = True)
            prec1, _ = self.step(val_loader, self.model.epoch, isTrain = False, sinkName=None)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.model.bestPrec
            if is_best:
                self.model.bestPrec = prec1
            self.model.epoch = epoch + 1
            self.saveCheckpoint(self.model.exportState(), is_best)

        # Final results
        self.doSink()
            
        print('DONE')

    def doSink(self):
        res = {}

        print('Running test...')
        res['test-top1'], res['test-top3'] = self.step(self.val_loader, self.model.epoch, isTrain = False, sinkName = 'test')

        print('Running test with clustering...')
        val_loader_cluster = self.loadDatasets('test', False, True)
        res['test_cluster-top1'], res['test_cluster-top3'] = self.step(val_loader_cluster, self.model.epoch, isTrain = False, sinkName = 'test_cluster')

        print('--------------\nResults:')
        for k,v in res.items():
            print('\t%s: %.3f %%' % (k,v))
            

    
    def initModel(self):
        cudnn.benchmark = True

        from .ClassificationModel import ClassificationModel as Model # the main model
        self.model = Model(numClasses = len(self.val_loader.dataset.meta['objects']), sequenceLength = self.nFrames)
        self.model.epoch = 0
        self.model.bestPrec = -1e20


    def step(self, data_loader, epoch, isTrain = True, sinkName = None):

        if isTrain:
            data_loader.dataset.refresh()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        results = {
            'batch': [],
            'rec': [],
            'frame': [],
        }
        catRes = lambda res,key: res[key].cpu().numpy() if not key in results else np.concatenate((results[key], res[key].cpu().numpy()), axis=0)
    
        end = time.time()
        for i, (inputs) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputsDict = {
                'image': inputs[1],
                'pressure': inputs[2],
                'objectId': inputs[3],
                }

            res, loss = self.model.step(inputsDict, isTrain, params = {'debug': True})

            losses.update(loss['Loss'], inputs[0].size(0))
            top1.update(loss['Top1'], inputs[0].size(0))
            top3.update(loss['Top3'], inputs[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if isTrain:
                self.counters['train'] = self.counters['train'] + 1

            print('{phase}: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'
                .format(
                 epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3,
                phase=('Train' if isTrain else 'Test')))
        
        self.counters['test'] = self.counters['test'] + 1

        return top1.avg, top3.avg



    def saveCheckpoint(self, state, is_best):
        if not os.path.isdir(self.snapshotDir):
            os.makedirs(self.snapshotDir, 0o777)
        chckFile = os.path.join(self.snapshotDir, 'checkpoint.pth.tar')
        print('Writing checkpoint to %s...' % chckFile)
        torch.save(state, chckFile)
        if is_best:
            bestFile = os.path.join(self.snapshotDir, 'model_best.pth.tar')
            shutil.copyfile(chckFile, bestFile)
        print('\t...Done.')


    @staticmethod
    def make(project_path, nFrames = 1):

        random.seed(454878 + time.time() + os.getpid())
        np.random.seed(int(12683 + time.time() + os.getpid()))
        torch.manual_seed(23142 + time.time() + os.getpid())

        ex = Trainer(project_path, nFrames)
        ex.run()



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

        
if __name__ == "__main__":    
    Trainer.make()
