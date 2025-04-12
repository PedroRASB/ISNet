import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import torchvision as tv
import torch.utils.data as data
import os
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import time
import copy
import numpy as np
import matplotlib
import math
import warnings
import SingleLabelEval as SLE
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import h5py
from argparse import ArgumentParser
import torch.utils.data as Tdata


import sys
import locations
sys.path.append(locations.ISNet)
TrainedPath=locations.TrainedPath


import ISNetFunctionsZe as IsNet
import LRPDenseNetZe as LRPDenseNet
import ISNetLayersZe as LRPL
import ISNetLightningZe as ISNetLightning

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


        
def InterpretArgs(hparams):
    batch=int(hparams.batch)
    
    if ('[' in hparams.devices):#call for specific gpu: use [x]
        device=[int(hparams.devices[1])]
    else:
        device=int(hparams.devices)
        
    findLr=False
    if (hparams.LrFinder is not None):
        findLr=True
        
    heat=False
    if (hparams.heat=='1'):
        heat=True
    
    bias=False
    if (hparams.bias=='1'):
        bias=True
    
    return batch,device,findLr,heat,bias
        
class MNISTDatset(Dataset):
    def __init__(self, mode,bias,masks=True,confounding=False,source=locations.source):
        self.images=[]
        if (mode=='train'):
            if (not bias):
                self.dataset=h5py.File(source+'trainUnbiased.h5py', 'r')
            else:
                self.dataset=h5py.File(source+'train.h5py', 'r')
        if (mode=='val'):
            if (not bias):
                self.dataset=h5py.File(source+'valUnbiased.h5py', 'r')
            else:
                self.dataset=h5py.File(source+'val.h5py', 'r')
        if (mode=='test'):
            if not bias:
                self.dataset=h5py.File(source+'unbiasedTest.h5py', 'r')
            else:
                if confounding:
                    self.dataset=h5py.File(source+'confoundingTest.h5py', 'r')
                else:
                    self.dataset=h5py.File(source+'biasedTest.h5py', 'r')
        self.masks=masks
    def __len__(self):
        return (len(self.dataset['images']))
    def __getitem__(self,idx):
        image=torch.from_numpy(self.dataset['images'][idx]).unsqueeze(0).repeat(3,1,1)
        if self.masks:
            mask=torch.from_numpy(self.dataset['masks'][idx]).unsqueeze(0).repeat(3,1,1)
        label=torch.tensor(self.dataset['labels'][idx]).long()
        if self.masks:
            return image,mask,label
        else:
            return image,label
    

def main(hparams):
    #unbiased ISNet
    
    NetName=hparams.name   
    print(NetName)
    batch,device,findLr,heat,bias=InterpretArgs(hparams)

    trainSet=MNISTDatset('train',masks=heat,bias=bias)
    testSet=MNISTDatset('test',masks=heat,bias=bias)
    valSet=MNISTDatset('val',masks=heat,bias=bias)
    
    precision=hparams.precision
    if (precision=='16' or precision=='32'):
        precision=int(precision)
    
    trainingLoader=Tdata.DataLoader(trainSet,batch_size=batch,shuffle=True,
                                    num_workers=int(hparams.workers))
    validatingLoader=Tdata.DataLoader(valSet,batch_size=batch,shuffle=False,
                                      num_workers=int(hparams.workers))
    testingLoader=Tdata.DataLoader(testSet,batch_size=1,shuffle=False,
                                   num_workers=1)

    if(hparams.train=='1' and (hparams.cut=='load') and (hparams.continuing=='0')): 
        cut,cut2,means,stds=torch.load(TrainedPath+'/'+NetName+'/cut.py')
    else:
        if hparams.rule=='e':
            try:
                cut={'input':float(hparams.cut)}
                cut2={'input':float(hparams.cut2)}
                if hparams.penalizeAll=='1':
                    raise ValueError('Use --cut tune')
            except:
                if hparams.cut!='tune':
                    raise ValueError('invalid cut parameter')
                #temporary values for tune cut
                cut={'input':1e-5}
                cut2={'input':1000.0}
        if hparams.rule=='z+e':
            try:
                cut={'input':[float(hparams.cutp),float(hparams.cut)]}
                cut2={'input':[float(hparams.cutp2),float(hparams.cut2)]}
            except:
                if hparams.cut!='tune' and hparams.cut!='load':
                    raise ValueError('invalid cut parameter')
                #temporary values for tune cut
                cut={'input':[1e-5,1e-5]}
                cut2={'input':[1000.0,1000.0]}
        
    if(hparams.train=='1' and (hparams.cut=='tune') and (hparams.continuing=='0')): 
        print('Cut tunning started')
        #train standard DNN (without heatmap loss) to get standard heatmap values range
        net=ISNetLightning.ISNetLgt(architecture=hparams.backbone,
                                       dropout=(hparams.dropout=='1'),
                                       multiLabel=False,
                                       pretrained=False,classes=10,
                                       heat=(hparams.heat=='1'),e=float(hparams.e),
                                       Zb=(hparams.Zb=='1'),
                                       LR=float(hparams.lr),optim='SGD',
                                       A=float(hparams.A),B=float(hparams.B),d=float(hparams.d),
                                       E=float(hparams.Ea),
                                       selective=(hparams.selective=='1'),
                                       highest=(hparams.highest=='1'),
                                       rule=hparams.rule,
                                       multiple=(hparams.multiple=='1'),
                                       cut=cut,cut2=cut2,
                                       detach=(hparams.detach=='1'),
                                       momentum=float(hparams.momentum),
                                       penalizeAll=(hparams.penalizeAll=='1'),
                                       dLoss=float(hparams.dLoss), 
                                       randomLogit=(hparams.random=='1'),
                                       alternativeForeground=hparams.alternativeForeground)
        net.initTuneCut(epochs=int(hparams.tuneCutEpochs))
        trainer=pl.Trainer(precision=precision,callbacks=None,
                           gradient_clip_val=float(hparams.clip),
                           accelerator=hparams.accelerator,devices=device,
                           max_epochs=int(hparams.tuneCutEpochs),
                           strategy=hparams.strategy,
                           num_nodes=int(hparams.nodes),
                           auto_select_gpus=True,
                           logger=False,
                           auto_lr_find=False,
                           deterministic=True)
        trainer.fit(net,trainingLoader,validatingLoader)
        cut,cut2,means,stds=net.returnCut()
        #erase trained network
        del net
        del trainer
        print('cut values: ',cut)
        print('cut2 values: ',cut2)
        print('heatmaps sum mean value: ',means)
        print('heatmaps sum std value: ',stds)
        print('Cut tunning finished')
        os.makedirs(TrainedPath+'/'+NetName, exist_ok=True)
        torch.save((cut,cut2,means,stds),TrainedPath+'/'+NetName+'/cut.py')
        
    
        
    net=ISNetLightning.ISNetLgt(architecture=hparams.backbone,
                                  dropout=(hparams.dropout=='1'),
                                  multiLabel=False,
                                   pretrained=False,classes=10,
                                   heat=(hparams.heat=='1'),e=float(hparams.e),
                                   Zb=(hparams.Zb=='1'),
                                   LR=float(hparams.lr),P=float(hparams.P),optim='SGD',
                                   A=float(hparams.A),B=float(hparams.B),d=float(hparams.d),
                                   E=float(hparams.Ea),
                                   selective=(hparams.selective=='1'),
                                   highest=(hparams.highest=='1'),
                                   rule=hparams.rule,
                                   multiple=(hparams.multiple=='1'), 
                                   cut=cut,cut2=cut2,
                                   detach=(hparams.detach=='1'),
                                   momentum=float(hparams.momentum),
                                   penalizeAll=(hparams.penalizeAll=='1'),
                                   dLoss=float(hparams.dLoss), 
                                   randomLogit=(hparams.random=='1'),
                                   alternativeForeground=hparams.alternativeForeground)
    
    if (not os. path. exists(TrainedPath+'/'+NetName)):
        os.makedirs(TrainedPath+'/'+NetName, exist_ok=True)
        
    checkpoint_callback = ModelCheckpoint(dirpath=TrainedPath+NetName+'/',
                                          filename=NetName+'{epoch}-{step}',
                                          monitor='val_iidLoss',
                                          verbose=True,
                                          save_top_k=1,
                                          mode='min',
                                          every_n_epochs=1,
                                          save_on_train_epoch_end=False,
                                          auto_insert_metric_name=False,
                                          save_weights_only=False,
                                          save_last=True
                                          )
    
    tb_logger=pl_loggers.TensorBoardLogger(save_dir='Logs/'+NetName+'/')
    if not os.path.exists('Logs/'+NetName+'/'):
        os.makedirs('Logs/'+NetName+'/', exist_ok=True)
        
    
    if(hparams.continuing=='1' and hparams.train=='1'):
        checkpoint=TrainedPath+NetName+'/'+'last.ckpt'
    else:
        checkpoint=hparams.checkpoint
        
    if(hparams.train=='1'):
        trainer=pl.Trainer(precision=precision,callbacks=[checkpoint_callback],
                           gradient_clip_val=float(hparams.clip),
                           accelerator=hparams.accelerator,devices=device,
                           max_epochs=int(hparams.epochs),strategy=hparams.strategy,
                           num_nodes=int(hparams.nodes),
                           auto_select_gpus=True,
                           logger=tb_logger,
                           auto_lr_find=findLr,
                           deterministic=True
                          )

        if (findLr):
            trainer.tune(net,trainingLoader,validatingLoader)
            
        if checkpoint is not None:
            del net
            net=ISNetLightning.ISNetLgt.load_from_checkpoint(checkpoint)
            print('cut is:',net.cut,net.cut2)

        trainer.fit(net,trainingLoader,validatingLoader,ckpt_path=checkpoint)
    else:
        trainer=pl.Trainer(precision=precision,accelerator=hparams.accelerator,devices=device,
                           strategy=hparams.strategy,num_nodes=1,
                           auto_select_gpus=True)
        net=net.load_from_checkpoint(checkpoint)

    #test:
    net.eval()
    net.heat=False
    if(bias):
        for i in list(range(5)):
            print('')
        print(NetName+' Test Biased')
        testSet=MNISTDatset('test',masks=False,bias=True)
        testingLoader=Tdata.DataLoader(testSet,batch_size=4,shuffle=False,
                                       num_workers=1)
        if(hparams.train=='1'):
            trainer.test(dataloaders=testingLoader)
        else:
            trainer.test(net,dataloaders=testingLoader)
        pred,labels=net.TestResults
        pred,labels=torch.nan_to_num(pred).float(),labels.float()
        acc=SLE.Acc(pred,labels)
        print(NetName+' Acc:',acc)
        
        
    for i in list(range(5)):
        print('')
    print(NetName+' Test Unbiased')
    testSet=MNISTDatset('test',masks=False,bias=False)
    testingLoader=Tdata.DataLoader(testSet,batch_size=4,shuffle=False,
                                   num_workers=1)
    if(hparams.train=='1'):
        trainer.test(dataloaders=testingLoader)
    else:
        trainer.test(net,dataloaders=testingLoader)
    pred,labels=net.TestResults
    pred,labels=torch.nan_to_num(pred).float(),labels.float()
    acc=SLE.Acc(pred,labels)
    print(NetName+' Acc:',acc)
    
    
    for i in list(range(5)):
        print('')
    print(NetName+' Test Confounding')
    testSet=MNISTDatset('test',masks=False,bias=True,confounding=True)
    testingLoader=Tdata.DataLoader(testSet,batch_size=4,shuffle=False,
                                   num_workers=1)
    if(hparams.train=='1'):
        trainer.test(dataloaders=testingLoader)
    else:
        trainer.test(net,dataloaders=testingLoader)
    pred,labels=net.TestResults
    pred,labels=torch.nan_to_num(pred).float(),labels.float()
    acc=SLE.Acc(pred,labels)
    print(NetName+' Acc:',acc)




if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('--train', default='1')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--local_rank', default=None)
    parser.add_argument('--devices', default='1')
    parser.add_argument('--nodes', default=1)
    parser.add_argument('--load', default=None)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--batch', default=128)
    parser.add_argument('--LrFinder', default=None)
    parser.add_argument('--name', default='ISNetMNIST')
    parser.add_argument('--continuing', default='0')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--bias', default='1')
    parser.add_argument('--workers', default='4')
    parser.add_argument('--lr', default='1e-2')

    parser.add_argument('--heat', default='1')
    parser.add_argument('--P', default='0.7')
    parser.add_argument('--cut', default='1')
    parser.add_argument('--cut2', default='25')
    parser.add_argument('--A', default='1')
    parser.add_argument('--B', default='3')
    parser.add_argument('--Ea', default='1')
    parser.add_argument('--d', default='0.9')
    parser.add_argument('--e', default='1e-2')
    parser.add_argument('--clip', default='1.0')
    parser.add_argument('--Zb', default='1')
    
    parser.add_argument('--selective', default='1')
    parser.add_argument('--highest', default='1')
    parser.add_argument('--rule', default='e')
    parser.add_argument('--random', default='0')
    
    parser.add_argument('--precision', default='32')
    parser.add_argument('--tuneCutEpochs', default='5')
    parser.add_argument('--multiple', default='0')
    parser.add_argument('--norm', default='1')
    parser.add_argument('--detach', default='1')
    parser.add_argument('--momentum', default='0.9')
    parser.add_argument('--alternativeForeground', default='L2')
    
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--dropout', default='0')
    parser.add_argument('--penalizeAll', default='0')
    parser.add_argument('--dLoss', default='0.8')
    
    
    args=parser.parse_args()

    main(args)