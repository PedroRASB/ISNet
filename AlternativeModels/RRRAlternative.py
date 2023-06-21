import torch
import torch.nn.functional as F
import torch.nn as nn
import ISNetFunctions
import LRPDenseNet
import pytorch_lightning as pl
import warnings
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
import torch.autograd as ag

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class RRRLgt(pl.LightningModule):
    def __init__(self,multiLabel=False,multiMask=False,
                 classes=1000,architecture='densenet121',
                 heat=True,
                 pretrained=False,
                 LR=1e-3,P=0.5,optim='SGD',
                 Cweight=None,
                 dropLr=None, baseModel=None,
                 dropout=False,
                 momentum=0.99,WD=0,
                 clip=1):

        super (RRRLgt,self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        if (baseModel==None):
            if (architecture=='densenet121'):
                self.classifierDNN=LRPDenseNet.densenet121(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet161'):
                self.classifierDNN=LRPDenseNet.densenet161(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet169'):
                self.classifierDNN=LRPDenseNet.densenet169(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet201'):
                self.classifierDNN=LRPDenseNet.densenet201(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet264'):
                if(pretrained):
                    raise ValueError('No available pretrained densenet264')
                self.classifierDNN=LRPDenseNet.densenet264(pretrained=False,
                                                      num_classes=classes)
            elif (architecture=='resnet18'):
                self.classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                             'resnet18', pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet34'):
                self.classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                             'resnet34', pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet50'):
                self.classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                             'resnet50', pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet101'):
                self.classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                             'resnet101', pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet152'):
                self.classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                             'resnet152', pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='wideResnet28'):
                if(pretrained):
                    raise ValueError('No available pretrained wideResnet28')
                import WideResNet
                self.classifierDNN=WideResNet.WideResNet28(num_classes=classes)
            else:
                raise ValueError('Architecture must be densenet121, 161, 169, 201 or 204; or resnet18, 34, 50, 101 or 152; or  wideResnet28. User may also supply a DenseNet or ResNet as baseModel.')
        
        else:
            self.classifierDNN=baseModel
            
        if dropout:
            if ('DenseNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.classifier=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                       self.classifierDNN.classifier)
            elif ('ResNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.fc=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                       self.classifierDNN.fc)
            else:
                raise ValueError('Unrecognized backbone')

        ISNetFunctions.ChangeInplace(self.classifierDNN)
            
        self.heat=heat
        self.lr=LR
        self.P=P
        self.multiMask=multiMask
        self.multiLabel=multiLabel
        self.criterion=nn.CrossEntropyLoss()
        self.clip=clip
        
        self.optim=optim
        self.classes=classes
        self.dropLr=dropLr
        self.momentum=momentum
        self.WD=WD
        
        if isinstance(P, dict):
            self.P=P[0]
            self.increaseP=P
            
        else:
            self.P=P
            self.increaseP=None
            
        
        
    def forward(self,x):
        #x:input
        y=self.classifierDNN(x)
        return y

    def configure_optimizers(self):
        if (self.optim=='Adam'):
            from deepspeed.ops.adam import FusedAdam
            optimizer=FusedAdam(filter(
                    lambda p: p.requires_grad,
                                        self.parameters()),
                                        lr=self.lr)
        else:
            optimizer=torch.optim.SGD(filter(
                lambda p: p.requires_grad,
                                    self.parameters()),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.WD)

        if(self.dropLr is not None):
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.dropLr,
                                                           verbose=True)            
            return [optimizer],[scheduler]
        else:
            return optimizer
    
    def compound_loss(self,outputs,labels,inputs=None,masks=None,val=False):
        if(self.multiLabel):
            classifierLoss=F.binary_cross_entropy_with_logits(outputs,labels,
                                                              pos_weight=self.Cweight)
        else:
            #classifierLoss=F.cross_entropy(outputs,labels.squeeze(1))
            try:
                classifierLoss=self.criterion(outputs,labels.squeeze(1))
            except:
                classifierLoss=self.criterion(outputs,labels)
                
                
        if not self.heat:
            return classifierLoss
        
        softmax=torch.nn.functional.softmax(outputs, dim=-1)
        lsoft=torch.log(softmax).sum(-1).mean(0)
        grad=ag.grad(lsoft, inputs, retain_graph=True, create_graph=True)
        grad=torch.nan_to_num(grad[0])
        heatmapLoss=RRRHeatmapLoss(grad,masks)
        return classifierLoss.mean(),heatmapLoss.mean()
        
    
    def training_step(self,train_batch,batch_idx):
        opt=self.optimizers()
        opt.zero_grad()
        
       
        #update dinamic P
        if (self.increaseP is not None):
            epochs=list(self.increaseP.keys())
            epochs.sort()
            for epoch in epochs:
                if (self.current_epoch>=epoch):
                    self.P=self.increaseP[epoch]

        #data format: channel first
        if (self.heat):
            inputs,masks,labels=train_batch
            inputs=Variable(inputs, requires_grad=True)
            logits=self.forward(inputs)
            cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                           inputs=inputs,masks=masks)
            loss=cLoss+self.P*hLoss
            self.log('train_loss', {'Classifier':cLoss,
                                    'Heatmap':hLoss,
                                    'Sum':loss},                     
                     on_epoch=True)

        else:#Common DenseNet
            inputs,labels=train_batch
            logits=self.forward(inputs)
            loss=self.compound_loss(logits,labels=labels)
            self.log('train_loss',loss,                     
                     on_epoch=True)
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')
            
        
        # clip gradients
        opt.zero_grad()
        self.manual_backward(loss)
        if self.clip is not None:
            if self.clip!=0:
                self.clip_gradients(opt, gradient_clip_val=self.clip, gradient_clip_algorithm="norm")
        opt.step()
            
        return loss
    
    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        torch.set_grad_enabled(True)
        if dataloader_idx==1:
            tmp=self.heat
            self.heat=False
        #data format: channel first
        if (self.heat and dataloader_idx==0):
            inputs,masks,labels=val_batch
        else:
            inputs,labels=val_batch
        
        if (self.heat):
            inputs=Variable(inputs, requires_grad=True)
            logits=self.forward(inputs)
            cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                           inputs=inputs,masks=masks,
                                           val=True)
            loss=cLoss+self.P*hLoss
        else:#Common DenseNet
            logits=self.forward(inputs)
            loss=self.compound_loss(logits,labels=labels)
            
        opt=self.optimizers()
        opt.zero_grad()
        loss=loss.detach()
            
        if dataloader_idx==0:
            return {'iidLoss':loss}
        if dataloader_idx==1:
            self.heat=tmp
            return {'oodLoss':loss}

    def validation_step_end(self, batch_parts):
        
        if 'iidLoss' in list(batch_parts.keys()):
            lossType='iidLoss'
        elif 'oodLoss' in list(batch_parts.keys()):
            lossType='oodLoss'
        else:
            raise ValueError('Unrecognized loss')
            
        if(batch_parts[lossType].dim()>1):
            losses=batch_parts[lossType]
            return {lossType: torch.mean(losses,dim=0)}
        else:
            return batch_parts

    def validation_epoch_end(self, validation_step_outputs):
        for item in validation_step_outputs:
            try:
                lossType=list(item[0].keys())[0]
                loss=item[0][lossType].unsqueeze(0)
            except:
                lossType=list(item.keys())[0]
                loss=item[lossType].unsqueeze(0)
            for i,out in enumerate(item,0):
                if(i!=0):
                    loss=torch.cat((loss,out[lossType].unsqueeze(0)),dim=0)
            self.log('val_'+lossType,torch.mean(loss,dim=0),
                     on_epoch=True,sync_dist=True)
    
    def test_step(self,test_batch,batch_idx):
        #data format: channel first
        inputs,labels=test_batch
        logits=self.forward(inputs)
        return {'pred': logits, 'labels': labels}
            

    def test_step_end(self, batch_parts):
        if(batch_parts['pred'].dim()>2):
            logits=batch_parts['pred']
            labels=batch_parts['labels']
            if (not self.heat):
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1])}
            else:
                images=batch_parts['images']
                heatmaps=batch_parts['heatmaps']
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[2]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1]),
                        'images': images.view(images.shape[0]*images.shape[1],images.shape[2],
                                              images.shape[3],images.shape[4]),
                        'heatmaps': heatmaps.view(heatmaps.shape[0]*heatmaps.shape[1],
                                                  heatmaps.shape[2],
                                                  heatmaps.shape[3],heatmaps.shape[4],
                                                  heatmaps.shape[5])}
        else:
            return batch_parts

    def test_epoch_end(self, test_step_outputs):
        pred=test_step_outputs[0]['pred']
        labels=test_step_outputs[0]['labels']
        if (self.heat):
            images=test_step_outputs[0]['images']
            heatmaps=test_step_outputs[0]['heatmaps']
            
        for i,out in enumerate(test_step_outputs,0):
            if (i!=0):
                pred=torch.cat((pred,out['pred']),dim=0)
                labels=torch.cat((labels,out['labels']),dim=0)
                if (self.heat):
                    images=torch.cat((images,out['images']),dim=0)
                    heatmaps=torch.cat((heatmaps,out['heatmaps']),dim=0)
                
        if (self.heat):
            self.TestResults=pred,labels,images,heatmaps
        else:
            self.TestResults=pred,labels
            
    def returnBackbone(self):
        model=self.classifierDNN
        ISNetFunctions.remove_all_forward_hooks(model)
        return model
    

        
def RRRHeatmapLoss(grad,mask):
    #mask is 1 at foreground and 0 at background
    Imask=1-mask
    x=torch.mul(grad,Imask)
    x=x**2
    x=x.sum(dim=(-1,-2,-3))#sum image dimensions
    return x
