import torch
import torch.nn.functional as F
import torch.nn as nn
import ISNetLayersZe as ISNetLayers
import ISNetFunctionsZe as ISNetFunctions
import LRPDenseNetZe as LRPDenseNet
import globalsZe as globals
import pytorch_lightning as pl
import warnings
from collections import OrderedDict
import numpy as np
import copy
from torch.autograd import Variable
import torch.autograd as ag
import random
import warnings
import ISNetFlexTorch

class ISNetFlexLgt(pl.LightningModule):
    def __init__(self,model=None,architecture='densenet121',
                 heat=True,P=0.7,d=0,
                 optim='SGD',nesterov=False,
                 LR=1e-3,momentum=0.99,WD=0,dropLr=None,
                 clip=1,classes=1,dropout=False,
                 HiddenLayerPenalization=False,dLoss=1,
                 cut=1,cut2=25,selective=False,selectiveEps=1e-7,Zb=True,e=0.01,
                 multiple=False,randomLogit=False,
                 pretrained=False,A=1,B=1,E=1,VGGRemoveLastMaxpool=False,
                 alternativeForeground='hybrid',
                 norm='absRoIMean',
                 val='ISNet',
                 explainLabels=False):
        """
        PyTorch Lightning module based on LRP-Flex. For Stochastic, Original and Selective ISNets.
        
        Args:
            model: arbitrary PyTorch DNN to be converted, not restricted to resnet densenet or VGG.
            If None, model is as pre-defined backbone, according to architecture parameter
            architecture: pre-defined backbone architecture name. Either densenets, resnets or vggs
            heat: if True, produces heatmaps and applies ISNets' training scheme (background 
            relevance minimization)
            P: heatmap loss weight, between 0 and 1
            d: background loss GWRP exponential decay
            optim: optimizer, SGD or Adam
            nesterov: if True, uses Nesterov momentum
            LR: learning rate
            momentum: training momentum
            WD: weight decay
            dropLr: if not None, list of tuples (epoch,lr) for scheduler
            clip: gradient norm clipping value, set to None for no clip
            dropout: if True, adds dropout to pre-defined backbone
            classes: number of output neurons in pre-defined backbones
            dLoss: LDS exponential decay (GWRP)
            cut,cut2: C1 and C2 hyper-parameters
            selective: if True, explains the softmax-based quantity Eta, instead of a logit 
            (Selective ISNet)
            selectiveEps: epsilon used for stabilizing the relevance pass through softmax (see 
            Selective ISNet Section in paper)
            e: LRP epsilon parameter
            Zb: if True, uses LRP-Zb in the first DNN layer
            multiple: if True, produces a single heatmap per sample (Faster ISNet). If False,
            creates one heatmap per class per sample (original ISNet)
            randomLogit: if True, stochastically selects a single logit to be explained (Stochastic
            ISNet)
            HiddenLayerPenalization: LRP Deep Supervision
            pretrained: if True, pre-defined backbone is downloaded pre-trained (ImageNet)
            A and B: w1 and w2 parameters, weights for the background and foreground loss
            E: parameter of the background loss activation x/(x+E)
            VGGRemoveLastMaxpool: if True, removes last maxPool from VGG pre-defined backbone
            alternativeForeground='hybrid' adds the loss modification in the paper Faster ISNet
            for Background Bias Mitigation on Deep Neural Networks, alternativeForeground='L2'
            uses the standard (original) loss
            norm: 'absRoIMean' uses standard ISNet loss, 'RoIMean' applies the background loss
            normalization step before the absolute value operation
            explainLabels: used for explanation only, not for ISNet training. If True, 
            heatmaps explain the classes indicated by the argument labels in the forward pass
            val: loss to use in validation, heatmap, classification or ISNet
        """
        super (ISNetFlexLgt,self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        self.model=ISNetFlexTorch.ISNetFlex(model=model,architecture=architecture,
                 e=e,classes=classes,dropout=dropout,
                 HiddenLayerPenalization=HiddenLayerPenalization,
                 selective=selective,selectiveEps=selectiveEps,
                 Zb=Zb,multiple=multiple,randomLogit=randomLogit,
                 pretrained=pretrained,VGGRemoveLastMaxpool=VGGRemoveLastMaxpool,
                 explainLabels=explainLabels)
        
        self.heat=heat
        self.lr=LR
        self.optim=optim
        self.momentum=momentum
        self.WD=WD
        self.nesterov=nesterov
        self.clip=clip
        self.dropLr=dropLr
        self.penalizeAll=HiddenLayerPenalization
        self.dLoss=dLoss
        self.testLoss=False
        self.alternativeForeground=alternativeForeground
        self.norm=norm
        self.val=val
        
        if isinstance(P, dict):
            self.P=P[0]
            self.increaseP=P
            
        else:
            self.P=P
            self.increaseP=None
        self.d=d
        
        self.cut=cut
        self.cut2=cut2
        self.tuneCut=False
        self.A=A
        self.B=B
        self.E=E
            
    def forward(self,x,labels=None):
        return self.model(x,runLRPFlex=self.heat,labels=labels)

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
                                    weight_decay=self.WD,
                                    nesterov=self.nesterov)

        if(self.dropLr is not None):
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.dropLr,
                                                           verbose=True)            
            return [optimizer],[scheduler]
        else:
            return optimizer

    def compound_loss(self,out,labels,masks=None,norm=None):        
        if norm is None:
            norm=self.norm
        
        #print('self.model.maximum:',self.model.maximum)
        loss=ISNetFlexTorch.CompoundLoss(out,labels,
                masks=masks,tuneCut=self.tuneCut,
                d=self.d,dLoss=self.dLoss,
                cutFlex=self.cut,cut2Flex=self.cut2,
                A=self.A,B=self.B,E=self.E,
                alternativeForeground=self.alternativeForeground,
                norm=norm)
            
        
    
        if not self.heat or masks is None:
            return loss['classification']
        if not self.tuneCut:
            return loss['classification'],loss['LRPFlex']
        else:
            self.keys=list(loss['mapAbsFlex'].keys())
            return loss['classification'],loss['LRPFlex'],loss['mapAbsFlex']
    
    def training_step(self,train_batch,batch_idx):
        opt=self.optimizers()
        opt.zero_grad()
        
        inputs,masks,labels=train_batch
        
        if self.tuneCut:
            if(self.current_epoch!=self.cutEpochs):
                self.heat=False
                out=self.forward(inputs)
                loss=self.compound_loss(out,labels=labels)
            if(self.current_epoch==self.cutEpochs):
                self.heat=True
                out=self.forward(inputs)
                cLoss,hLoss,mapAbs=self.compound_loss(out,labels=labels,
                                                      masks=masks)
                #take only values from last tuning epoch
                self.updateCut(mapAbs)
                #use only for tuning cut value, ignores heatmap loss
                loss=cLoss
            
            self.log('train_loss',loss.detach(),                     
                     on_epoch=True)
        else:
            #update dinamic P
            if (self.increaseP is not None):
                epochs=list(self.increaseP.keys())
                epochs.sort()
                for epoch in epochs:
                    if (self.current_epoch>=epoch):
                        self.P=self.increaseP[epoch]

            #data format: channel first
            if (self.heat):#ISNet
                out=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(out,labels=labels,
                                               masks=masks)
                loss=(1-self.P)*cLoss+self.P*hLoss

                self.log('train_loss', {'Classifier':cLoss.detach(),
                                        'Heatmap':hLoss.detach(),
                                        'Sum':loss.detach()},                     
                         on_epoch=True)

            else:#Common DenseNet
                out=self.forward(inputs)
                loss=self.compound_loss(out,labels=labels)
                self.log('train_loss',loss,                     
                             on_epoch=True)
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')
        
        opt.zero_grad()
        self.manual_backward(loss)
        
        if self.clip is not None:
            if self.clip!=0:
                self.clip_gradients(opt, gradient_clip_val=self.clip,
                                    gradient_clip_algorithm="norm")
        opt.step()
        
    def on_train_epoch_start(self, training_step_outputs=None):  
        #lr step
        if self.dropLr:
            sch = self.lr_schedulers()
            sch.step()

    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        torch.set_grad_enabled(True)
        tmp=self.heat
        
        try:
            inputs,masks,labels=val_batch
        except:
            inputs,labels=val_batch
            
        if self.tuneCut:
            self.heat=False
        if dataloader_idx!=0:
            self.heat=False
            masks=None
        
        if (self.heat):#ISNet
            out=self.forward(inputs)
            norm=self.norm
            cLoss,hLoss=self.compound_loss(out,labels=labels,masks=masks,norm=norm)
            if self.val=='ISNet':
                loss=(1-self.P)*cLoss+self.P*hLoss
            elif self.val=='heatmap':
                loss=hLoss
            elif self.val=='classification':
                loss=cLoss
            else:
                raise ValueError('Unrecognized validation loss choice')
            self.log('val_loss', {'Classifier':cLoss.detach(),
                                  'Heatmap':hLoss.detach()},                     
                         on_epoch=True,on_step=False)
        else:#Common DenseNet
            logits=self.forward(inputs)
            loss=self.compound_loss(logits,labels=labels)
            
        self.heat=tmp
        if dataloader_idx==0:
            self.log('val_loss_iid', loss.detach(), on_step=True, on_epoch=True)
        if dataloader_idx==1:
            self.log('val_loss_ood', loss.detach(), on_step=True, on_epoch=True)
        if dataloader_idx==2:
            self.log('val_loss_cnf', loss.detach(), on_step=True, on_epoch=True)
        
        self.manual_backward(loss)
        opt=self.optimizers()
        opt.zero_grad()
        
    def test_step(self,test_batch,batch_idx):
        #data format: channel first
        inputs,labels=test_batch
        
        if self.testLoss:
            if self.heat:
                out=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(out,labels=labels,masks=masks)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(), 'hLoss': hLoss.detach()}
            else:
                out=self.forward(inputs)
                cLoss=self.compound_loss(out,labels=labels)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(),
                        'hLoss': torch.zeros(cLoss.shape).type_as(cLoss)}
        elif (self.heat):#ISNet
            out=self.forward(inputs)
            logits=out['output']
            heatmaps=out['LRPFlex']['input']
            return {'pred': logits.detach(), 'labels': labels,
                    'images': inputs.cpu().float().detach(),
                    'heatmaps': heatmaps.cpu().float().detach()}

        else:#Common DenseNet
            out=self.forward(inputs)
            logits=out['output']
            return {'pred': logits.detach(), 'labels': labels}
            

    def test_step_end(self, batch_parts):
        if(batch_parts['pred'].dim()>2):
            logits=batch_parts['pred']
            labels=batch_parts['labels']
            if (not self.heat):
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1])}
            elif self.testLoss:
                cLoss=batch_parts['cLoss']
                hLoss=batch_parts['hLoss']
                #print(cLoss.shape)
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1]),
                        'cLoss': cLoss.view(cLoss.shape[0]*cLoss.shape[1],cLoss.shape[-1]), 
                        'hLoss': hLoss.view(hLoss.shape[0]*hLoss.shape[1],hLoss.shape[-1])}
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
        if self.testLoss:
            cLoss=test_step_outputs[0]['cLoss'].unsqueeze(0)
            hLoss=test_step_outputs[0]['hLoss'].unsqueeze(0)
        elif (self.heat):
            images=test_step_outputs[0]['images']
            heatmaps=test_step_outputs[0]['heatmaps']
            
        for i,out in enumerate(test_step_outputs,0):
            if (i!=0):
                pred=torch.cat((pred,out['pred']),dim=0)
                labels=torch.cat((labels,out['labels']),dim=0)
                if self.testLoss:
                    cLoss=torch.cat((cLoss,out['cLoss'].unsqueeze(0)),dim=0)
                    hLoss=torch.cat((hLoss,out['hLoss'].unsqueeze(0)),dim=0)
                elif (self.heat):
                    images=torch.cat((images,out['images']),dim=0)
                    heatmaps=torch.cat((heatmaps,out['heatmaps']),dim=0)
                
        if self.testLoss:
            self.TestResults=pred,labels,cLoss.mean().item(),hLoss.mean().item()
        elif (self.heat):
            self.TestResults=pred,labels,images,heatmaps
        else:
            self.TestResults=pred,labels
            
    def returnBackbone(self):
        return self.model.returnBackbone()
    
    def initTuneCut(self,epochs):
        #train for self.cutEpochs to find cut values, do not use heatmap loss
        self.tuneCut=True
        self.cutEpochs=epochs-1
            
    def resetCut(self):
        self.aggregateE={}
        for name in self.keys:
            self.aggregateE[name]=[0,0,0]

    def updateWelford(self,existingAggregate,newValue):
        #https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)
    
    
    def updateCut(self,maps):
        if not hasattr(self, 'aggregateE'):
            self.resetCut()
        #print(maps)
        for layer in self.keys:
            mapAbs=maps[layer]
            
            mapAbsZ=mapAbs[:,:int(mapAbs.shape[1]/2)]
            mapAbsE=mapAbs[:,int(mapAbs.shape[1]/2):]


            for i,_ in enumerate(mapAbsE,0):#batch iteration
                valueE=torch.mean(mapAbsE[i].detach().float()).item()
                self.aggregateE[layer]=self.updateWelford(self.aggregateE[layer],valueE)



    def finalizeWelford(self,existingAggregate):
        # Retrieve the mean, variance and sample variance from an aggregate
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float("nan")
        else:
            mean, sampleVariance = mean, M2 / (count - 1)
            std=sampleVariance**(0.5)
            return mean, std
        
    def returnCut(self):
        self.tuneCut=False
        
        cut0={}
        cut1={}
        means={}
        stds={}
            
        for layer in self.keys:
            means[layer],stds[layer],cut0[layer],cut1[layer]=[],[],[],[]
            #order: Z, E, En
            
            
            mean,std=self.finalizeWelford(self.aggregateE[layer])
            means[layer].append(mean)
            stds[layer].append(std)
            c0=np.maximum(mean/5,mean-3*std)
            c1=np.minimum(c0*25,mean+3*std)
            cut0[layer].append(c0)
            cut1[layer].append(c1)

        return cut0,cut1,means,stds
