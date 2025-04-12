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
from torch.autograd import Variable
import torch.autograd as ag
import torchvision

import sys
sys.path.append('../')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class ISNetLgt(pl.LightningModule):
    def __init__(self,multiLabel=False,
                 classes=1000,architecture='densenet121',
                 heat=True,
                 pretrained=False,
                 LR=1e-3,P=2,E=10,
                 A=1,B=3,d=0.996,
                 cut=1,cut2=25,
                 norm=True,optim='SGD',
                 Cweight=None,
                 dropLr=None, baseModel=None,
                 dropout=False,
                 momentum=0.99,WD=0,
                 penalizeAll=False,
                 dLoss=1, gradientMode='logits',
                 clip=1,
                 VGGRemoveLastMaxpool=False,
                 multiply=True,
                 alternativeForeground='hybrid'):
        """
        PyTorch Lightning ISNet Softmax Grad*Input.
        
        Args:
            multiLabel: for multi-label problems
            classes: number of output classes
            architecture: densenet,resnet,VGG or Sequential
            heat: allows relevance propagation and heatmap creation. If False,
            no signal is propagated through LRP block.
            pretrained: if a pretrained DenseNet121 shall be downloaded
            LR:learning rate, list of tuples (epoch,lr) for scheduler
            P: loss balancing hyperparameter. int or dictionary, with epochs (int) and P (float)
            E: heatmap loss hyperparameter
            A and B: w1 and w2 parameters, weights for the background and foreground loss
            d: background loss GWRP exponential decay
            cut,cut2: C1 and C2 hyper-parameters
            norm: 'absRoIMean' uses standard ISNet loss, 'RoIMean' applies the background loss
            normalization step before the absolute value operation
            optim: SGD or Adam
            Cweight: BCE loss weights to deal with unbalanced datasets
            dropLr: if not None, list of tuples (epoch,lr) for scheduler
            baseModel: instantiated backbone, overrides architecture parameter
            dropout:adds dropout before last layer
            momentum: training momentum
            WD: weight decay
            penalizeAll: if True, applies LRP Deep Supervision
            dLoss: LDS exponential decay (GWRP)
            gradientMode: quantity from which gradients will be propagated. Set to softmax
            for ISNet Softmax Grad*Input, and to logits for ISNet Grad*Input
            clip: gradient norm clipping value, set to None for no clip
            VGGRemoveLastMaxpool: if True, removes last MaxPool of VGG backbones
            multiply: if True, penalizes Gradient*Input, else, penalizes only gradients
            alternativeForeground='hybrid' adds the loss modification in the paper Faster ISNet
            for Background Bias Mitigation on Deep Neural Networks, alternativeForeground='L2'
            uses the standard (original) loss
        """
        super (ISNetLgt,self).__init__()
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
            elif (architecture=='vgg11'):
                self.classifierDNN=torchvision.models.vgg11(pretrained=pretrained,
                                                            num_classes=classes)
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
            elif (architecture=='vgg11_bn'):
                self.classifierDNN=torchvision.models.vgg11_bn(pretrained=pretrained,
                                                               num_classes=classes)    
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
            elif (architecture=='vgg16'):
                self.classifierDNN=torchvision.models.vgg16(pretrained=pretrained,
                                                            num_classes=classes) 
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[30]=torch.nn.Identity()
            elif (architecture=='vgg16_bn'):
                self.classifierDNN=torchvision.models.vgg16_bn(pretrained=pretrained,
                                                               num_classes=classes) 
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
            elif (architecture=='vgg13'):
                self.classifierDNN=torchvision.models.vgg13(pretrained=pretrained,
                                                            num_classes=classes) 
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
            elif (architecture=='vgg13_bn'):
                self.classifierDNN=torchvision.models.vgg13_bn(pretrained=pretrained,
                                                               num_classes=classes) 
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
            elif (architecture=='vgg19'):
                self.classifierDNN=torchvision.models.vgg19(pretrained=pretrained,
                                                            num_classes=classes) 
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
                    
            elif (architecture=='vgg19_bn'):
                self.classifierDNN=torchvision.models.vgg19_bn(pretrained=pretrained,
                                                               num_classes=classes) 
                if VGGRemoveLastMaxpool:
                    self.classifierDNN.features[-1]=torch.nn.Identity()
            else:
                raise ValueError('Architecture must be densenet121, 161, 169, 201 or 204; or resnet18, 34, 50, 101 or 152; or  wideResnet28; orr vgg 11, 13, 16, 19. User may also supply a DenseNet or ResNet as baseModel.')
        
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
                raise ValueError('Unrecognized backbone or vgg should have dropout false')

        ISNetFunctions.ChangeInplace(self.classifierDNN)
        
        #add hooks for gradients
        if heat and penalizeAll:
            if ('ResNet' in self.classifierDNN.__class__.__name__):
                self.storedRelevance={}
                self.storedRelevance['Classifier']=ISNetFunctions.HookFwd(self.classifierDNN.avgpool,
                                                                          'input')
                for name, module in self.classifierDNN.named_modules():
                    if name.count('.')==1:#bottlenecks or basic blocks
                        self.storedRelevance[name]=ISNetFunctions.HookFwd(module,'output')
            elif ('DenseNet' in self.classifierDNN.__class__.__name__):
                self.storedRelevance={}
                self.storedRelevance['Classifier']=ISNetFunctions.HookFwd(
                    self.classifierDNN.AdaptPool,'input')
                for name, module in self.classifierDNN.named_modules():
                    if 'transition' and name.count('.')==1:
                        self.storedRelevance[name]=ISNetFunctions.HookFwd(module,'output')
            else:
                raise ValueError('penalizeAll not implemented for backbone')
                        
        self.gradientMode=gradientMode
        self.heat=heat
        self.lr=LR
        self.P=P
        self.E=E
        self.multiLabel=multiLabel
        self.cut=cut
        self.cut2=cut2
        print('RECEIVED CUT OF:',self.cut,self.cut2)
        self.A=A
        self.B=B
        self.d=d
        self.Cweight=Cweight
        self.criterion=nn.CrossEntropyLoss()
        self.norm=norm
        self.clip=clip
        self.multiply=multiply
        self.optim=optim
        self.classes=classes
        self.dropLr=dropLr
        self.momentum=momentum
        self.WD=WD
        self.penalizeAll=penalizeAll
        self.dLoss=dLoss
        self.alternativeForeground=alternativeForeground
        
        if isinstance(P, dict):
            self.P=P[0]
            self.increaseP=P
            
        else:
            self.P=P
            self.increaseP=None
            
        self.tuneCut=False
        self.displayMode=False
        
        
    def forward(self,x):
        #x:input
        if not self.displayMode:
            y=self.classifierDNN(x)
            return y
        else:
            y,grads=self.getGrad(x)
            return y,grads

    def getGrad(self,x):
        #for printing heatmaps
        x=Variable(x, requires_grad=True)
        y=self.classifierDNN(x)
        
        grads=[]
        for i in list(range(y.shape[-1])):
            logits=y[:,i].mean()
            heatmaps=ag.grad(logits, x, retain_graph=True)[0]#, create_graph=True)
            grads.append(heatmaps)
        grads=torch.stack(grads,1)
        
        #print(y.shape,grads.shape)
        return y,grads
        
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
                
        
                
        if not self.heat or masks is None:
            return classifierLoss
        
        #heatmaps=inputs.grad.clone()
        outs={'input':inputs}
        if self.penalizeAll:
            ISNetFunctions.getRelevance(self.storedRelevance,outs,'')
            #print(list(outs.keys()))
            
        #get gradients:
        values=[]
        for key in outs:
            values.append(outs[key])
            
        if self.gradientMode=='logits':
            #one heatmap per logit
            grads={}
            for key in list(outs.keys()):
                grads[key]=[]
            for i in list(range(outputs.shape[-1])):
                logits=outputs[:,i].mean()
                heatmaps=ag.grad(logits, values, retain_graph=True, create_graph=True)
                for i,H in enumerate(heatmaps,0):
                    #print(H.shape)
                    grads[list(outs.keys())[i]].append(H)#*outs[list(outs.keys())[i]])
                    #print('here')
            for key in list(outs.keys()):
                grads[key]=torch.stack(grads[key],1)
            
        else:
            if self.gradientMode=='softmax':
                softmax=torch.nn.functional.softmax(outputs, dim=-1)
                softmax=torch.amax(softmax,dim=-1).mean()
                heatmaps=ag.grad(softmax, values, retain_graph=True, create_graph=True)
            else:
                raise ValueError('not implemented gradientMode')
                
            grads={}
            for i,H in enumerate(heatmaps,0):
                grads[list(outs.keys())[i]]=torch.nan_to_num(H)
            
        losses=[]
        tune={}
        for key in grads:
            if (self.heat and self.tuneCut):
                if self.multiply:
                    if self.gradientMode!='logits':
                        grad=(grads[key]*outs[key]).unsqueeze(1)
                    else:
                        tmp=outs[key].unsqueeze(1).repeat(1,grads[key].shape[1],1,1,1)
                        grad=grads[key]*tmp
                else:
                    if self.gradientMode!='logits':
                        grad=grads[key].unsqueeze(1)
                    else:
                        grad=grads[key]
                heatmapLoss,foreg=ISNetFunctions.LRPLossCEValleysGWRP(grad,masks,
                                                               A=self.A,B=self.B,d=self.d,
                                                               E=self.E,
                                                               rule='e',
                                                               tuneCut=self.tuneCut,
                                                               norm=self.norm,
                                               alternativeForeground=self.alternativeForeground)

                losses.append(heatmapLoss)
                tune[key]=foreg

            if (self.heat and not self.tuneCut):
                if self.multiply:
                    if self.gradientMode!='logits':
                        grad=(grads[key]*outs[key]).unsqueeze(1)
                    else:
                        tmp=outs[key].unsqueeze(1).repeat(1,grads[key].shape[1],1,1,1)
                        grad=grads[key]*tmp
                else:
                    if self.gradientMode!='logits':
                        grad=grads[key].unsqueeze(1)
                    else:
                        grad=grads[key]
                heatmapLoss=ISNetFunctions.LRPLossCEValleysGWRP(grad,masks,
                                                               cut=self.cut[key],
                                                               cut2=self.cut2[key],
                                                               A=self.A,B=self.B,d=self.d,
                                                               E=self.E,
                                                               rule='e',
                                                               tuneCut=self.tuneCut,
                                                               alternativeForeground=self.alternativeForeground)
                
                losses.append(heatmapLoss)
            
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=ISNetFunctions.GlobalWeightedRankPooling(heatmapLoss,d=self.dLoss)
        self.keys=list(grads.keys())
        if not self.tuneCut:
            return classifierLoss,heatmapLoss
        else:
            return classifierLoss,heatmapLoss,tune

    
    def training_step(self,train_batch,batch_idx):
        opt=self.optimizers()
        opt.zero_grad()
        
        if self.tuneCut:
            inputs,masks,labels=train_batch
            if(self.current_epoch!=self.cutEpochs):
                self.heat=False
                logits=self.forward(inputs)
                loss=self.compound_loss(logits,labels=labels)
            if(self.current_epoch==self.cutEpochs):
                self.heat=True
                inputs=Variable(inputs, requires_grad=True)
                logits=self.forward(inputs)
                cLoss,hLoss,mapAbs=self.compound_loss(logits,labels=labels,
                                           inputs=inputs,masks=masks)
                #take only values from last tuning epoch
                self.updateCut(mapAbs)
                #use only for tuning cut value, ignores heatmap loss
                loss=cLoss
            
            self.log('train_loss',loss,                     
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
                inputs,masks,labels=train_batch
                inputs=Variable(inputs, requires_grad=True)
                logits=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                               inputs=inputs,masks=masks)
                loss=(1-self.P)*cLoss+self.P*hLoss
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
        self.clip_gradients(opt, gradient_clip_val=self.clip, gradient_clip_algorithm="norm")
        opt.step()
            
        return loss
    
    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        torch.set_grad_enabled(True)
        if self.tuneCut:
            self.heat=False
        if dataloader_idx==1:
            tmp=self.heat
            self.heat=False
        #data format: channel first
        if ((self.heat or self.tuneCut) and dataloader_idx==0):
            inputs,masks,labels=val_batch
        else:
            inputs,labels=val_batch
        
        if (self.heat):#ISNet
            inputs=Variable(inputs, requires_grad=True)
            logits=self.forward(inputs)
            cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                           inputs=inputs,masks=masks,
                                           val=True)
            loss=(1-self.P)*cLoss+self.P*hLoss
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
            mapAbsE=mapAbs

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



class GradHook():
    def __init__(self, module):
        self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, grad_in, grad_out):
        self.output = grad_out[0].clone()
    def close(self):
        self.hook.remove()

        

