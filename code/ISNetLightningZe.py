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
import torchvision
import sys
sys.path.append('../')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class ISNetLgt(pl.LightningModule):
    def __init__(self,multiLabel=False,
                 classes=1000,architecture='densenet121',
                 e=1e-2,heat=True,
                 Zb=True,pretrained=False,
                 LR=1e-3,P=0.7,E=10,
                 A=1,B=3,d=0.996,
                 cut=1,cut2=25,
                 norm=True,optim='SGD',
                 Cweight=None,rule='e',multiple=True,
                 ignore=None,
                 dropLr=None, selective=False,
                 highest=False, baseModel=None,
                 dropout=False,
                 momentum=0.99,WD=0,
                 detach=True,
                 penalizeAll=False,
                 FSP=False,dLoss=1,
                 labelLogit=False, randomLogit=False,
                 SequentialInputShape=None,SequentialPreFlattenShape=None,
                 channelGWRP=1.0,VGGRemoveLastMaxpool=False,
                 normLayer='batchNorm',nesterov=False,
                 alternativeForeground='hybrid'):
        """
        PyTorch Lightning ISNet based on LRP-Block.
        
        Args:
            baseModel: instantiated backbone, overrides architecture parameter
            e: LRP-e term
            Zb: allows Zb rule. If false, will use traditional LRP-e.
            heat: allows relevance propagation and heatmap creation. If False,
            no signal is propagated through LRP block.
            pretrained: if a pretrained DenseNet121 shall be downloaded
            classes: number of output classes
            rule: LRP rule, choose e, z+, AB or z+e. For background relevance 
            minimization use either e or z+e.
            multiple: whether to produce a single heatmap or one per class
            ignore: list with classes which will not suffer attention control
            selective: uses class selective propagation
            architecture:  densenet,resnet,VGG or Sequential
            if not None: standard resnet or densenet to be converted
            dropout: adds dropout before last layer
            channelGWRP: gwrp decay for reducing channel dimension in background heatmap loss
            LR:learning rate, list of tuples (epoch,lr) for scheduler
            P: loss balancing hyperparameter. int or dictionary, with epochs (int) and P (float)
            E: heatmap loss hyperparameter
            multiLabel: for multi-label problems
            optim: SGD or Adam
            Cweight: BCE loss weights to deal with unbalanced datasets
            validation loader should return dataset identifier (0=iid,1=ood) with each label
            dropLr: if not None, list of tuples (epoch,lr) for scheduler
            meanMaps: standard value for heatmap sums
            norm: 'absRoIMean' uses standard ISNet loss, 'RoIMean' applies the background loss
            normalization step before the absolute value operation
            highest: set to true for selective ISNet
            momentum: training momentum
            WD: weight decay
            detach: detach biases from graph when propagating relevances
            penalizeAll: if True, applies LRP Deep Supervision
            FSP: if True, applies LRP Deep Supervision supervising all DNN layers
            dLoss: LDS exponential decay (GWRP)
            labelLogit: LRP heatmaps explain ground truth classes, not used in ISNet training
            randomLogit: if True, stochastically selects a single logit to be explained (Stochastic
            ISNet)
            SequentialInputShape: input shape, needed for training simple sequential DNNs
            (Sequential architecture)
            SequentialPreFlattenShape: shape before flatten operation, needed for training
            simple sequential DNNs (Sequential architecture)
            VGGRemoveLastMaxpool: if True, removes last MaxPool of VGG backbones
            normLayer: only batchNorm supported
            nesterov: if True, employs nesterov momentum
            alternativeForeground='hybrid' adds the loss modification in the paper Faster ISNet
            for Background Bias Mitigation on Deep Neural Networks, alternativeForeground='L2'
            uses the standard (original) loss
            A and B: w1 and w2 parameters, weights for the background and foreground loss
            d: background loss GWRP exponential decay
            cut,cut2: C1 and C2 hyper-parameters
        """
        super (ISNetLgt,self).__init__()
        self.save_hyperparameters()
        
        if (ignore is not None and selective):
            raise ValueError('class ignore not implemented for selective output')
        
        if (baseModel==None):
            if pretrained:
                classesBackup=classes
                classes=1000
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
            
        if pretrained:
            classes=classesBackup
            if ('DenseNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.classifier=nn.Linear(self.classifierDNN.classifier.in_features,
                                                        classes)
            elif ('ResNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.fc=nn.Linear(self.classifierDNN.fc.in_features,classes)
            elif ('vgg' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.classifier[-1]=nn.Linear(
                    self.classifierDNN.classifier[-1].in_features,classes)
            else:
                raise ValueError('Unrecognized backbone')
            
        if dropout:
            if ('DenseNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.classifier=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                       self.classifierDNN.classifier)
            elif ('ResNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.fc=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                       self.classifierDNN.fc)
            else:
                raise ValueError('Unrecognized backbone')
                

        if normLayer!='batchNorm':
            ISNetFunctions.ChangeNorm(self.classifierDNN,normLayer)
            if heat:
                raise ValueError('LRP block not implemented for instance norm or layer norm')

        #print()
        #print(self.classifierDNN)
        if (normLayer=='batchNorm' or normLayer=='instanceNorm'):#temporary
            #LRp block
            if ('DenseNet' in self.classifierDNN.__class__.__name__):
                self.LRPBlock=ISNetLayers._LRPDenseNet(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                       multiple=multiple,positive=False,
                                                       ignore=ignore,selective=selective,
                                                       highest=highest,detach=detach,
                                                       storeRelevance=penalizeAll,
                                                       FSP=FSP,randomLogit=randomLogit)
            elif ('ResNet' in self.classifierDNN.__class__.__name__):
                #if FSP:
                #    raise ValueError('not implemented')
                if (hasattr(self.classifierDNN, 'maxpool') and \
                self.classifierDNN.maxpool.__class__.__name__=='MaxPool2d'):
                    self.classifierDNN.maxpool.return_indices=True
                    self.classifierDNN.maxpool=nn.Sequential(OrderedDict([('maxpool',
                                                                           self.classifierDNN.maxpool),
                                                        ('special',ISNetLayers.IgnoreIndexes())]))
                self.LRPBlock=ISNetLayers._LRPResNet(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                       multiple=multiple,positive=False,
                                                       ignore=ignore,selective=selective,
                                                       highest=highest,
                                                       amplify=1,detach=detach,
                                                       storeRelevance=penalizeAll,
                                                       FSP=FSP,randomLogit=randomLogit)
            elif ('Sequential' in self.classifierDNN.__class__.__name__):
                self.LRPBlock=ISNetLayers._LRPSequential(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                       multiple=multiple,selective=selective,
                                                       highest=highest,
                                                       amplify=1,detach=detach,
                                                       storeRelevance=penalizeAll,
                                                       inputShape=SequentialInputShape,
                                                       preFlattenShape=SequentialPreFlattenShape,
                                                       randomLogit=randomLogit)

            elif('VGG' in self.classifierDNN.__class__.__name__):
                if (positive or ignore):
                    raise ValueError('not implemented')
                self.LRPBlock=ISNetLayers._LRPVGG(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                       multiple=multiple,selective=selective,
                                                       highest=highest,
                                                       amplify=1,detach=detach,
                                                       randomLogit=randomLogit,
                                                       storeRelevance=penalizeAll)

            else:
                raise ValueError('Unrecognized backbone')
            
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
        self.channelGWRP=channelGWRP
        self.optim=optim
        self.rule=rule
        self.classes=classes
        self.dropLr=dropLr
        self.momentum=momentum
        self.WD=WD
        self.penalizeAll=penalizeAll
        self.FSP=FSP
        self.dLoss=dLoss
        self.labelLogit=labelLogit
        if labelLogit:
            print('WARNING: LABEL LOGIT SET TO TRUE, DO NOT TRAIN ISNET WITH LABEL LOGIT')
        self.keys={}
        self.testLoss=False
        self.nesterov=nesterov
        self.alternativeForeground=alternativeForeground
        
        self.valEpochCLoss=0
        self.valEpochHLoss=0
        self.loggedHLoss=[]
        self.loggedCLoss=[]
        self.valCount=0
        
        if isinstance(P, dict):
            self.P=P[0]
            self.increaseP=P
            
        else:
            self.P=P
            self.increaseP=None
            
        self.tuneCut=False
        
        
        
    def forward(self,x,label=None):
        #x:input
        
        y=self.classifierDNN(x)
        if(self.heat):#only run the LRP block if self.heat==True
            R=self.LRPBlock(x=x,y=y,label=label)
            return y,R
        else: 
            globals.X=[]
            globals.XI=[]
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
                                    weight_decay=self.WD,
                                    nesterov=self.nesterov)

        if(self.dropLr is not None):
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.dropLr,
                                                           verbose=True)            
            return [optimizer],[scheduler]
        else:
            return optimizer
    
    def compound_loss(self,outputs,labels,heatmaps=None,masks=None):
        
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
        
        
        LRPs={'input':heatmaps}
        #penalize heatmaps for multiple layers:
        if self.penalizeAll and not self.FSP:
            ISNetFunctions.getRelevance(self.LRPBlock.storedRelevance,LRPs,'')
        elif self.FSP:
            for key in self.LRPBlock.storedRelevance:
                    LRPs[key]=self.LRPBlock.storedRelevance[key].output
        self.keys=list(LRPs.keys())
        losses=[]
        tune={}
        for key in LRPs:
            if (self.heat and self.tuneCut):
                heatmapLoss,foreg=ISNetFunctions.LRPLossCEValleysGWRP(LRPs[key],masks,
                                                               A=self.A,B=self.B,d=self.d,
                                                               E=self.E,
                                                               rule=self.rule,
                                                               tuneCut=self.tuneCut,
                                                               norm=self.norm,
                                                               channelGWRP=self.channelGWRP,
                                           alternativeForeground=self.alternativeForeground)
                losses.append(heatmapLoss)
                tune[key]=foreg

            if (self.heat and not self.tuneCut):
                heatmapLoss=ISNetFunctions.LRPLossCEValleysGWRP(LRPs[key],masks,
                                                           cut=self.cut[key],
                                                           cut2=self.cut2[key],
                                                           A=self.A,B=self.B,d=self.d,
                                                           E=self.E,
                                                           rule=self.rule,
                                                           tuneCut=self.tuneCut,
                                                           channelGWRP=self.channelGWRP,
                                                           norm=self.norm,
                                           alternativeForeground=self.alternativeForeground)
                
                losses.append(heatmapLoss)
            
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=ISNetFunctions.GlobalWeightedRankPooling(heatmapLoss,d=self.dLoss)
        #heatmapLoss=torch.mean(heatmapLoss,dim=-1)
        if not self.tuneCut:
            return classifierLoss,heatmapLoss
        else:
            return classifierLoss,heatmapLoss,tune

    
    def training_step(self,train_batch,batch_idx):
        if self.tuneCut:
            inputs,masks,labels=train_batch
            if(self.current_epoch!=self.cutEpochs):
                self.heat=False
                logits=self.forward(inputs)
                loss=self.compound_loss(logits,labels=labels)
            if(self.current_epoch==self.cutEpochs):
                self.heat=True
                if self.labelLogit:
                    logits,heatmaps=self.forward(inputs,labels)
                else:
                    logits,heatmaps=self.forward(inputs)
                cLoss,hLoss,mapAbs=self.compound_loss(logits,labels=labels,
                                           heatmaps=heatmaps,masks=masks)
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

            if (self.heat):#ISNet
                inputs,masks,labels=train_batch
                if self.labelLogit:
                    logits,heatmaps=self.forward(inputs,labels)
                else:
                    logits,heatmaps=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                               heatmaps=heatmaps,masks=masks)
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
        return loss
    
    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        if self.tuneCut:
            self.heat=False
        if dataloader_idx==1:
            tmp=self.heat
            self.heat=False
        #data format: channel first
        if ((self.heat or self.tuneCut) and dataloader_idx==0):
            inputs,masks,labels=val_batch
            #print(inputs.shape,masks.shape)
        else:
            inputs,labels=val_batch
        
        if (self.heat):#ISNet
            logits,heatmaps=self.forward(inputs)
            cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                           heatmaps=heatmaps,masks=masks)
            loss=(1-self.P)*cLoss+self.P*hLoss
            self.valEpochCLoss+=cLoss.item()
            self.valEpochHLoss+=hLoss.item()
            self.valCount+=1
        else:#Common DenseNet
            logits=self.forward(inputs)
            loss=self.compound_loss(logits,labels=labels)
            
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
            
        self.valEpochHLoss=0
        self.valEpochCLoss=0
        self.valCount=0
    
    def test_step(self,test_batch,batch_idx):
        #data format: channel first
        
        if self.testLoss:
            if self.heat:
                inputs,masks,labels=test_batch
                logits,heatmaps=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                               heatmaps=heatmaps,masks=masks)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(), 'hLoss': hLoss.detach()}
            else:
                inputs,labels=test_batch
                logits=self.forward(inputs)
                cLoss=self.compound_loss(logits,labels=labels)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(),
                        'hLoss': torch.zeros(cLoss.shape).type_as(cLoss)}
        elif (self.heat):#ISNet
            inputs,masks,labels=test_batch
            logits,heatmaps=self.forward(inputs)
            return {'pred': logits.detach(), 'labels': labels,
                    'images': inputs.cpu().float().detach(),
                    'heatmaps': heatmaps.cpu().float().detach()}

        else:#Common DenseNet
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

        if self.rule=='z+e':
            self.aggregateZ={}
            for name in self.keys:
                self.aggregateZ[name]=[0,0,0]
                    
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

            if self.rule=='z+e':
                mapAbsZ=mapAbs[:,:int(mapAbs.shape[1]/2)]
                mapAbsE=mapAbs[:,int(mapAbs.shape[1]/2):]
            else:
                mapAbsE=mapAbs

            for i,_ in enumerate(mapAbsE,0):#batch iteration
                valueE=torch.mean(mapAbsE[i].detach().float()).item()
                self.aggregateE[layer]=self.updateWelford(self.aggregateE[layer],valueE)

                if self.rule=='z+e':
                    valueZ=torch.mean(mapAbsZ[i].detach().float()).item()
                    self.aggregateZ[layer]=self.updateWelford(self.aggregateZ[layer],valueZ)


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
            
            if self.rule=='z+e':
                mean,std=self.finalizeWelford(self.aggregateZ[layer])
                means[layer].append(mean)
                stds[layer].append(std)
                c0=np.maximum(mean/5,mean-3*std)
                c1=np.minimum(c0*25,mean+3*std)
                cut0[layer].append(c0)
                cut1[layer].append(c1)
                
            
            mean,std=self.finalizeWelford(self.aggregateE[layer])
            means[layer].append(mean)
            stds[layer].append(std)
            c0=np.maximum(mean/5,mean-3*std)
            c1=np.minimum(c0*25,mean+3*std)
            cut0[layer].append(c0)
            cut1[layer].append(c1)
            
        return cut0,cut1,means,stds