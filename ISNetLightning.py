import torch
import torch.nn.functional as F
import torch.nn as nn
import ISNetLayers
import ISNetFunctions
import LRPDenseNet
import globals
import pytorch_lightning as pl

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class IsDense121Lightning(pl.LightningModule):
    def __init__(self,multiLabel,multiMask,
                 classes,
                 e=10e-2,heat=True,
                 Zb=True,pretrained=False,
                 LR=1e-3,P=0.7,E=10,saveMaps=False,
                 mapsLocation='',optim='Adam',
                 Cweight=None):
        #Creates ISNet based on DenseNet121, non instantiated
        
        #model parameters:
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #heat: allows relevance propagation and heatmap creation. If False,
        #no signal is propagated through LRP block.
        #pretrained: if a pretrained DenseNet121 shall be downloaded
        #classes: number of output classes
        
        #training parameters:
        #LR:learning rate
        #P: loss balancing hyperparameter
        #E: heatmap loss hyperparameter
        #multiMask: for a segmentation mask per class
        #multiLabel: for multi-label problems
        #saveMaps: saves test hetamaps
        #optim: SGD or Adam
        #Cweight: BCE loss weights to deal with unbalanced datasets
        
        super (IsDense121Lightning,self).__init__()
        self.save_hyperparameters()
        
        self.DenseNet=LRPDenseNet.densenet121(pretrained=pretrained)
        self.DenseNet.classifier=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                               nn.Linear(in_features=1024, 
                                                         out_features=classes,
                                                         bias=True))
        
        self.LRPBlock=ISNetLayers._LRPDenseNet(self.DenseNet,e=e,Zb=Zb)
        self.heat=heat
        self.lr=LR
        self.P=P
        self.E=E
        self.multiMask=multiMask
        self.multiLabel=multiLabel
        self.saveMaps=saveMaps
        self.mapsLocation=mapsLocation
        self.optim=optim
        self.criterion=nn.CrossEntropyLoss()
        if (Cweight is not None):
            self.Cweight=nn.parameter.Parameter(data=Cweight, requires_grad=False)
        else:
            self.Cweight=None

    def forward(self,x):
        #x:input
        
        y=self.DenseNet(x)
        if(self.heat):#only run the LRP block if self.heat==True
            R=self.LRPBlock(x=x,y=y)
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
                                    momentum=0.99,
                                    weight_decay=0)
        return optimizer
    
    def compound_loss(self,outputs,labels,heatmaps=None,masks=None):
        if(self.multiLabel):
            classifierLoss=F.binary_cross_entropy_with_logits(outputs,labels,
                                                              pos_weight=self.Cweight)
        else:
            #classifierLoss=F.cross_entropy(outputs,labels.squeeze(1))
            classifierLoss=self.criterion(outputs,labels.squeeze(1))
        
        if (self.heat):
            heatmapLoss=ISNetFunctions.LRPLossActivated(heatmaps,masks,
                                                        E=self.E,
                                                        multiMask=self.multiMask)
            return classifierLoss,heatmapLoss
        else:
            return classifierLoss
    
    def training_step(self,train_batch,batch_idx):
        #data format: channel first

        if (self.heat):#ISNet
            inputs,masks,labels=train_batch
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
            
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        #data format: channel first
        
        if (self.heat):#ISNet
            inputs,masks,labels=val_batch
            logits,heatmaps=self.forward(inputs)
            cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                      heatmaps=heatmaps,masks=masks)
            loss=(1-self.P)*cLoss+self.P*hLoss

        else:#Common DenseNet
            inputs,labels=val_batch
            logits=self.forward(inputs)
            loss=self.compound_loss(logits,labels=labels)

        #self.log('val_loss',loss,on_epoch=True,sync_dist=True)
        return {'loss': loss}
            

    def validation_step_end(self, batch_parts):
        if(batch_parts['loss'].dim()>1):
            losses=batch_parts['loss']
            return {'loss': torch.mean(losses,dim=0)}
        else:
            return batch_parts

    def validation_epoch_end(self, validation_step_outputs):
        loss=validation_step_outputs[0]['loss'].unsqueeze(0)
        for out in validation_step_outputs:
            loss=torch.cat((loss,out['loss'].unsqueeze(0)),dim=0)

        self.log('val_loss',torch.mean(loss,dim=0),
                 on_epoch=True,sync_dist=True)
    
    def test_step(self,test_batch,batch_idx):
        #data format: channel first
        
        if (self.heat):#ISNet
            inputs,masks,labels=test_batch
            logits,heatmaps=self.forward(inputs)
            return {'pred': logits, 'labels': labels, 'images': inputs.cpu().float(), 'heatmaps': heatmaps.cpu().float()}

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
            else:
                images=batch_parts['images']
                heatmaps=batch_parts['heatmaps']
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[2]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1]),
                        'images': images.view(images.shape[0]*images.shape[1],images.shape[2],
                                              images.shape[3],images.shape[4]),
                        'heatmaps': heatmaps.view(heatmaps.shape[0]*heatmaps.shape[1],heatmaps.shape[2],
                                                  heatmaps.shape[3],heatmaps.shape[4],heatmaps.shape[5])}
        else:
            return batch_parts

    def test_epoch_end(self, test_step_outputs):
        pred=test_step_outputs[0]['pred']
        labels=test_step_outputs[0]['labels']
        if (self.heat):
            images=test_step_outputs[0]['images']
            heatmaps=test_step_outputs[0]['heatmaps']
            
        for out in test_step_outputs:
            pred=torch.cat((pred,out['pred']),dim=0)
            labels=torch.cat((labels,out['labels']),dim=0)
            if (self.heat):
                images=torch.cat((images,out['images']),dim=0)
                heatmaps=torch.cat((heatmaps,out['heatmaps']),dim=0)
                
        if (self.heat):
            self.TestResults=pred,labels,images,heatmaps
        else:
            self.TestResults=pred,labels