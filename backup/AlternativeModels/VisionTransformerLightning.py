import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

class VisionTransformer(pl.LightningModule):
    def __init__(self,classes,multiLabel=False,LR=1e-3,optim='SGD',momentum=0.99,pretrained=False):

        #LR:learning rate
        #optim: SGD or Adam
        
        super (VisionTransformer,self).__init__()
        self.save_hyperparameters()
        
        if not pretrained:
            self.model=torchvision.models.vit_b_16()
        else:
            #https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
            self.model=torchvision.models.vit_b_16("pretrained")
        self.model.heads.head=nn.Linear(in_features=768, out_features=classes, bias=True)
        
        self.lr=LR
        self.optim=optim
        self.momentum=momentum
        self.multiLabel=multiLabel

    def forward(self,x):
        #x:input
        x=self.model(x)
        return x

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
                                    weight_decay=0)
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        inputs,labels=train_batch
        logits=self.forward(inputs)
        if not self.multiLabel:
            loss=F.cross_entropy(logits,labels.squeeze(-1))
        else:
            loss=F.binary_cross_entropy_with_logits(logits,labels)
        self.log('train_loss',loss,                     
                 on_epoch=True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        inputs,labels=val_batch
        logits=self.forward(inputs)
        if not self.multiLabel:
            loss=F.cross_entropy(logits,labels.squeeze(-1))
        else:
            loss=F.binary_cross_entropy_with_logits(logits.float(),labels.float())
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
        inputs,labels=test_batch
        logits=self.forward(inputs)
        return {'pred': logits.detach().cpu(),
                'labels': labels.detach().cpu()}
            

    def test_step_end(self, batch_parts):
        if(batch_parts['pred'].dim()>3):
            logits=batch_parts['pred']
            labels=batch_parts['labels']
            return {'pred': logits.view(logits.shape[0]*logits.shape[1],
                                        logits.shape[2],logits.shape[3]),
                    'labels': labels.view(labels.shape[0]*labels.shape[1],
                                          labels.shape[2],labels.shape[3])}
        else:
            return batch_parts

    def test_epoch_end(self, test_step_outputs):
        pred=test_step_outputs[0]['pred']
        labels=test_step_outputs[0]['labels']
            
        for i,out in enumerate(test_step_outputs,0):
            if (i!=0):
                pred=torch.cat((pred,out['pred']),dim=0)
                labels=torch.cat((labels,out['labels']),dim=0)
                
        self.TestResults=pred,labels