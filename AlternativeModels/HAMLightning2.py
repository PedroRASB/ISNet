#based on https://github.com/oyxhust/HAM

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from HAM.models import AttentionNet
from HAM.utils.tools import load_checkpoint
import yaml


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class HAMLightning(pl.LightningModule):
    def __init__(self,
                 labelOrder,
                 LR=1e-3,
                 optim='SGD',
                 amse=0.5,pn=0.01,bound=0.001,union=0.001):
        #Creates ISNet based on ResNet50 HAM, non instantiated
        #labelOrder:ordered list with label names
        
        #training parameters:
        #LR:learning rate
        #optim: SGD or Adam
        #amse,pn,bound,union:loss weights
        
        super (HAMLightning,self).__init__()
        self.save_hyperparameters()
        self.numClasses=len(labelOrder)-1
        
        config = yaml.load(open('fold1.yaml'), Loader=yaml.FullLoader)
        
        self.HAM = AttentionNet.build_model(config['arch'],
                                          config['Downsampling'], 
                                          config['Using_pooling'], config['Using_dilation'],
                                          self.numClasses)
        self.lr=LR
        self.optim=optim
        self.amse=amse
        self.bound=bound
        self.union=union
        self.pn=pn
        
        for i,y in enumerate(labelOrder,0):
            if y=='No Finding':
                self.NormalIdx=i
            
        
    def forward(self,x):
        #x:input
        #output:out_anomaly, out_classes, cam_anomaly_refined, cam_classes_refined, cam_anomaly, cam_classes
        return  self.HAM(x) 

    def configure_optimizers(self):
        if (self.optim=='Adam'):
            optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        else:
            optimizer=torch.optim.SGD(self.parameters(),
                                    lr=self.lr,
                                    momentum=0.99,
                                    weight_decay=0)
        return optimizer
    
    def compound_loss(self,outputs,label,mask):    
        outs_anomaly,outs_classes,cam_anomaly_refined,cam_classes_refined,\
        cams_anomaly,cams_classes=outputs
        
        #labels for classes (remove normal)
        cls_labels=torch.cat([label[:,:self.NormalIdx],label[:,self.NormalIdx+1:]],dim=1)
        ano_labels,_=torch.max(cls_labels,dim=1,keepdim=True)
        #remove normal class to get abnormality labels/masks
        cls_mask=torch.cat([mask[:,:self.NormalIdx],mask[:,self.NormalIdx+1:]],dim=1)
        #remove channel dimension
        cls_mask=cls_mask[:,:,0,:,:]
        
        clsLoss,anoLoss,boundLoss,unionLoss,exClsLoss=\
        OriginalLoss(outs_anomaly, outs_classes, cam_anomaly_refined,
                     cam_classes_refined, cams_anomaly, cams_classes,
                     cls_mask, cls_labels, ano_labels,
                     num_classes=self.numClasses)
        
        #cLoss=F.binary_cross_entropy_with_logits(outs_classes, cls_label)
        #pnLoss=F.binary_cross_entropy_with_logits(outs_anomaly, ano_label)
        #boundLoss=AttentionBoundLoss(cams_anomaly,cams_classes,cls_label,cLoss)
        #unionLoss=AttentionUnionLoss(cams_anomaly,cams_classes,cls_label)
        #amseLoss=AMSELoss(cam_classes_refined, cls_mask)
        
        return clsLoss,anoLoss,boundLoss,unionLoss,exClsLoss
    #[cLoss,pnLoss,boundLoss,unionLoss,amseLoss]
    
    def training_step(self,train_batch,batch_idx):
        #data format: channel first
        inputs,masks,labels=train_batch
        out=self.forward(inputs)
        cLoss,pnLoss,boundLoss,unionLoss,amseLoss=self.compound_loss(out,labels,masks)
        #print(self.pn,self.bound,self.union,self.amse)
        loss=cLoss+self.pn*pnLoss+self.bound*boundLoss+self.union*unionLoss+self.amse*amseLoss
        
        losses=[cLoss,pnLoss,boundLoss,unionLoss,amseLoss]
        for i,name in enumerate(['cLoss','pnLoss','boundLoss','unionLoss','amseLoss'],0):
            if (torch.isnan(losses[i]).any()):
                print('NaN loss in '+name)
            if (torch.isinf(losses[i]).any()):
                print('inf loss in '+name)
        outs_anomaly,outs_classes,cam_anomaly_refined,cam_classes_refined,\
        cams_anomaly,cams_classes=out
        out=[outs_anomaly,outs_classes,cam_anomaly_refined,cam_classes_refined,\
        cams_anomaly,cams_classes]
        for i,name in enumerate(['outs_anomaly','outs_classes',
                                 'cam_anomaly_refined','cam_classes_refined',
                                 'cams_anomaly','cams_classes'],0):
            if (torch.isnan(out[i]).any()):
                print('NaN loss in '+name)
                
        if (torch.isnan(loss).any()):
                raise ValueError('NaN training loss, aborting')

        self.log('train_loss', {'cLoss':cLoss,
                                'pnLoss':pnLoss,
                                'boundLoss':boundLoss,
                                'unionLoss':unionLoss,
                                'amseLoss':amseLoss},                     
                 on_epoch=True)
        return loss
    
    def validation_step(self,val_batch,batch_idx):
        inputs,masks,labels=val_batch
        out=self.forward(inputs)
        cLoss,pnLoss,boundLoss,unionLoss,amseLoss=self.compound_loss(out,labels,masks)
        loss=cLoss+self.pn*pnLoss+self.bound*boundLoss+self.union*unionLoss+self.amse*amseLoss
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

        inputs,labels=test_batch
        out_anomaly,out_classes,_,_,_,_=self.forward(inputs)
        out_NoFinding=-1*out_anomaly#logit for no findings class (sigmoid(x)+sigmoid(-x)=100%)
        logits=torch.cat([out_classes[:,:self.NormalIdx],out_NoFinding,
                          out_classes[:,self.NormalIdx:]],dim=-1)
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
            
        for i,out in enumerate(test_step_outputs,0):
            if i==0:
                continue
            pred=torch.cat((pred,out['pred']),dim=0)
            labels=torch.cat((labels,out['labels']),dim=0)
            if (self.heat):
                images=torch.cat((images,out['images']),dim=0)
                heatmaps=torch.cat((heatmaps,out['heatmaps']),dim=0)
                
        if (self.heat):
            self.TestResults=pred,labels,images,heatmaps
        else:
            self.TestResults=pred,labels
            
#losses                                                        
def OriginalLoss(outs_anomaly, outs_classes, cam_anomaly_refined,
                 cam_classes_refined, cams_anomaly, cams_classes,
                 masks, cls_labels, ano_labels, num_classes=5,cam_w=100, 
                 cam_loss_sigma=0.4):
    #based on https://github.com/oyxhust/HAM/blob/main/utils/net_utils.py
    #outs_anomaly: positive or negative logit
    #outs_classes: diseases logits
    
    #classification and anomaly (pn) loss
    clsLoss=F.binary_cross_entropy_with_logits(outs_classes, cls_labels)
    anoLoss=F.binary_cross_entropy_with_logits(outs_anomaly, ano_labels)
    
    ex_criterion = nn.MSELoss(reduction='none')
    
    bbox_tags=torch.clamp(masks.sum(dim=(-1,-2)),max=1.0)
    for i,_ in enumerate(bbox_tags,0):
        if bbox_tags[i].sum()==0.0:
            bbox_tags[i]=outs_classes[i]
    flags=torch.clamp(masks.sum(dim=(-1,-2,-3)),max=1.0)
    
    # class-wise cam extra supervision
    exClsLosses = []
    boundLosses = []
    cams_cls_vis = []
    count1 = 0
    count2 = 0
    anomaly_cam_hidden = cams_anomaly.squeeze(1) * ano_labels.unsqueeze(2)
    anomaly_cam_mask = torch.sigmoid(cam_w*(anomaly_cam_hidden - cam_loss_sigma))
    cams_all = []
    for idx in range(num_classes):
        #amse
        class_masks = masks[:, idx, :, :]
        class_cams = cam_classes_refined[:, idx, :, :]
        ex_cls_outs = ex_criterion(class_cams, class_masks)
        norm = ((class_cams.sum((1,2))+class_masks.sum((1,2)))*bbox_tags[:, idx]*flags).sum()
        ex_cls_outs = (ex_cls_outs.sum((1,2))*bbox_tags[:, idx]*flags).sum()/max(norm, 1e-5)
        exClsLosses.append(ex_cls_outs)
        
        #bound
        class_cams_hidden = cams_classes[:, idx, :, :]*cls_labels[:, idx].unsqueeze(1).unsqueeze(2)
        cams_all.append(class_cams_hidden)
        class_cams_mask = torch.sigmoid(cam_w*(class_cams_hidden - cam_loss_sigma))
        bound_outs = ((class_cams_hidden.sum((1,2)) - (torch.min(class_cams_hidden, anomaly_cam_hidden)*class_cams_mask).sum((1,2))) / torch.clamp(class_cams_hidden.sum((1,2)), min=1e-5))*cls_labels[:, idx]
        norm = cls_labels[:, idx].sum()
        bound_outs = bound_outs.sum() / max(norm, 1e-5)
        boundLosses.append(bound_outs)
        
        if (bbox_tags[:, idx]*flags).sum() > 0:#bbox present in batch
            count1 += 1
        if cls_labels[:, idx].sum() > 0:
            count2 += 1
    exClsLoss = sum(exClsLosses) / max(count1, 1)
    boundLoss = sum(boundLosses) / max(count2, 1)
    
    #union
    cams_all = torch.stack(cams_all, dim=1)
    cams_all = torch.max(cams_all, dim=1)[0]
    unionLoss = ((anomaly_cam_hidden.sum((1,2)) - (torch.min(cams_all, anomaly_cam_hidden)*anomaly_cam_mask).sum((1,2))) / torch.clamp((anomaly_cam_hidden).sum((1,2)), min=1e-5))*ano_labels.squeeze(1)
    norm = ano_labels.sum()
    unionLoss = unionLoss.sum() / max(norm, 1e-5)

    return clsLoss,anoLoss,boundLoss,unionLoss,exClsLoss
