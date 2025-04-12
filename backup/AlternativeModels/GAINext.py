#based on https://github.com/AustinDoolittle/Pytorch-Gain

import torch
import torch.nn.functional as F
import numpy as np
import os
import re
import io
import json
import math
import cv2
from collections import OrderedDict
from typing import Dict, Callable
import copy

def scalar(tensor):
    return tensor.data.cpu().item()

def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)
            
def remove_all_backward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_backward_hooks(child)

class AttentionGAIN:
    def __init__(self, model, device, multiLabel, gradient_layer_name='features.fReLU',
                 heatmap_dir=None, alpha=1, omega=10, sigma=0.5, labels=None,
                 batch_norm=True, w=10, modified=False, clamp=None, abs=False,modReLU=False,
                 heat=False, original=True, sigmoid_am=False):
        # define model
        self.model=model
        if device is not None:
            self.model=self.model.to(device)
        self.num_classes=model.classifier[1].out_features

        # wire up our hooks for heatmap creation
        self._register_hooks(gradient_layer_name)

        # create loss function
        if not multiLabel:
            self.loss_cl = torch.nn.CrossEntropyLoss()
        else:
            self.loss_cl = torch.nn.BCEWithLogitsLoss()
        

        # output directory setup
        self.heatmap_dir = heatmap_dir
        if self.heatmap_dir:
            self.heatmap_dir = os.path.abspath(self.heatmap_dir)

        # misc. parameters
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        self.labels = labels
        self.multiLabel=multiLabel
        self.gradient_layer_name=gradient_layer_name
        self.w=w
        self.modified=modified
        self.clamp=clamp
        self.abs=abs
        self.modReLU=modReLU
        self.heat=heat
        self.original=original
        self.sigmoid_am=sigmoid_am

    def _register_hooks(self, layer_name):
        # this wires up a hook that stores both the activation and gradient of the conv layer we are interested in
        def forward_hook(module, input_, output_):
            self._last_activation = output_

        def backward_hook(module, grad_in, grad_out):
            self._last_grad = grad_out[0]

        # locate the layer that we are concerned about
        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == layer_name:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model'%layer_name)


    def _convert_data_and_label(self, data, label):
        # converts our data and label over to variables, gpu optional
        data = torch.autograd.Variable(data)
        label = torch.autograd.Variable(label)

        return data, label

    
    def _maybe_save_heatmap(self, image, label, heatmap, I_star, epoch, heatmap_nbr):
        if self.heatmap_dir is None:
            return

        heatmap_image = self._combine_heatmap_with_image(image, heatmap, self.labels[label])

        I_star = I_star.data.cpu().numpy().transpose((1,2,0)) * 255.0
        out_image = tile_images([heatmap_image, I_star])

        # write it to a file
        if not os.path.exists(self.heatmap_dir):
            os.makedirs(self.heatmap_dir)

        out_file = os.path.join(self.heatmap_dir, 'epoch_%i_%i.png'%(epoch, heatmap_nbr))
        cv2.imwrite(out_file, out_image)

        print('HEATMAP saved to %s'%out_file)

    @staticmethod
    def _combine_heatmap_with_image(image, heatmap, label_name, font_scale=0.75, 
                                    font_name=cv2.FONT_HERSHEY_SIMPLEX, font_color=(255,255,255), 
                                    font_pixel_width=1):
        # get the min and max values once to be used with scaling
        min_val = heatmap.min()
        max_val = heatmap.max()

        # Scale the heatmap in range 0-255
        heatmap = (255 * (heatmap - min_val)) / (max_val - min_val + 1e-5)
        heatmap = heatmap.data.cpu().numpy().astype(np.uint8).transpose((1,2,0))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Scale the image as well
        scaled_image = image * 255.0
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1,2,0))

        if scaled_image.shape[2] == 1:
            scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2RGB)

        # generate the heatmap
        heatmap_image = cv2.addWeighted(scaled_image, 0.7, heatmap, 0.3, 0)

        # superimpose label_name
        if label_name:
            (_, text_size_h), baseline = cv2.getTextSize(label_name, font_name, font_scale, font_pixel_width)
            heatmap_image = cv2.putText(heatmap_image, label_name,
                                        (10, text_size_h + baseline + 10),
                                        font_name,
                                        font_scale,
                                        font_color,
                                        thickness=font_pixel_width)
        return heatmap_image

    def generate_heatmap(self, data, label, width=3):
        data_var, label_var = self._convert_data_and_label(data, label)
        if (self.labels):
            label_name = self.labels[int(label.max())]
        else:
            label_name = None
        output_cl, loss_cl, A_c = self._attention_map_forward(data_var, label_var)
        heatmap = self._combine_heatmap_with_image(data[0], A_c[0], label_name)
        return output_cl, loss_cl, A_c, heatmap

    def forward(self, data, label, mask=None):
        data, label = self._convert_data_and_label(data, label)     
        if not self.multiLabel:
            label=torch.nn.functional.one_hot(label.long(), num_classes=self.num_classes).float()
        return self._forward(data, label, mask)

    def _attention_map_forward(self, data, label):
        output_cl = self.model(data)
        grad_target = (output_cl * label).sum()
        grad_target.backward(retain_graph=True)
        

        self.model.zero_grad()

        # Eq 1
        grad = self._last_grad
        w_c = F.avg_pool2d(self._last_grad, (self._last_grad.shape[-2], 
                                             self._last_grad.shape[-1]), 1)
        #takes space average of gradients to get channel weights for last feature map
        #print (w_c.shape)
        #print(self._last_activation.shape)
        # Eq 2
        #updated by Bassi to use group convolution and batch. 
        batch_size=self._last_activation.shape[0]
        feature_map = self._last_activation
        feature_map=feature_map.reshape(feature_map.shape[0] * feature_map.shape[1],
                                feature_map.shape[2], feature_map.shape[3]).unsqueeze(0)
        
        if self.modified:
            gcam = F.conv2d(feature_map, w_c, groups=batch_size).squeeze(0).unsqueeze(1)
            #print(gcam.shape)
            if self.modReLU:
                gcam= F.relu(gcam)
            if self.abs:
                gcam=torch.abs(gcam)
            if self.clamp is not None:
                gcam = torch.clamp(gcam,max=self.clamp)
        else:
            gcam = F.relu(F.conv2d(feature_map, w_c, groups=batch_size)).squeeze(0).unsqueeze(1)
            
        gcam=gcam.repeat(1,3,1,1)#added 3 channels
        A_c = F.interpolate(gcam, size=data.size()[2:], mode='bilinear')
        if not self.multiLabel:
            label=torch.argmax(label, dim=1)
            
        loss_cl = self.loss_cl(output_cl, label)

        return output_cl, loss_cl, A_c

    def _mask_image(self, gcam, image):
        #gcam_min,_ = torch.min(gcam,dim=-1,keepdim=True)
        #gcam_max,_ = torch.max(gcam,dim=-1,keepdim=True)
        #scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min)
        mask = torch.sigmoid(self.omega * (gcam - self.sigma))
        #print(torch.min(image),torch.max(image))
        if self.multiLabel:
            masked_image = image.unsqueeze(1).repeat(1,mask.shape[1],1,1,1) - \
            torch.mul(image.unsqueeze(1).repeat(1,mask.shape[1],1,1,1),mask)
        else:
            masked_image = image - torch.mul(image,mask)
        return masked_image

    def _forward(self, data, label, mask):
        # TODO normalize elsewhere, this feels wrong
        output_cl, loss_cl, gcam = self._attention_map_forward(data, label)
        
        if ((self.modified and mask is not None) or self.multiLabel):
            classes=list(range(label.shape[-1]))
            heatmaps=[]
            for i in classes:
                #create one heatmap per class,
                target=torch.zeros(label.shape).type_as(label)
                target[:,i]=1.0
                _, _, heatmap = self._attention_map_forward(data, target)   
                heatmaps.append(heatmap)
            heatmaps=torch.stack(heatmaps,1)
            

        # Eq 4
        if self.multiLabel:
            #consider all present classes
            I_star = self._mask_image(heatmaps, data)
            output_am=[]
            for i in classes:
                #all maps for class i:
                x=I_star[:,i]
                #remove maps when class i is not in label:
                tmp=[]
                for j,item in enumerate(x,0):
                    if (label[j,i].float().item()==1.0):
                        tmp.append(item)
                if (len(tmp)==0):
                    #skip if all labels are 0 for class i
                    continue
                x=torch.stack(tmp,dim=0)
                x=self.model(x)
                x=x[:,i]#get logit for class i
                output_am.append(x)
            if (len(output_am)!=0):
                output_am=torch.cat(output_am,dim=0)
            else:
                #all labels in the batch were 0
                output_am=None
        else:
            I_star = self._mask_image(gcam, data)
            output_am = self.model(I_star)

        # Eq 5
        if self.multiLabel:
            if output_am is not None:
                loss_am = torch.sigmoid(output_am)
                loss_am = torch.mean(loss_am)#batch
            else:
                loss_am = torch.tensor(0).type_as(label)
        else:
            if not self.sigmoid_am:
                loss_am = torch.softmax(output_am,dim=-1) * label
            else:
                loss_am = torch.sigmoid(output_am) * label
            loss_am = torch.sum(loss_am,dim=-1)#classes
            loss_am = torch.mean(loss_am)#batch
            
        #extra supervision:
        loss_e=None
        if mask is not None:
            if (self.multiLabel and self.modified):
                raise ValueError('Multi label not implemented for extra supervision')
                
            if not self.modified and not self.multiLabel:
                loss_e=F.mse_loss(gcam,mask)
            elif self.multiLabel:
                #calculate for all classes and samples, calculate mean over only the classes present in labels
                if torch.isnan(heatmaps).any():
                    print('nan 0')
                loss_e=F.mse_loss(heatmaps,mask.unsqueeze(1).repeat(1,label.shape[-1],1,1,1),
                                  reduction='none')
                loss_e=torch.mul(loss_e, label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                loss_e=loss_e.mean(dim=[-1,-2,-3])
                loss_e=loss_e.sum()/(label.sum()+1e-5)
            else:
                loss_e=F.mse_loss(heatmaps,mask.unsqueeze(1).repeat(1,label.shape[-1],1,1,1))

        # Eq 6
        total_loss = loss_cl + self.alpha*loss_am
        
        if mask is not None:
            total_loss=total_loss+self.w*loss_e
            
        if self.original:
            if torch.isnan(loss_cl).any():
                    print('loss_cl nan')
            if torch.isnan(loss_am).any():
                    print('loss_am nan')
            if loss_e is not None:
                if torch.isnan(loss_e).any():
                        print('loss_e nan')
            return total_loss, loss_cl, loss_am, gcam, output_cl,loss_e
        elif self.heat:
            return output_cl,heatmaps
        else:
            return output_cl
    
    def return_model(self):
        #remove hooks
        remove_all_forward_hooks(self.model)
        remove_all_backward_hooks(self.model)
        #get clean model
        FreeModel=copy.deepcopy(self.model)
        #restore hooks
        self._register_hooks(self.gradient_layer_name)
        return FreeModel
        