import torch
import torch.nn as nn
import globals
from collections import OrderedDict
from typing import Dict, Callable

Detached=True
    
def GlobalWeightedRankPooling(x,d=0.9):
    x,_=torch.sort(x.view(x.shape[0],x.shape[1],x.shape[2],
                                       x.shape[3]*x.shape[4]),
                   dim=-1,descending=True)
    weights=torch.tensor([d ** i for i in range(x.shape[-1])]).type_as(x)
    weights=weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x=torch.mul(x,weights)
    x=x.sum(-1)/weights.sum()
    return x

def LRPLossCEValleysGWRP (heatmap, mask, cut=1, cut2=25, reduction='mean', 
                                     norm=True, A=1, B=3, E=1,d=0.9,normRoI=True,var=False,
                                     alternativeCut=False,detachNorm=False,multiMask=False,
                                     eps=1e-10):
    #ISNet heatmap loss.
    #A: W1 in paper
    #B: W2 in paper
    #cut1 and cut2: C1 and C2 in paper
    
    batchSize=heatmap.shape[0]
    classesSize=heatmap.shape[1]
    channels=heatmap.shape[2]
    length=heatmap.shape[-1]
    width=heatmap.shape[-2]
    if (not multiMask):
        mask=mask.unsqueeze(1).repeat(1,classesSize,1,1,1)
    Imask=torch.ones(mask.shape).type_as(mask)-mask

    with torch.cuda.amp.autocast(enabled=False):
        heatmap=heatmap.float()
        mask=mask.float()
        Imask=Imask.float()
        #abs:
        heatmap=torch.abs(heatmap)
        
        #substitute nans if necessary:
        if torch.isnan(heatmap).any():
            print('nan 0')
        if torch.isinf(heatmap).any():
            print('inf 0')
        RoIMean=torch.sum(torch.nan_to_num(heatmap,posinf=0.0,neginf=0.0)*mask)/(torch.sum(mask)+eps)
        #print(RoIMean)
        heatmap=torch.nan_to_num(heatmap,nan=RoIMean.item(), 
                                 posinf=torch.max(torch.nan_to_num(heatmap,posinf=0)).item(),
                                 neginf=torch.max(torch.nan_to_num(heatmap,posinf=0)).item())
        
        #save non-normalized heatmap:
        heatmapRaw=heatmap[:]
        
        if torch.isnan(heatmap).any():
            print('nan 1')
        if torch.isinf(heatmap).any():
            print('inf 1')
        #normalize heatmap:
        if norm:
            if var:
                denom=torch.var(heatmap, dim=(-1,-2,-3), keepdim=True)
                denom=torch.sqrt(denom)
            elif normRoI:
                #roi mean value:
                denom=torch.sum(heatmap*mask, dim=(-1,-2,-3),
                                keepdim=True)/(torch.sum(mask,dim=(-1,-2,-3),keepdim=True)+eps)
                if torch.isnan(denom).any():#nan in denom
                    print('nan 0.5')
                if torch.isinf(denom).any():
                    print('inf 0.5')
            else:
                denom=torch.mean(heatmap, dim=(-1,-2,-3), keepdim=True)
                
            if detachNorm:
                heatmap=heatmap/(denom.detach()+eps)
            else:
                heatmap=heatmap/(denom+eps)
        else:
            heatmap=heatmap*channels*length*width
            
        if torch.isnan(heatmap).any():
            print('nan 2')
        if torch.isinf(heatmap).any():
            print('inf 2')
            
        #Background:
        heatmapBKG=torch.mul(heatmap,Imask)
        #global maxpool on spatial dimensions:
        heatmapBKG=GlobalWeightedRankPooling(heatmapBKG,d=d)
        #activation:
        heatmapBKG=heatmapBKG/(heatmapBKG+E)
        #cross entropy (pixel-wise):
        heatmapBKG=torch.clamp(heatmapBKG,max=1-1e-7)
        loss=-torch.log(torch.ones(heatmapBKG.shape).type_as(heatmapBKG)-heatmapBKG)
        loss=torch.mean(loss,dim=(-1,-2))#channels and classes mean
        
        if torch.isnan(loss).any():
            print('nan 3')
        if torch.isinf(loss).any():#here
            print('inf 3')
            if (not torch.isinf(heatmapBKG).any()):
                print('inf by log')
            
            
        #avoid foreground values too low or too high:
        heatmapF=torch.mul(heatmapRaw,mask).sum(dim=(-1,-2,-3))
        if torch.is_tensor(cut):
            if alternativeCut:
                raise ValueError('Alternative cut should not be used with per class cut values')
            target=cut.unsqueeze(0).repeat(heatmapF.shape[0],1).type_as(heatmapF)
            target2=cut2.unsqueeze(0).repeat(heatmapF.shape[0],1).type_as(heatmapF)
        else:
            target=cut*torch.ones(heatmapF.shape).type_as(heatmapF)
            target2=cut2*torch.ones(heatmapF.shape).type_as(heatmapF)
        #print(heatmapF)
        
        if alternativeCut:
            #apply minimum over sum of heatmaps for all classes
            target=cut*torch.ones(heatmapF.sum(dim=-1).shape).type_as(heatmapF)
            lossF=nn.functional.mse_loss(torch.clamp(heatmapF.sum(dim=-1),min=None,max=target),
                                         target, 
                                         reduction='none')
        else:
            lossF=nn.functional.mse_loss(torch.clamp(heatmapF,min=None,max=target),
                                         target, 
                                         reduction='none')
            lossF=torch.mean(lossF,dim=-1)
        lossF=lossF/(cut**2)
        
        
        lossF=lossF+torch.mean(nn.functional.mse_loss(torch.clamp(heatmapF,min=target2,max=None),
                                                           target2, 
                                                           reduction='none'),dim=-1)#classes mean
        
        loss=A*loss+B*lossF
        if torch.isnan(loss).any():
            print('nan 4')
        if torch.isinf(loss).any():
            print('inf 4')
        
        if (reduction=='sum'):
            loss=torch.sum(loss)
        elif (reduction=='mean'):
            loss=torch.mean(loss)
        elif (reduction=='none'):
            pass
        else:
            raise ValueError('reduction should be none, mean or sum')
        
        return loss
    

        

def stabilizer(aK,e):
    #Used for LRP-e, returns terms sign(ak)*e
    #ak: values to stabilize
    #e: LRP-e term 

    signs=torch.sign(aK)
    #zeros as positives
    signs[signs==0]=1
    signs=signs*e
    return signs


    
def LRPDenseReLU(layer,rK,aJ,aK,e):
    #Propagates relevance through fully-connected layer.
    #layer: layer L throgh which we propagate relevance, fully-connected
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
    #aK: activations before ReLU
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]

    weights=layer.weight
        
    if (Detached and layer.bias is not None):
        aK=nn.functional.linear(aJ,weights,layer.bias.detach())
    
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1)

        
    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k

    W=torch.transpose(weights,0,1)
    #mix batch and class dimensions
    s=s.view(batchSize*numOutputs,s.shape[-1])
    #back relevance with transposed weigths
    c=nn.functional.linear(s,W)
    #unmix:
    c=c.view(batchSize,numOutputs,c.shape[-1])
    #add classes dimension:
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1)   
    RJ=torch.mul(AJ,c)

    return RJ

def LRPConvReLU(layer,rK,aJ,aK,e):
    #returns: RJ, relevance at input of layer L
    #layer: layer L throgh which we propagate relevance, convolutional
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
    #aK: activations before ReLU
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
        
    weights=layer.weight
        
    if (Detached and layer.bias is not None):
        aK=nn.functional.conv2d(aJ,weight=layer.weight,bias=layer.bias.detach(),
                                stride=layer.stride,padding=layer.padding)
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
    c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                     stride=layer.stride,padding=layer.padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)

    return RJ

def LRPOutput(y):
    #returns output relevance, create a new batch dimenison for classes,
    #of size C, the number of DNN outputs/classes. Returns diagonal matrix.
    #y:classifier output
    
    numOutputs=y.shape[-1]
    #define identity matrix:
    I=torch.diag(torch.ones((numOutputs)).type_as(y))
    #repeat the outputs and element-wise multiply with identity
    RO=torch.mul(y.unsqueeze(-1).repeat(1,1,numOutputs),I)
    return RO

def AvgPoolWeights(kernel_size,stride,channels):
    #create convolution parameters equivalent to average pooling
    #kernel_size: pooling kernel size
    #stride: pooling stride
    #channels: pooling number of channels
    
    if (isinstance(kernel_size, int)):
        k0=kernel_size
        k1=kernel_size
    else:
        k0=kernel_size[0]
        k1=kernel_size[1]
        
    weights=torch.zeros((channels,channels,k0,k1))
    for i in list(range(channels)):
        weights[i,i,:,:]=torch.ones((k0,k1))
    weights=weights
    #define stride:
    if(stride is None):
        stride=(k0,k1)
    else:
        stride=stride
        
    biases=torch.zeros((channels))
    
    #average:
    weights=weights/(k0*k1)
    
    return (weights,stride,biases)

def SumPoolWeights(kernel_size,stride,channels):
    #create convolution parameters equivalent to sum pooling
    #kernel_size: pooling kernel size
    #stride: pooling stride
    #channels: pooling channels
    
    if (isinstance(kernel_size, int)):
        k0=kernel_size
        k1=kernel_size
    else:
        k0=kernel_size[0]
        k1=kernel_size[1]
        
    weights=torch.zeros((channels,channels,k0,k1))
    for i in list(range(channels)):
        weights[i,i,:,:]=torch.ones((k0,k1))
    weights=weights
    #define stride:
    if(stride is None):
        stride=(k0,k1)
    else:
        stride=stride
        
    biases=torch.zeros((channels))
    
    return (weights,stride,biases)

def LRPPool2d(layer,rK,aJ,aK,e,adaptative=False):
    #relevance propagation through average pooling
    #adaptative: For adaptative average pooling
    #layer: pooling layer
    #e:e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer output
    #aJ: pooling layer inputs
    #aK: activations after pooling
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=rK.shape[2]
    
    if(adaptative):
        stride=(int(aJ.shape[-2]/layer.output_size[0]),int(aJ.shape[-1]/layer.output_size[1]))
        kernel_size=(aJ.shape[-2]-(layer.output_size[0]-1)*stride[0],
                     aJ.shape[-1]-(layer.output_size[1]-1)*stride[1])
        padding=0
    else:
        kernel_size=layer.kernel_size
        stride=layer.stride
        padding=layer.padding
        
    #avgpooling weights
    weights,stride,_=AvgPoolWeights(kernel_size,stride,channels=channels)
    weights=weights.type_as(rK)
    #create aK from avg pooling:
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

    #transpose conv:
    c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                   stride=stride,padding=padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)


    return RJ

def LRPSum(rK,aJ,aK,e):
    #relevance propagation through torch.sum in the spacial dimensions
    #e:e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer output
    #aJ: pooling layer inputs
    #aK: activations after pooling
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=rK.shape[2]
    
    #add dimensions removed by sum
    aK=aK.unsqueeze(-1).unsqueeze(-1)
    rK=rK.unsqueeze(-1).unsqueeze(-1)
    
    kernel_size=(aJ.shape[-2],aJ.shape[-1])
    stride=kernel_size
    padding=0
        
    #sumpooling weights
    weights,stride,_=SumPoolWeights(kernel_size,stride,channels=channels)
    weights=weights.type_as(rK)
    
    #create aK:
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

    #transpose conv:
    c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                   stride=stride,padding=padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)

    return RJ

def FuseBN(layerWeights, BN, aKConv,Ch0=0,Ch1=None,layerBias=None,
           bias=True,BNbeforeReLU=True):
    #returns parameters of convolution fused with batch normalization
    #BN:batch normalization layer
    #layerWeights: convolutional layer wieghts
    #layerBias: convolutional layer bias
    #aKConv: BN input
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the convolutional layer
    #bias: allows returning of bias of equivalent convolution
    #BNbeforeReLU: true if BN is placed before the layer activation
    
    if(layerBias is not None):
        layerBias=layerBias.detach()
    
    if(Ch1==BN.running_var.shape[0]):
        Ch1=None
        
    if (BN.training):
        mean=torch.mean(aKConv,dim=(0,-2,-1)).detach()
        var=torch.var(aKConv,dim=(0,-2,-1)).detach()
        std=torch.sqrt(var+BN.eps)
    else:
        mean=BN.running_mean[Ch0:Ch1].detach()
        var=BN.running_var[Ch0:Ch1].detach()
        std=torch.sqrt(var+BN.eps)    
        
    #multiplicative factor for each channel, caused by BN:
    if(BN.weight is None):
        std=std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights=layerWeights/std
    else:
        #y/rho:
        BNweights=torch.div(BN.weight[Ch0:Ch1],std)
        #add 3 dimensions (in channels, width and height) in BN weights 
        #to match convolutional weights shape:
        if(BNbeforeReLU):
            BNweights=BNweights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shp=layerWeights.shape
            BNweights=BNweights.repeat(1,shp[-3],shp[-2],shp[-1])
        else:#align with output channels dimension
            BNweights=BNweights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            shp=layerWeights.shape
            BNweights=BNweights.repeat(shp[0],1,shp[2],shp[3])
        #w.y/rho:
        weights=torch.mul(BNweights,layerWeights)
    
    if (not bias):
        return weights
    
    if(bias):
        if(layerBias is None):
            layerBias=torch.zeros((weights.shape[0])).type_as(layerWeights)
        if(BN.weight is None and BN.bias is None):
            biases=layerBias*0
        else:
            if(BNbeforeReLU):
                biases=layerBias-mean
                biases=torch.div(biases,std)
                biases=torch.mul(BN.weight[Ch0:Ch1],biases)
                biases=biases+BN.bias[Ch0:Ch1].detach()
            else:
                biases=torch.mul(BN.weight[Ch0:Ch1],mean)
                biases=torch.div(biases,std)
                biases=BN.bias[Ch0:Ch1].detach()-biases
                biases=biases.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                shp=layerWeights.shape
                biases=biases.repeat(shp[0],1,shp[2],shp[3])
                biases=torch.mul(biases,layerWeights)
                biases=biases.sum([1,2,3])
                biases=biases+layerBias
                

        return (weights, biases)

def LRPPool2dBNReLU(layer,BN,rK,aJ,aK,e,aKPool,Ch0=0,Ch1=None):
    #propagates relevance through average pooling followed by batch normalization and ReLU
    #layer: pooling layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L pooling input
    #aK: activations before ReLU
    #aKPool: pooling output
    #BN: batch normalization layer
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the pooling layer
        
    rK=rK[:,:,Ch0:Ch1,:,:]
    
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=rK.shape[2]

    #create weight that represents sum pooling:
    weights,stride,biases=AvgPoolWeights(layer.kernel_size,layer.stride,channels=channels)
    weights,biases=weights.type_as(rK),biases.type_as(rK)
    
    #consider BN:
    weights,biases=FuseBN(layerWeights=weights, BN=BN, aKConv=aKPool,bias=True,
                          Ch0=Ch0,Ch1=Ch1,layerBias=biases)
    
    #create aK from sum pooling:
    if (Detached and BN.bias is not None):
        aK=nn.functional.conv2d(aJ,weight=weights,bias=biases,stride=stride,padding=layer.padding)
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

    #transpose conv:
    c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                   stride=stride,padding=layer.padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)

    return RJ


def MultiBlockPoolBNReLU(layer,BN,rK,aJ,e,aKPool,aK,Ch0=0,Ch1=None):
    #propagates relevance through average pooling followed by batch normalization and ReLU 
    #in transition layers
    #layer: pooling layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L pooling input
    #aK: activations after each BN, list
    #aKPool: pooling output
    #BN: batch normalization layers, list of first BN layers for each dense layer in the next dense block
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the pooling layer
        
    for i,r in enumerate(rK,0):
        if(Ch1==rK[i].shape[2]):
            TCh1=None
        else: 
            TCh1=Ch1
        rK[i]=rK[i][:,:,Ch0:TCh1,:,:]
    rK=torch.cat(rK,dim=2)
    
    #size of batch dimension (0):
    batchSize=aKPool.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=aKPool.shape[1]
    
    #create weight that represents avg pooling:
    weights,stride,biases=AvgPoolWeights(layer.kernel_size,
                                         layer.stride,channels=channels)
    weights,biases=weights.type_as(rK),biases.type_as(rK)
    
    #consider BN: Fuse with each BN layer in list:
    W=[]
    B=[]
    for i,norm in enumerate(BN,0):
        w,b=FuseBN(layerWeights=weights,layerBias=biases, BN=BN[i], aKConv=aKPool,
                   Ch0=Ch0,Ch1=Ch1,bias=True)
        W.append(w)
        B.append(b)
        
    weights=torch.cat(W,dim=0)
    biases=torch.cat(B,dim=0)
    
    if (Detached and BN[0].bias is not None):
        aK=nn.functional.conv2d(aJ,weight=weights,bias=biases,stride=stride,padding=layer.padding) 
    else:
        for i,a in enumerate(aK,0):
            if(Ch1==aK[i].shape[1]):
                TCh1=None
            else: 
                TCh1=Ch1
            aK[i]=aK[i][:,Ch0:TCh1,:,:]
        aK=torch.cat(aK,dim=1)
        
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

    #transpose conv:
    if(isinstance(stride, int)):
        st=stride
    else:
        st=max(stride[0],stride[1])
    if(isinstance(layer.kernel_size, int)):
        ks=layer.kernel_size
    else:
        ks=max(layer.kernel_size[0],layer.kernel_size[1])
        
    if(ks>st):
        c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=stride,padding=layer.padding,
                                        output_padding=(1,1))
    else:
        c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=stride,padding=layer.padding)
   
            
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)  
    
    RJ=torch.mul(AJ,c)

    return RJ


def MultiLayerConvBNReLU(layer,BN,rK,aK,aJ,e,aKConv,Ch0=0,Ch1=None):
    #propagates relevance through last convolution in the dense layers in dense blocks
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN, list (one for each BN layer)
    #aKConv: convolution output
    #BN: batch normalization layer, list of first BN operations in layers connecting to the convolution in "layer"
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the concolutional layer

    for i,r in enumerate(rK,0):
        if(Ch1==rK[i].shape[2]):
            TCh1=None
        else: 
            TCh1=Ch1
        rK[i]=rK[i][:,:,Ch0:TCh1,:,:]
    rK=torch.cat(rK,dim=2)
    #size of batch dimension (0):
    batchSize=aKConv.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=aKConv.shape[1]
    
    #consider BN: Fuse with each BN layer in list:
    W=[]
    B=[]
    for i,norm in enumerate(BN,0):
        w,b=FuseBN(layerWeights=layer.weight,layerBias=layer.bias, BN=BN[i], aKConv=aKConv,
                   Ch0=Ch0,Ch1=Ch1,bias=True)
        W.append(w)
        B.append(b)
        
    weights=torch.cat(W,dim=0)
    biases=torch.cat(B,dim=0)
    

    if (Detached and BN[0].bias is not None):
        aK=nn.functional.conv2d(aJ,weights,biases,stride=layer.stride,padding=layer.padding)
    else:
        #concatenate ak in channels dimension:
        for i,a in enumerate(aK,0):
            if(Ch1==aK[i].shape[1]):
                TCh1=None
            else: 
                TCh1=Ch1
            aK[i]=aK[i][:,Ch0:TCh1,:,:]
        aK=torch.cat(aK,dim=1)
        
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k

        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

    #transpose conv:
    c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                   stride=layer.stride,padding=layer.padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)

    return RJ

def LRPConvBNReLU(layer,BN,rK,aJ,aK,e,aKConv,Ch0=0,Ch1=None):
    #used to propagate relevance through the sequence: Convolution, Batchnorm, ReLU
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN
    #aKConv: convolution output
    #BN: batch normalization layer
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the concolutional layer
    
    aK=aK[:,Ch0:Ch1,:,:]
    rK=rK[:,:,Ch0:Ch1,:,:]
    
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    weights,biases=FuseBN(layerWeights=layer.weight, BN=BN, aKConv=aKConv,
                   Ch0=Ch0,Ch1=Ch1,layerBias=layer.bias,
                   bias=True)
    
    if (Detached and BN.bias is not None):
        aK=nn.functional.conv2d(aJ,weights,biases,stride=layer.stride,padding=layer.padding)
        
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
        
    #transpose conv:
    if(isinstance(layer.stride, int)):
        if(layer.stride>1):
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding,
                                        output_padding=(1,1))
        else:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding)
    else:
        if(layer.stride[0]>1 or layer.stride[1]>1):
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding,
                                        output_padding=(1,1))
        else:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding)
            
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)

    return RJ

def LRPBNConvReLU(layer,BN,rK,aJ,aK,e,Ch0=0,Ch1=None):
    #used to propagate relevance through the sequence: Batchnorm, Convolution, ReLU
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L BN input
    #aK: activations after convolution
    #BN: batch normalization layer
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the concolutional layer
    
    aK=aK[:,Ch0:Ch1,:,:]
    rK=rK[:,:,Ch0:Ch1,:,:]
    
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    #BN parameters:
    BNChannels=aK.shape[1]
    
    weights=FuseBN(layerWeights=layer.weight, BN=BN, aKConv=aJ,
                   Ch0=Ch0,Ch1=Ch1,layerBias=layer.bias,
                   bias=False,BNbeforeReLU=False)
    #no detach due to small border bias inconsistency with padding
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
        
    #transpose conv:
    if(isinstance(layer.stride, int)):
        if(layer.stride>1):
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding,
                                        output_padding=(1,1))
        else:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding)
    else:
        if(layer.stride[0]>1 or layer.stride[1]>1):
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding,
                                        output_padding=(1,1))
        else:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                       stride=layer.stride,padding=layer.padding)
            
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=torch.mul(AJ,c)

    return RJ

def w2RuleInput(layer,rK,aJ,aK,e):
    #used to propagate relevance through first convolutional layer using w2 rule
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: layer activations
    
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    if(layer.bias is None):
        bias=torch.zeros(rK.shape[3])
    else:
        bias=layer.bias
    
    W2=torch.pow(layer.weight,2)
    B2=torch.pow(bias,2)
    AK=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK),weight=W2,
                            bias=B2,stride=layer.stride,padding=layer.padding)
    z=AK+stabilizer(aK=AK,e=e)
    z=z.unsqueeze(1).repeat(1,numOutputs,1,1,1)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k       
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
    c=nn.functional.conv_transpose2d(s,weight=W2,bias=None,
                                    stride=layer.stride,padding=layer.padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=c

    return RJ


def w2BNRuleInput(layer,BN,rK,aJ,aK,e,aKConv):
    #used to propagate relevance through first convolutional+BN layer using w2 rule
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN, list (one for each BN layer)
    
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    if(layer.bias is None):
        bias=torch.zeros(rK.shape[3])
    else:
        bias=layer.bias

    weights,biases=FuseBN(layerWeights=layer.weight, BN=BN, aKConv=aKConv,
                          Ch0=Ch0,Ch1=Ch1,layerBias=layer.bias,
                          bias=True)    
    
    W2=torch.pow(weights,2)
    B2=torch.pow(biases,2)
    AK=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK),weight=W2,
                            bias=B2,stride=layer.stride,padding=layer.padding)
    z=AK+stabilizer(aK=AK,e=e)
    z=z.unsqueeze(1).repeat(1,numOutputs,1,1,1)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k       
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
    c=nn.functional.conv_transpose2d(s,weight=W2,bias=None,
                                    stride=layer.stride,padding=layer.padding)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        
    RJ=c

    return RJ


def ZbRuleConvBNInput(layer,BN,rK,aJ,aK,aKConv,e,l=0,h=1,Zb0=False):
    #used to propagate relevance through the sequence: Convolution, Batchnorm, ReLU using Zb rule
    #l and h: minimum and maximum allowed pixel values
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN
    #aKConv: convolution output
    #BN: batch normalization layer
    #Zb0: removes stabilizer term
    
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    weights,biases=FuseBN(layerWeights=layer.weight, BN=BN, aKConv=aKConv,
                          layerBias=layer.bias,
                          bias=True)
        

    #positive and negative weights:
    WPos=torch.max(weights,torch.zeros(weights.shape).type_as(rK))
    WNeg=torch.min(weights,torch.zeros(weights.shape).type_as(rK))
    #positive and negative bias:
    BPos=torch.max(biases,torch.zeros(biases.shape).type_as(rK))
    BNeg=torch.min(biases,torch.zeros(biases.shape).type_as(rK))
        
    #propagation:
    aKPos=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK)*l,weight=WPos,
                                bias=BPos*l,stride=layer.stride,padding=layer.padding)
    aKNeg=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK)*h,weight=WNeg,
                                bias=BNeg*h,stride=layer.stride,padding=layer.padding)
    
    
    if (Detached and BN.bias is not None):
        aK=nn.functional.conv2d(aJ,weights,biases,stride=layer.stride,padding=layer.padding)

    z=aK-aKPos-aKNeg
    if (not Zb0):
        z=z+stabilizer(aK=z,e=e)
        
    z=z.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    s=torch.div(rK,z)
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
    
    try:
        op=1#output padding
        c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                            stride=layer.stride,padding=layer.padding,output_padding=op)
    except:
        op=0
        c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                            stride=layer.stride,padding=layer.padding,output_padding=op)
        
    cPos=l*nn.functional.conv_transpose2d(s,weight=WPos,bias=None,stride=layer.stride,
                                          padding=layer.padding,output_padding=op)
    cNeg=h*nn.functional.conv_transpose2d(s,weight=WNeg,bias=None,stride=layer.stride,
                                          padding=layer.padding,output_padding=op)

    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    cPos=cPos.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    cNeg=cNeg.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1) 
    R0=torch.mul(AJ,c)-cPos-cNeg
    
    return R0

def ZbRuleConvInput(layer,rK,aJ,aK,e,l=0,h=1,Zb0=False):
    #used to propagate relevance through the sequence: Convolution, ReLU using Zb rule
    #l and h: minimum and maximum allowed pixel values
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN
    
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    weights=layer.weight
    
    if(layer.bias is not None):
        biases=layer.bias.detach()
    else:
        biases=torch.zeros(layer.out_channels).type_as(rK)

    #positive and negative weights:
    WPos=torch.max(weights,torch.zeros(weights.shape).type_as(rK))
    WNeg=torch.min(weights,torch.zeros(weights.shape).type_as(rK))
    #positive and negative bias:
    BPos=torch.max(biases,torch.zeros(biases.shape).type_as(rK))
    BNeg=torch.min(biases,torch.zeros(biases.shape).type_as(rK))
        
    #propagation:
    aKPos=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK)*l,weight=WPos,
                                bias=BPos*l,stride=layer.stride,padding=layer.padding)
    aKNeg=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK)*h,weight=WNeg,
                                bias=BNeg*h,stride=layer.stride,padding=layer.padding)
    
    if (Detached and layer.bias is not None):
        aK=nn.functional.conv2d(aJ,weights,biases,stride=layer.stride,padding=layer.padding)

    z=aK-aKPos-aKNeg
    
    if (not Zb0):
        z=z+stabilizer(aK=z,e=e)
        
    z=z.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    s=torch.div(rK,z)
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
    
    try:
        op=1
        c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                            stride=layer.stride,padding=layer.padding,output_padding=op)
        
    except:
        op=0
        c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                            stride=layer.stride,padding=layer.padding,output_padding=op)
        
    cPos=l*nn.functional.conv_transpose2d(s,weight=WPos,bias=None,stride=layer.stride,
                                          padding=layer.padding,output_padding=op)
    cNeg=h*nn.functional.conv_transpose2d(s,weight=WNeg,bias=None,stride=layer.stride,
                                          padding=layer.padding,output_padding=op)
        
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    cPos=cPos.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    cNeg=cNeg.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1) 
    R0=torch.mul(AJ,c)-cPos-cNeg
    
    return R0

def LRPMaxPool2d(layer,rK,aJ,aK,e):
    #propagates relevance through max pooling (preceded by ReLU) or max pool+ReLU 
    #layer: pooling layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L pooling input
    #aK: activations before ReLU
    
    #get max pooling indexes:
    indexes=aK[1]
    #output of pooling:
    aK=aK[0]
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=rK.shape[2]
    
    kernel_size=layer.kernel_size
    stride=layer.stride
    padding=layer.padding
        
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        

    #Unpool: (propagate relevance through Maxpool)
    #begins with output classes dimension, then batch dimension
    s=s.permute(1,0,2,3,4)
    #change to classes,batches:
    s=s.reshape(numOutputs*batchSize,s.shape[-3],s.shape[-2],s.shape[-1])
    indexes=indexes.repeat(numOutputs,1,1,1)
    #unpool:
    with torch.cuda.amp.autocast(enabled=False):
        c=nn.functional.max_unpool2d(s.float(),indices=indexes,kernel_size=layer.kernel_size,
                                     stride=layer.stride,padding=layer.padding,
                                     output_size=(s.shape[0],aJ.shape[1],aJ.shape[2],aJ.shape[3]))
    #reshape:   
    c=c.view(numOutputs,batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
    #change to batches, classes again:
    c=c.permute(1,0,2,3,4)                   

    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)  
    
    RJ=torch.mul(AJ,c)

    return RJ



def MultiBlockMaxPoolBNReLU(layer,BN,rK,aJ,aK,e,aKPool,Ch0=0,Ch1=None):
    #propagates relevance through max pooling followed by batch normalization and ReLU 
    #in beginning of DenseNet, considering the BN, ReLU in the first block
    #layer: pooling layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L pooling input
    #aK: activations after each BN, list
    #aKPool: pooling output, containing maxpool indexes
    #BN: batch normalization layers, list of first BN layers for each dense layer in the next dense block
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the pooling layer
    
    #get max pooling indexes:
    indexes=aKPool[1]
    #output of pooling:
    aKPool=aKPool[0]
    
    for i,r in enumerate(rK,0):
        if(Ch1==rK[i].shape[2]):
            TCh1=None
        else: 
            TCh1=Ch1
        rK[i]=rK[i][:,:,Ch0:TCh1,:,:]
    rK=torch.cat(rK,dim=2)
    #size of batch dimension (0):
    batchSize=aKPool.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=aKPool.shape[1]
    
    #identity weights for convolution (to get BN weights):
    stride=(1,1)
    bias=torch.zeros(channels).type_as(rK)
    weights=torch.zeros(channels,channels,1,1).type_as(rK)
    for i in range(channels):
        weights[i,i,:,:]=torch.ones(1,1).type_as(rK)
        
    #consider BN: Fuse with each BN layer in list:
    W=[]
    B=[]
    for i,norm in enumerate(BN,0):
        w,b=FuseBN(layerWeights=weights,layerBias=bias, BN=BN[i], aKConv=aKPool,
                   Ch0=Ch0,Ch1=Ch1,bias=True)
        W.append(w)
        B.append(b)
        
    weights=torch.cat(W,dim=0)
    biases=torch.cat(B,dim=0)

    if (Detached and BN[0].bias is not None):
        aK=nn.functional.conv2d(aKPool,weights,biases)
    else:
        #concatenate ak in channels dimension:
        for i,a in enumerate(aK,0):
            if(Ch1==aK[i].shape[1]):
                TCh1=None
            else: 
                TCh1=Ch1
            aK[i]=aK[i][:,Ch0:TCh1,:,:]
        aK=torch.cat(aK,dim=1)
        
    aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

    z=aK+stabilizer(aK=aK,e=e)
    #element-wise inversion:s
    s=torch.div(rK,z)
    #shape: batch,o,k
        
    s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

    #transpose conv: (propagate relevance through BN)
    c=nn.functional.conv_transpose2d(s,weight=weights,stride=stride)
    c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
    
    #Unpool: (propagate relevance through Maxpool)
    #begins with output classes dimension, then batch dimension
    c=c.permute(1,0,2,3,4)
    #change to classes,batches:
    c=c.reshape(numOutputs*batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
    indexes=indexes.repeat(numOutputs,1,1,1)
    #unpool:
    with torch.cuda.amp.autocast(enabled=False):
        c=nn.functional.max_unpool2d(c.float(),indices=indexes,kernel_size=layer.kernel_size,
                                     stride=layer.stride,padding=layer.padding,
                                     output_size=(c.shape[0],aJ.shape[1],aJ.shape[2],aJ.shape[3]))
    #reshape:   
    c=c.view(numOutputs,batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
    #change to batches, classes again:
    c=c.permute(1,0,2,3,4)

    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)  
    
    RJ=torch.mul(AJ,c)

    return RJ

def AppendOutput(self,input,output):
    #forward hook to save layer output
    
    globals.X.append(output)
    
def AppendInput(self,input,output):
    #forward hook to save layer input
    
    globals.XI.append(input[0])
    
def InsertHooks(m: torch.nn.Module):
    #Function to insert multiple forward hooks in the classifier, for later use in the LRP block
    #m: classifier
    
    children = dict(m.named_children())
    output = {}
    if children == {}:
        m.register_forward_hook(AppendOutput)
        l=globals.LayerIndex
        globals.LayerIndex=globals.LayerIndex+1
        return (m,l)
    else:
        for name, child in children.items():
            try:
                output[name] = InsertHooks(child)
            except TypeError:
                output[name] = InsertHooks(child)
    return output

def ChangeE(m: torch.nn.Module,e):
    #function to change all LRP-e e values in the network
    #m: ISNet
    
    children = dict(m.named_children())
    output = {}
    if hasattr(m,'e'):
        m.e=e
    if children == {}:
        return (m)
    else:
        for name, child in children.items():
            try:
                output[name] = ChangeE(child,e)
            except TypeError:
                output[name] = ChangeE(child,e)
    return output

def ChangeInplace(m: torch.nn.Module):
    #function to change all LRP-e e values in the network
    #m: ISNet
    
    children = dict(m.named_children())
    output = {}
    if hasattr(m,'inplace'):
        m.inplace=False
    if children == {}:
        return (m)
    else:
        for name, child in children.items():
            try:
                output[name] = ChangeInplace(child)
            except TypeError:
                output[name] = ChangeInplace(child)
    return output

def resetGlobals():
    #reset all global variables
    globals.LayerIndex=0
    globals.X=[]
    globals.XI=[]
    globals.t=0
    globals.mean_l=0
    globals.mean_L=0
    globals.Ml=0
    
def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)
            
def RemoveLRPBlock(ISNet):
    model=ISNet.DenseNet
    remove_all_forward_hooks(model)
    return model
