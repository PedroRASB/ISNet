import torch
import torch.nn as nn
import globalsZe as globals
from collections import OrderedDict
from typing import Dict, Callable
import copy
import random

def GlobalWeightedRankPooling(x,d=0.9,descending=True,oneD=False,rank=True):
    #GWRP, performs weighted average in spatial dimensions https://arxiv.org/abs/1809.08264
    #x: input
    #d: rate of weights' exponential decay
    #descending: if true elements are ordered in descending order before weighting, if false, in ascending order
    #oneD: set to True only for aggregating LDS loss, aggregates one-dimensional vectors instead of
    #2D objects
    #rank: allows ordering of elements. If False, elements are aggregated in the provided order
    
    if len(x.shape)==5 and not oneD:#2D pool
        x=x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3]*x.shape[4])
    if rank:
        x,_=torch.sort(x,dim=-1,descending=descending)
    weights=torch.tensor([d ** i for i in range(x.shape[-1])]).type_as(x)
    while len(weights.shape)<len(x.shape):
        weights=weights.unsqueeze(0)
    x=torch.mul(x,weights)
    x=x.sum(-1)/weights.sum(-1)
    return x


def LRPLossCEValleysGWRP (heatmap, mask, cut=1, cut2=25, reduction='mean', 
                         norm='absRoIMean', A=1, B=3, E=1,d=0.9,
                         eps=1e-10,rule='e',tuneCut=False,
                         channelGWRP=1.0,
                         alternativeForeground='L2'):
    #ISNet Loss Function
    #heatmap: heatmaps to be optimized
    #mask: foreground segmentation masks, should be 1 in foreground and 0 in background
    #cut1 and cut2: C1 and C2 parameters, should define natural range of total absolute heatmap relevance
    #reduction: how the loss is reduced
    #norm: 'absRoIMean' uses standard ISNet loss, 'RoIMean' applies the background loss normalization step before the absolute value operation
    #A and B: w1 and w2 parameters, weights for the background and foreground loss
    #E: parameter of the background loss activation x/(x+E)
    #d: GWRP exponential decay parameter
    #channelGWRP: exponential decay of 1D GWRP to aggregate heatmap losses of each channel in LDS, set to 1 in paper (simple mean)
    #alternativeForeground='hybrid' adds the loss modification in the paper Faster ISNet for Background Bias Mitigation on Deep Neural Networks, alternativeForeground='L2' uses the standard (original) loss
    
    if len(heatmap.shape)!=5:
        raise ValueError('Incorrect heatmap format, correct is: batch, classes, channels, wdith, lenth')
    
    if isinstance(alternativeForeground, bool):#backward compat
        if alternativeForeground:
            alternativeForeground='L1'
        else:
            alternativeForeground='L2'
            
    
    if isinstance(norm, bool):#backward compat
        if norm:
            norm='absRoIMean'
        else:
            norm='none'
        

    batchSize=heatmap.shape[0]
    classesSize=heatmap.shape[1]
    channels=heatmap.shape[2]
    length=heatmap.shape[-1]
    width=heatmap.shape[-2]
    
    #create copies to avoid changing argument variables
    cut=copy.deepcopy(cut)
    cut2=copy.deepcopy(cut2)
    mask=mask.clone()
    heatmap=heatmap.clone()
    
    #resize masks to match heatmap shape if necessary
    if (mask.shape[-2]!=width or mask.shape[-1]!=length):
        mask=torch.nn.functional.adaptive_avg_pool2d(mask, [heatmap.shape[-2],heatmap.shape[-1]])
        #ensure binary:
        mask=torch.where(mask==0.0,0.0,1.0)#conservative approach, only minimize attention to regions we have no foreground
    if not torch.equal((torch.where(mask==0.0,1.0,0.0)+torch.where(mask==1.0,1.0,0.0)),
                    torch.ones(mask.shape).type_as(mask)):
        non_binary_elements = mask[(mask != 0) & (mask != 1)]
        print("Non-binary elements:", non_binary_elements)
        raise ValueError('Non binary mask')
    if len(mask.shape)!=len(heatmap.shape):
        mask=mask.unsqueeze(1).repeat(1,classesSize,1,1,1)
    if mask.shape[2]!=channels:
        mask=mask[:,:,0,:,:].unsqueeze(2).repeat(1,1,channels,1,1)
    if mask.sum().item()==0:
        print('Zero mask')
    
        
    #inverse mask:
    Imask=torch.ones(mask.shape).type_as(mask)-mask
    

    with torch.cuda.amp.autocast(enabled=False):
        heatmap=heatmap.float()
        mask=mask.float()
        Imask=Imask.float()
        
        #substitute nans if necessary:
        if torch.isnan(heatmap).any():
            print('nan 0')
        if torch.isinf(heatmap).any():
            print('inf 0')
        RoIMean=torch.sum(torch.nan_to_num(heatmap,posinf=0.0,neginf=0.0)*mask)/(torch.sum(mask)+eps)
        #print(RoIMean)
        heatmap=torch.nan_to_num(heatmap,nan=RoIMean.item(), 
                                posinf=torch.max(torch.nan_to_num(heatmap,posinf=0,neginf=0)).item(),
                                neginf=torch.min(torch.nan_to_num(heatmap,posinf=0,neginf=0)).item())
        
        #save non-normalized heatmap:
        heatmapRaw=torch.abs(heatmap).clone()
        
        if A!=0:
            if torch.isnan(heatmap).any():
                print('nan 1')
            if torch.isinf(heatmap).any():
                print('inf 1')
            #normalize heatmap:
            if norm=='absRoIMean':
                #abs:
                heatmap=torch.abs(heatmap)
                #roi mean value:
                denom=torch.sum(heatmap*mask, dim=(-1,-2,-3),
                                keepdim=True)/(torch.sum(mask,dim=(-1,-2,-3),keepdim=True)+eps)
                if torch.isnan(denom).any():#nan in denom
                    print('nan 0.5')
                if torch.isinf(denom).any():
                    print('inf 0.5')
                heatmap=heatmap/(denom+eps)
                #print('heatmap:',heatmap.shape, 'denom', denom.shape)
            
            elif norm=='RoIMean':
                #print('running new norm')
                #roi mean value (no abs):
                denom=torch.sum(heatmap*mask, dim=(-1,-2,-3),
                                keepdim=True)/(torch.sum(mask,dim=(-1,-2,-3),keepdim=True)+eps)
                if torch.isnan(denom).any():#nan in denom
                    print('nan 0.5')
                if torch.isinf(denom).any():
                    print('inf 0.5')
                heatmap=heatmap/(denom+eps)
                #abs:
                heatmap=torch.abs(heatmap)  
                
            elif norm=='none':
                #abs:
                heatmap=torch.abs(heatmap)
                heatmap=heatmap*(channels*length*width)
            else:
                raise ValueError('Unrcognized norm')

            if torch.isnan(heatmap).any():
                print('nan 2')
            if torch.isinf(heatmap).any():
                print('inf 2')

            #Background:
            heatmapBKG=torch.mul(heatmap,Imask)
            heatmapBKG=GlobalWeightedRankPooling(heatmapBKG,d=d)
            #activation:
            heatmapBKG=heatmapBKG/(heatmapBKG+E)
            #cross entropy (pixel-wise):
            heatmapBKG=torch.clamp(heatmapBKG,max=1-1e-7)
            loss=-torch.log(torch.ones(heatmapBKG.shape).type_as(heatmapBKG)-heatmapBKG)
            if channelGWRP==1.0:
                loss=torch.mean(loss,dim=(-1,-2))#channels and classes mean
            else:#GWRP over channel loss, penalizing more channels with with background attention
                loss=GlobalWeightedRankPooling(loss,d=channelGWRP,oneD=True)#reduce over channel dimension
                loss=torch.mean(loss,dim=-1)#reduce over classes dimension

            if torch.isnan(loss).any():
                print('nan 3')
            if torch.isinf(loss).any():#here
                print('inf 3')
                if (not torch.isinf(heatmapBKG).any()):
                    print('inf by log')
        else:
            loss=0
            
        
        if tuneCut: #use for finding ideal cut values
            if (reduction=='sum'):
                loss=torch.sum(loss)
            elif (reduction=='mean'):
                loss=torch.mean(loss)
            elif (reduction=='none'):
                pass
            else:
                raise ValueError('reduction should be none, mean or sum')
            #return loss,(heatmapRaw*mask).sum(dim=(-1,-2,-3))
            return loss,(heatmapRaw).sum(dim=(-1,-2,-3))
            
        if B!=0:
            #avoid foreground values too low or too high:
            heatmapF=torch.mul(heatmapRaw,mask).sum(dim=(-1,-2,-3))
            #divide heatmapF by cut, same as dividing square losses by cut**2, but avoids underflow
            if (rule=='e' and isinstance(cut, list)):
                cut=cut[0]
                cut2=cut2[0]
                
            #Set targets
            if rule=='z+e':
                #for z+e, the z+ and the e heatmaps can be at different scales,
                #provide cut0=[cut0 for LRP-z+,cut0 for LRP-e]
                #and cut2=[cut2 for LRP-z+,cut2 for LRP-e]
                shape=(heatmapF.shape[0],int(heatmapF.shape[1]/2))
                target=torch.cat((cut[0]*torch.ones(shape).type_as(heatmapF),
                                  cut[1]*torch.ones(shape).type_as(heatmapF)),dim=1)
                target2=torch.cat((cut2[0]*torch.ones(shape).type_as(heatmapF),
                                   cut2[1]*torch.ones(shape).type_as(heatmapF)),dim=1)
            else:                
                target=cut*torch.ones(heatmapF.shape).type_as(heatmapF)
                target2=cut2*torch.ones(heatmapF.shape).type_as(heatmapF)
            #print(heatmapF)
            
            #left side: lossF
            #target/target=1
            lossF=nn.functional.mse_loss(torch.clamp(heatmapF/target,min=None,
                                                     max=torch.ones(target.shape).type_as(target)),
                                         torch.ones(target.shape).type_as(target),reduction='none')

            lossF=torch.mean(lossF,dim=-1)#classes mean

            #right side: lossF2
            if alternativeForeground=='L2':
                lossF2=nn.functional.mse_loss(torch.clamp(heatmapF/target,
                                                          min=target2/target,max=None),
                                              target2/target,reduction='none')
            elif alternativeForeground=='L1':
                #uses not scaled L1 loss in the right side of the foreground loss
                lossF2=nn.functional.l1_loss(torch.clamp(heatmapF,min=target2,max=None),
                                              target2,reduction='none')
            elif alternativeForeground=='hybrid':
                #symetric L2 loss around [C1,C2] range, then L1 loss
                #mask:what elements are higher than C1+C2?
                high=torch.where(heatmapF.detach()>(target+target2),1.0,0.0).type_as(heatmapF)#use L1
                low=1.0-high#use L2
                L2=nn.functional.mse_loss(torch.clamp(low*(heatmapF/target),
                                                          min=target2/target,max=None),
                                              target2/target,reduction='none')
                L1=nn.functional.l1_loss(torch.clamp(high*heatmapF,min=(target2+target),max=None),
                                              (target2+target),reduction='none')
                L1=L1+torch.ones(target.shape).type_as(target)
                lossF2=L2*low+L1*high
            else:
                raise ValueError('Unrecognized alternativeForeground')

            lossF2=torch.mean(lossF2,dim=-1)#classes mean

            lossF=lossF+lossF2


            loss=A*loss+B*lossF
        
        if torch.isnan(loss).any():
            print('nan 4')
        if torch.isinf(loss).any():
            print('inf 4')
        
        #reduction of batch dimension
        if (reduction=='sum'):
            loss=torch.sum(loss)
        elif (reduction=='mean'):
            loss=torch.mean(loss)
        elif (reduction=='none'):
            pass
        else:
            raise ValueError('reduction should be none, mean or sum')

        return loss

def stabilizer(aK,e,sign=None):
    #Used for LRP-e, returns terms sign(ak)*e
    #ak: values to stabilize
    #e: LRP-e term 
    #sign: 1, -1 or None
    
    if(sign is None):
        signs=torch.sign(aK)
        #zeros as positives
        signs[signs==0]=1
        signs=signs*e
    else:
        signs=sign*torch.ones(aK.shape).type_as(aK)*e
        
    return signs

def PosNeg(x):
    #splits positive and negative parts of a signal x
    xp=torch.max(x,torch.zeros(x.shape).type_as(x))
    xn=torch.min(x,torch.zeros(x.shape).type_as(x))
    return xp,xn

def LRPOutput(layer,aJ,y,positive,multiple,rule,ignore=None,amplify=1,highest=False,label=None,
              randomLogit=False):
    #returns output relevance, create a new batch dimenison for classes,
    #of size C, the number of DNN outputs/classes. Returns diagonal matrix.
    #y:classifier output
    #positive: deprecated
    #multiple: creates one map per class if True
    #ignore: lsit with classes which will not suffer attention control
    
    #check parameters:
    tmp=0
    for i in [(label is not None),randomLogit, multiple, highest]:
        if i:
            tmp+=1
    if tmp>1:
        raise ValueError('Choose label, randomLogit, multiple or highest')
    
    
    if(rule=='z+e'):
        if multiple:
            raise ValueError('Not implemented')
        ROp=LRPOutput(layer,aJ,y,positive=False,multiple=False,rule='z+',
                      ignore=ignore,amplify=amplify)
        ROe=LRPOutput(layer,aJ,y,positive=False,multiple=False,rule='e',
                      ignore=ignore,amplify=amplify)
        RO=torch.cat((ROp,ROe),dim=1)#one map for each rule, in the classes dim
        return RO

    #recreate y with detached bias
    weights=layer.weight
    if (layer.bias is not None):
        bias=layer.bias.detach()
    else:
        bias=None
    if (rule=='z+'):
        if(torch.min(aJ)<0):
            raise ValueError('negative aJ elements input and z+ rule, not implemented for this case')
        weights=torch.max(weights,torch.zeros(weights.shape).type_as(weights))
        if (bias is not None):
            bias=torch.max(bias,torch.zeros(bias.shape).type_as(bias))
        else:
            bias=None
    if ((bias is not None and globals.detach) or rule=='z+'):
        aK=nn.functional.linear(aJ,weights,bias)
    else:
        aK=y.clone()
    
    numOutputs=aK.shape[-1]
    ones=torch.ones((numOutputs)).type_as(aK)
    if (ignore is not None):
        for i,_ in enumerate(ones,0):
            if(i in ignore):
                ones[i]=ones[i]*0
                
    if rule=='z+' or rule=='e':
        if(multiple):
            #define identity matrix:
            I=torch.diag(ones)
            #repeat the outputs and element-wise multiply with identity
            RO=torch.mul(aK.unsqueeze(-1).repeat(1,1,numOutputs),I)
       
        else:
            if label is not None:
                print('Label logit being used, do not use for training ISNet')
                #propagate highest logit
                I=torch.diag(ones)
                #repeat the outputs and element-wise multiply with identity
                RO=torch.mul(aK.unsqueeze(-1).repeat(1,1,numOutputs),I)
                idx=label
                tmp=[]
                for i,val in enumerate(idx,0):
                    tmp.append(RO[i,val,:])
                RO=torch.stack(tmp,dim=0)#batch dimension
                RO=RO.unsqueeze(1)#add classes dimension back
            elif randomLogit:
                I=torch.diag(ones)
                #repeat the outputs and element-wise multiply with identity
                RO=torch.mul(aK.unsqueeze(-1).repeat(1,1,numOutputs),I)
                if random.randint(1,10)>5:#propagate logit for highest class, 50% chance
                    #propagate highest logit
                    idx=torch.argmax(y,dim=-1)
                    tmp=[]
                    for i,val in enumerate(idx,0):
                        tmp.append(RO[i,val,:])
                    RO=torch.stack(tmp,dim=0)#batch dimension
                    RO=RO.unsqueeze(1)#add classes dimension back
                else:#select logit randomly in other classes
                    idx=torch.argmax(y,dim=-1)
                    tmp=[]
                    for i,val in enumerate(idx,0):
                        rand=random.randint(0,RO.shape[1]-1)
                        while (rand==val):
                            rand=random.randint(0,(RO.shape[1]-1))#select random class
                        tmp.append(RO[i,rand,:])
                    RO=torch.stack(tmp,dim=0)#batch dimension
                    RO=RO.unsqueeze(1)#add classes dimension back
            elif highest:
                #propagate highest logit
                I=torch.diag(ones)
                #repeat the outputs and element-wise multiply with identity
                RO=torch.mul(aK.unsqueeze(-1).repeat(1,1,numOutputs),I)
                idx=torch.argmax(y,dim=-1)
                tmp=[]
                for i,val in enumerate(idx,0):
                    tmp.append(RO[i,val,:])
                RO=torch.stack(tmp,dim=0)#batch dimension
                RO=RO.unsqueeze(1)#add classes dimension back
            else:
                #propagate every logit simultaneously
                RO=torch.mul(aK,ones).unsqueeze(1)
            
    if amplify!=1:
        RO=RO*amplify
        
    if rule!='z+' and rule!='e' and rule!='z+e':
        raise ValueError('Not implemented')
    
    #print(RO.shape)
    return RO

def LRPDenseReLU(layer,rK,aJ,aK,e,weights=None,bias=None,rule='e',alpha=2,beta=-1):
    #Propagates relevance through fully-connected layer
    #layer: layer L throgh which we propagate relevance, fully-connected
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
    #aK: activations before ReLU
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    if layer is not None:
        weights=layer.weight
        if (layer.bias is not None):
            bias=layer.bias.detach()
        else:
            bias=None
    
    if (rule=='e' or rule=='z+'):
        if (rule=='z+'):
            if(torch.min(aJ)<0):
                raise ValueError('negative aJ elements in MaxPool input and z+ rule')
            weights=torch.max(weights,torch.zeros(weights.shape).type_as(weights))
            if (bias is not None):
                bias=torch.max(bias,torch.zeros(bias.shape).type_as(bias))
            else:
                bias=None
            
        if ((bias is not None and globals.detach) or rule=='z+'):
            aK=nn.functional.linear(aJ,weights,bias)
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
        AJ=aJ.unsqueeze(1)
        if numOutputs>1:
            AJ=AJ.repeat(1,numOutputs,1)  
            
        #print(AJ.shape,c.shape)
        RJ=torch.mul(AJ,c)
        
    elif (rule=='z+e'):
        #numOutputs should be 2, considering one map per rule. set it to half:
        numOutputs=int(numOutputs/2)
        
        if(torch.min(aJ)>=0):
            #raise ValueError('negative aJ elements in MaxPool input and z+ rule')
            weights_p=torch.max(weights,torch.zeros(weights.shape).type_as(weights))

            if (bias is not None and globals.detach):
                biases_p=torch.max(bias,torch.zeros(bias.shape).type_as(bias))
                aKe=nn.functional.linear(aJ,weights,bias)

            else:
                biases_p=None
                aKe=aK
            aKp=nn.functional.linear(aJ,weights_p,biases_p)

            aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1)
            aKe=aKe.unsqueeze(1).repeat(1,numOutputs,1)

            zp=aKp+stabilizer(aK=aKp,e=e)
            ze=aKe+stabilizer(aK=aKe,e=e)
            #element-wise inversion:s
            sp=torch.div(rK[:,:numOutputs,:],zp)
            se=torch.div(rK[:,numOutputs:,:],ze)
            #shape: batch,o,k

            W=torch.transpose(weights,0,1)
            Wp=torch.transpose(weights_p,0,1)
            #mix batch and class dimensions
            sp=sp.view(batchSize*numOutputs,sp.shape[-1])
            se=se.view(batchSize*numOutputs,se.shape[-1])
            #back relevance with transposed weigths
            cp=nn.functional.linear(sp,Wp)
            ce=nn.functional.linear(se,W)
            #unmix:
            ce=ce.view(batchSize,numOutputs,ce.shape[-1])
            cp=cp.view(batchSize,numOutputs,cp.shape[-1])

            #add 2 rules to classes dimension
            c=torch.cat((cp,ce), dim=1)

            #add classes dimension:
            AJ=aJ.unsqueeze(1).repeat(1,numOutputs*2,1)   
            RJ=torch.mul(AJ,c)
            
        else:
            RJp=LRPDenseReLU(layer,rK[:,:numOutputs,:],aJ,aK,e,rule='AB',alpha=1,beta=0,
                            weights=weights,bias=bias)
            RJe=LRPDenseReLU(layer,rK[:,numOutputs:,:],aJ,aK,e,rule='e',
                            weights=weights,bias=bias)
            RJ=torch.cat((RJp,RJe),dim=1)
    
    elif (rule=='AB'):
        if((alpha+beta)!=1):
            raise ValueError('Ensure alpha+beta=1')
        #positive and negative parts
        Wp,Wn=PosNeg(weights)
        if bias is not None:
            Bp,Bn=PosNeg(bias)
        else:
            Bp,Bn=None,None
        aJp,aJn=PosNeg(aJ)
        
        #step 1- Forward pass
        
        aKp=nn.functional.linear(aJp,Wp,Bp)+nn.functional.linear(aJn,Wn)
        aKn=nn.functional.linear(aJn,Wp,Bn)+nn.functional.linear(aJp,Wn)
        
        aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1)
        aKn=aKn.unsqueeze(1).repeat(1,numOutputs,1)
        
        #step 2- Division and stabilizer
        zp=aKp+stabilizer(aK=aKp,e=e,sign=1)
        sp=torch.div(rK,zp)
        
        zn=aKn+stabilizer(aK=aKn,e=e,sign=-1)
        sn=torch.div(rK,zn)
        
        #step 3- Transpose pass
        Wp=torch.transpose(Wp,0,1)
        Wn=torch.transpose(Wn,0,1)
        sp=sp.view(batchSize*numOutputs,sp.shape[-1])
        sn=sn.view(batchSize*numOutputs,sn.shape[-1])
        
        cp_a=nn.functional.linear(sp,Wp)
        cp_b=nn.functional.linear(sp,Wn)
        cn_a=nn.functional.linear(sn,Wp)
        cn_b=nn.functional.linear(sn,Wn)
        
        cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-1])
        cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-1])
        cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-1])
        cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-1])
        
        #step 4- Multiplication with input
        aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1)
        aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1)
        RJ=alpha*(torch.mul(aJp.squeeze(1),cp_a.squeeze(1))+\
                  torch.mul(aJn.squeeze(1),cp_b.squeeze(1)))
        RJ=RJ+beta*(torch.mul(aJn.squeeze(1),cn_a.squeeze(1))+\
                    torch.mul(aJp.squeeze(1),cn_b.squeeze(1)))
        if(len(RJ.shape)<5):
            RJ=RJ.unsqueeze(1)
        

    else:
        raise ValueError('only Epsilon (e), Alpha Beta (AB), and z+ rules implemented')
        
    return RJ



def ZbRuleDenseInput(layer,rK,aJ,aK,e,l=0,h=1):
    #used to propagate relevance through the sequence: Convolution, ReLU using Zb rule
    #l and h: minimum and maximum allowed pixel values
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN
    #print(rK.shape,aJ.shape,aK.shape,layer)
    
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    weights=layer.weight
    
    if(layer.bias is not None):
        biases=layer.bias.detach()
    else:
        biases=torch.zeros(layer.out_features).type_as(rK)

    #positive and negative weights:
    Wpos,Wneg=PosNeg(weights)
    #positive and negative bias:
    BPos,BNeg=PosNeg(layer.bias.detach())
        
    #propagation:
    aKPos=nn.functional.linear(torch.ones(aJ.shape).type_as(rK)*l,weight=Wpos,
                                bias=BPos*l)
    aKNeg=nn.functional.linear(torch.ones(aJ.shape).type_as(rK)*h,weight=Wneg,
                                bias=BNeg*h)
    
    if (layer.bias is not None and globals.detach):
        aK=nn.functional.linear(aJ,weights,biases)

    z=aK-aKPos-aKNeg
        
    z=z.unsqueeze(1).repeat(1,numOutputs,1)
    #print('z',z.shape)
    
    z=z+stabilizer(z,e=e)
        
    s=torch.div(rK,z)
    #print(s.shape)
    #print(numOutputs)
    
    W=torch.transpose(weights,0,1)
    Wpos=torch.transpose(Wpos,0,1)
    Wneg=torch.transpose(Wneg,0,1)
    
    #mix batch and class dimensions
    s=s.view(batchSize*numOutputs,s.shape[-1])
    #back relevance with transposed weigths
    c=nn.functional.linear(s,W)
    cPos=l*nn.functional.linear(s,Wpos)
    cNeg=h*nn.functional.linear(s,Wneg)
    #unmix:
    c=c.view(batchSize,numOutputs,c.shape[-1])
    cPos=cPos.view(batchSize,numOutputs,c.shape[-1])
    cNeg=cNeg.view(batchSize,numOutputs,c.shape[-1])
    AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1) 
    R0=torch.mul(AJ,c)-cPos-cNeg
    
    
    return R0

def LRPClassSelectiveOutputLayerOld(layer, aJ, aK, e, highest=False,amplify=1,label=None):
    #non lse
    #Explains a more class selective quantity, instead of the logits
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input

    #print(aJ.shape,aK.shape)    

    #size of batch dimension (0):
    batchSize=aJ.shape[0]
    numOutputs=layer.weight.shape[0]

    Zcc=[]
    for c in list(range(numOutputs)):
        #Zcc=Zc-Zc'
        #forward pass:
        weight=layer.weight[c].unsqueeze(0).repeat(numOutputs,1)-layer.weight
        if layer.bias is not None:
            bias=layer.bias[c].repeat(numOutputs)-layer.bias
            if globals.detach:
                bias=bias.detach()
        else:
            bias=None
        Zcc.append(nn.functional.linear(aJ,weight,bias))
    #stack in the new classes dimension
    Zcc=torch.stack(Zcc,dim=1)

    #backward pass to Zcc:
    eZcc=torch.exp(-Zcc)
    Rcc=Zcc*eZcc

    #print('eZcc:',eZcc)
    #Pc=eZcc/eZcc.sum(dim=-1,keepdim=True)
    #print('Pc:',Pc)
    #Nc=(-1)*torch.log(eZcc.sum(dim=-1,keepdim=True)-1)
    #print('Nc:',Nc)
    #print('Zc:',nn.functional.linear(aJ,layer.weight,layer.bias))

    #-1 removes the element for c''=c, as it is e^0, as Zcc=0
    denom=eZcc.sum(dim=-1,keepdim=True).repeat(1,1,numOutputs)-1
    denom=denom+stabilizer(denom,e=1e-5)
    Rcc=Rcc/denom
    #print('Rcc:',Rcc)
    #print('Rcc sum:',Rcc.sum(-1))
    #akOl=aK.clone()
    #backward pass to aJ, Zcc'=Sumj(aj*(wjc-wjc'))
    RJ=[]
    for c in list(range(numOutputs)):
        weight=layer.weight[c].unsqueeze(0).repeat(numOutputs,1)-layer.weight
        if layer.bias is not None:
            bias=layer.bias[c].repeat(numOutputs)-layer.bias
            if globals.detach:
                bias=bias.detach()
        else:
            bias=None
        aK=Zcc[:,c,:]
        rK=Rcc[:,c,:].unsqueeze(1)
        RJ.append(LRPDenseReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,
                               weights=weight, bias=bias).squeeze(1))
    RJ=torch.stack(RJ,dim=1)

    if highest or (label is not None):
        #propagates only heatmap for highest logit
        if highest:
            idx=torch.argmax(aK,dim=-1)
        elif (label is not None):
            print('Label logit being used, do not use for ISNet training')
            idx=label
            print(idx, torch.argmax(aK,dim=-1))
        #print(torch.equal(idx,torch.argmin(akOl,dim=-1)))
        tmp=[]
        for i,val in enumerate(idx,0):
            tmp.append(RJ[i,val,:])
        RJ=torch.stack(tmp,dim=0)
        RJ=RJ.unsqueeze(1)#add classes dimension back
    
    if amplify!=1:
        RJ=RJ*amplify


    return RJ



def LRPClassSelectiveOutputLayerMultiple(layer, aJ, aK, e, highest=False,amplify=1):
    #used only for explanation, not for ISNet training
    #Explains a more class selective quantity, instead of the logits
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
        
    #print(aJ.shape,aK.shape)    
    
    #size of batch dimension (0):
    batchSize=aJ.shape[0]
    numOutputs=layer.weight.shape[0]
    
    Zcc=[]
    for c in list(range(numOutputs)):
        #Zcc=Zc-Zc'
        #forward pass:
        weight=layer.weight[c].unsqueeze(0).repeat(numOutputs,1)-layer.weight
        if layer.bias is not None:
            bias=layer.bias[c].repeat(numOutputs)-layer.bias
            if globals.detach:
                bias=bias.detach()
        else:
            bias=None
        Zcc.append(nn.functional.linear(aJ,weight,bias))
    #stack in the new classes dimension
    Zcc=torch.stack(Zcc,dim=1)
    
    #backward pass to Zcc:
    eZcc=torch.exp(-Zcc)
    #if (torch.isinf(eZcc).any()):
    #    eZcc=torch.exp(torch.minimum(-Zcc,11*torch.ones(Zcc.shape).type_as(Zcc)))
        #this maximum was added after isnetZ+E experiments for paper!!!
    Rcc=Zcc*eZcc
    
    #print('eZcc:',eZcc)
    #Pc=eZcc/eZcc.sum(dim=-1,keepdim=True)
    #print('Pc:',Pc)
    #Nc=(-1)*torch.log(eZcc.sum(dim=-1,keepdim=True)-1)
    #print('Nc:',Nc)
    #print('Zc:',nn.functional.linear(aJ,layer.weight,layer.bias))
    
    #-1 removes the element for c''=c, as it is e^0, as Zcc=0
    denom=eZcc.sum(dim=-1,keepdim=True).repeat(1,1,numOutputs)-1
    denom=denom+stabilizer(denom,e=1e-5)
    Rcc=Rcc/denom
    
    
    #print('Rcc:',Rcc)
    #print('Rcc sum:',Rcc.sum(-1))
    
    #backward pass to aJ, Zcc'=Sumj(aj*(wjc-wjc'))
    
    RJ=[]
    for c in list(range(numOutputs)):
        weight=layer.weight[c].unsqueeze(0).repeat(numOutputs,1)-layer.weight
        if layer.bias is not None:
            bias=layer.bias[c].repeat(numOutputs)-layer.bias
            if globals.detach:
                bias=bias.detach()
        else:
            bias=None
        #aK=Zcc[:,c,:]
        rK=Rcc[:,c,:].unsqueeze(1)
        RJ.append(LRPDenseReLU(layer=None,rK=rK,aJ=aJ,aK=Zcc[:,c,:],e=e,
                               weights=weight, bias=bias).squeeze(1))
    RJ=torch.stack(RJ,dim=1)
    
    #print('RJ:',RJ)
    #print('RJ sum:',RJ.sum(-1))
    
    if highest:
        #propagates only heatmap for highest logit
        idx=torch.argmax(aK,dim=-1)
        tmp=[]
        #zalt=[]
        for i,val in enumerate(idx,0):
            tmp.append(RJ[i,val,:])
            #zalt.append(Zcc[i,val,:])
        #zalt=torch.stack(zalt,dim=0)
        RJ=torch.stack(tmp,dim=0)
        RJ=RJ.unsqueeze(1)#add classes dimension back
        
        
    if amplify!=1:
        RJ=RJ*amplify
    
    
    return RJ#,zalt

def LRPClassSelectiveOutputLayer(layer, aJ, aK, e, highest=False,amplify=1,mode='lowest',label=None):
    #used only for explanation, not for ISNet training
    #Explains a more class selective quantity, instead of the logits
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
    
    if not highest and label is None:
        return LRPClassSelectiveOutputLayerMultiple(layer,aJ,aK,e,highest,amplify)
    if label is not None:
        return LRPClassSelectiveOutputLayerOld(layer,aJ,aK,e,highest,amplify,label=label)
        print('Using label logit')


    #size of batch dimension (0):
    batchSize=aJ.shape[0]
    numOutputs=layer.weight.shape[0]

    #highest:
    #idx=torch.argmax(aK,dim=-1)
    if mode=='random':
        idx=torch.randint(low=0,high=numOutputs,size=(aK.shape[0],))
    if mode=='random2':
        mx=torch.argmax(aK,dim=-1)
        mi=torch.argmin(aK,dim=-1)
        idx=[]
        for i,_ in enumerate(mx,0):
            if numOutputs>2:
                r=random.randint(1,3)
            else:
                r=random.randint(1,2)

            if r==1:
                idx.append(mx[i])
            elif r==2:
                idx.append(mi[i])
            elif r==3:
                tmp=random.randint(0,numOutputs-1)
                while (tmp==mi[i] or tmp==mx[i]):
                    tmp=random.randint(0,numOutputs-1)
                idx.append(tmp)
    if mode=='lowest':
        idx=torch.argmin(aK,dim=-1)
    if mode=='highest':
        sft=torch.softmax(aK,dim=-1)
        n=torch.log(sft/(1-sft))
        idx=torch.argmax(torch.abs(n),dim=-1)


    Zcc=[]
    for i,c in enumerate(idx,0):
    #for c in list(range(numOutputs)):
        #Zcc=Zc-Zc'
        #forward pass:
        weight=layer.weight[c].unsqueeze(0).repeat(numOutputs,1)-layer.weight
        if layer.bias is not None:
            bias=layer.bias[c].repeat(numOutputs)-layer.bias
            if globals.detach:
                bias=bias.detach()
        else:
            bias=None
        Zcc.append(nn.functional.linear(aJ[i].unsqueeze(0),weight,bias))
    #stack in the new classes dimension
    Zcc=torch.cat(Zcc,dim=0)
    if torch.isnan(Zcc).any():
        print('nan 1')

    #backward pass to Zcc:
    eZcc=torch.exp(-Zcc)
    #if (torch.isinf(eZcc).any()):
    #    eZcc=torch.exp(torch.minimum(-Zcc,11*torch.ones(Zcc.shape).type_as(Zcc)))
        #this maximum was added after isnetZ+E experiments for paper!!!
    Rcc=Zcc*eZcc
    if torch.isnan(eZcc).any():
        print('nan 2')
    #-1 removes the element for c''=c, as it is e^0, as Zcc=0
    denom=eZcc.sum(dim=-1,keepdim=True).repeat(1,numOutputs)-1
    if torch.isnan(denom).any():
        print('nan denom')
    if torch.isinf(denom).any():
        print('inf denom')
    denom=denom+stabilizer(denom,e=1e-5)
    Rcc=Rcc/denom
    if torch.isnan(Rcc).any():
        print('nan 3')
    RJ=[]
    for i,c in enumerate(idx,0):
    #for c in list(range(numOutputs)):
        weight=layer.weight[c].unsqueeze(0).repeat(numOutputs,1)-layer.weight
        if layer.bias is not None:
            bias=layer.bias[c].repeat(numOutputs)-layer.bias
            if globals.detach:
                bias=bias.detach()
        else:
            bias=None
        aK=Zcc[i,:].unsqueeze(0)
        rK=Rcc[i,:].unsqueeze(0).unsqueeze(1)
        #print('aj',aJ[i].unsqueeze(0).shape)
        #print('rK',rK.shape)
        RJ.append(LRPDenseReLU(layer=None,rK=rK,aJ=aJ[i].unsqueeze(0),aK=aK,e=e,
                               weights=weight, bias=bias).squeeze(1))
    RJ=torch.cat(RJ,dim=0).unsqueeze(1)
    if torch.isnan(RJ).any():
        print('nan 4')


    if amplify!=1:
        RJ=RJ*amplify

    
    return RJ
        
def LRPLogSumExpPool(rK, aJ, lse_r , e):
    #mix spatial dimensions and exponential:
    exp_aJ=torch.exp(lse_r*aJ)
    denom=exp_aJ.sum(dim=(-1,-2),keepdim=True)
    denom=denom+stabilizer(denom,e)
    R=torch.div(exp_aJ,denom)
    #add classes dimension
    R=R.unsqueeze(1).repeat(1,rK.shape[1],1,1,1)
    
    R=torch.mul(R,rK.repeat(1,1,1,R.shape[-2],R.shape[-1]))
    return R


def LRPConvReLUParallel(layer,rK,aJ,aK,e,rule='e',alpha=2,beta=-1,
               weights=None,bias=None,stride=None,padding=None,dilation=1):
    #returns: RJ, relevance at input of layer L
    #layer: layer L throgh which we propagate relevance, convolutional
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
    #aK: activations before ReLU
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    #weights, bias, stride and padding: layer parameters, overwritten if layer is not None
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    if (layer is not None):
        weights=layer.weight
        stride=layer.stride
        padding=layer.padding
        dilation=layer.dilation
        if(layer.bias is not None):
            bias=layer.bias.detach()
        else:
            bias=None
    
    if (rule=='e' or rule=='z+'):
        if (rule=='z+'):
            if(torch.min(aJ)<0):
                return LRPConvReLU(layer,rK,aJ,aK,e,rule='AB',alpha=1,beta=0,
                                weights=weights,bias=bias,stride=stride,
                                padding=padding,dilation=dilation)
                #raise ValueError('negative aJ elements in MaxPool input and z+ rule')
            weights=torch.max(weights,torch.zeros(weights.shape).type_as(weights))
            if(bias is not None):
                bias=torch.max(bias,torch.zeros(bias.shape).type_as(bias))
            
        if ((bias is not None and globals.detach) or rule=='z+'):
            aK=nn.functional.conv2d(aJ,weight=weights,bias=bias,
                                    stride=stride,padding=padding,
                                    dilation=dilation)
        aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

        z=aK+stabilizer(aK=aK,e=e)
        #element-wise inversion:s
        s=torch.div(rK,z)
        #shape: batch,o,k

        AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        

        s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])
        try:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                             stride=stride,padding=padding,
                                             dilation=dilation)
            c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

            RJ=torch.mul(AJ,c)
            
        except:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                             stride=stride,padding=padding,
                                             output_padding=(1,1),
                                             dilation=dilation)
            c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

            RJ=torch.mul(AJ,c)
        
    elif(rule=='z+e'):
        #numOutputs should be 2, considering one map per rule. set it to half:
        numOutputs=int(numOutputs/2)
        
        if torch.min(aJ)>=0:
            #print('original z+e rK:', torch.min(rK[:,1]))
            #if(torch.min(aJ)<0):
            #    raise ValueError('negative aJ elements in MaxPool input and z+ rule')

            weights_p=torch.max(weights,torch.zeros(weights.shape).type_as(weights))


            if (bias is not None):
                bias_p=torch.max(bias,torch.zeros(bias.shape).type_as(bias))
                bias=torch.cat((bias_p,bias),dim=0)

            #print('ak orig',aK[0,0,0])
            AK=nn.functional.conv2d(torch.cat((aJ,aJ),dim=1),
                                    torch.cat((weights_p,weights),dim=0),
                                    bias=bias,
                                    stride=stride,padding=padding,
                                    groups=2,dilation=dilation)

            size=int(AK.shape[1]/2)
            aKp=AK[:,:size,:,:].unsqueeze(1).repeat(1,numOutputs,1,1,1)
            if detach:
                aKe=AK[:,size:,:,:].unsqueeze(1).repeat(1,numOutputs,1,1,1)
            else:
                aKe=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            
            
            #print('ake orig',aKe[0,0,0,0])

            zp=aKp+stabilizer(aK=aKp,e=e)
            ze=aKe+stabilizer(aK=aKe,e=e)
            #element-wise inversion:s
            sp=torch.div(rK[:,:numOutputs,:,:,:],zp)
            se=torch.div(rK[:,numOutputs:,:,:,:],ze)
            #shape: batch,o,k



            sp=sp.view(batchSize*numOutputs,sp.shape[-3],sp.shape[-2],sp.shape[-1])
            se=se.view(batchSize*numOutputs,se.shape[-3],se.shape[-2],se.shape[-1])

            try:
                c=nn.functional.conv_transpose2d(torch.cat((sp,se),dim=1),
                                                     torch.cat((weights_p,weights),dim=0),
                                                     bias=None,stride=stride,padding=padding,
                                                     groups=2,dilation=dilation)
                size=int(c.shape[1]/2) 
                cp=c[:,:size,:,:]
                ce=c[:,size:,:,:]
                cp=cp.view(batchSize,numOutputs,cp.shape[-3],cp.shape[-2],cp.shape[-1])
                ce=ce.view(batchSize,numOutputs,ce.shape[-3],ce.shape[-2],ce.shape[-1])
                #add 2 rules to classes dimension
                c=torch.cat((cp,ce), dim=1)

                AJ=aJ.unsqueeze(1).repeat(1,numOutputs*2,1,1,1)
                RJ=torch.mul(AJ,c)

            except:
                c=nn.functional.conv_transpose2d(torch.cat((sp,se),dim=1),
                                                     torch.cat((weights_p,weights),dim=0),
                                                     bias=None,stride=stride,padding=padding,
                                                     groups=2,output_padding=(1,1),dilation=dilation)
                size=int(c.shape[1]/2) 
                cp=c[:,:size,:,:]
                ce=c[:,size:,:,:]
                cp=cp.view(batchSize,numOutputs,cp.shape[-3],cp.shape[-2],cp.shape[-1])
                ce=ce.view(batchSize,numOutputs,ce.shape[-3],ce.shape[-2],ce.shape[-1])
                #add 2 rules to classes dimension
                c=torch.cat((cp,ce), dim=1)

                AJ=aJ.unsqueeze(1).repeat(1,numOutputs*2,1,1,1)
                RJ=torch.mul(AJ,c)
            #print('original z+e RJ:', torch.min(RJ[:,1]))
            
        else:
            #print('new z+e rK:', torch.min(rK[:,1]))
            #use LRP-AB with A=1 and B=0 instead of LRP z+,
            #still conswevative, avoids negative relavance
            #positive and negative parts
            Wp,Wn=PosNeg(weights)
            if(bias is not None):
                Bp,Bn=PosNeg(bias)
                zeros=torch.zeros(Bp.shape).type_as(Bp)
                biases=torch.cat((Bp,zeros,bias),dim=0)
            else:
                biases=None
                
            aJp,aJn=PosNeg(aJ)
            #step 1- Forward pass
            #aKconc in the channels dimension: aKp_a,aKp_b,aK
            if (bias is not None and globals.detach):
                aKconc=nn.functional.conv2d(torch.cat((aJp,aJn,aJ),dim=1),
                                            torch.cat((Wp,Wn,weights),dim=0),
                                            bias=biases,
                                            stride=stride,padding=padding,
                                            groups=3,dilation=dilation)
                size=int(aKconc.shape[1]/3)
                aKp=aKconc[:,0:size,:,:]+aKconc[:,size:2*size,:,:]
                aKe=aKconc[:,2*size:,:,:]
            else:
                if biases is not None:
                    biases=biases[:-bias.shape[0]]
                aKconc=nn.functional.conv2d(torch.cat((aJp,aJn),dim=1),
                                            torch.cat((Wp,Wn),dim=0),
                                            bias=biases,
                                            stride=stride,padding=padding,
                                            groups=2,dilation=dilation)
                size=int(aKconc.shape[1]/2)
                aKp=aKconc[:,:size,:,:]+aKconc[:,size:,:,:]
                aKe=aK

            aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            aKe=aKe.unsqueeze(1).repeat(1,numOutputs,1,1,1)

            #step 2- Division and stabilizer
            zp=aKp+stabilizer(aK=aKp,e=e,sign=1)
            ze=aKe+stabilizer(aK=aKe,e=e)
            sp=torch.div(rK[:,:numOutputs,:,:,:],zp)
            se=torch.div(rK[:,numOutputs:,:,:,:],ze)

            #step 3- Transpose pass
            sp=sp.view(batchSize*numOutputs,sp.shape[-3],sp.shape[-2],sp.shape[-1])
            se=se.view(batchSize*numOutputs,se.shape[-3],se.shape[-2],se.shape[-1])

            aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            aJe=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            
            try:
                cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,se),dim=1),
                                                     torch.cat((Wp,Wn,weights),dim=0),
                                                     bias=None,stride=stride,padding=padding,
                                                     groups=3,dilation=dilation)
                size=int(cconc.shape[1]/3)
                cp_a=cconc[:,0:size,:,:]
                cp_b=cconc[:,size:2*size,:,:]
                ce=cconc[:,2*size:,:,:]

                cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
                cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
                ce=ce.view(batchSize,numOutputs,ce.shape[-3],ce.shape[-2],ce.shape[-1])

                #step 4- Multiplication with input

                RJp=torch.mul(aJp,cp_a)+torch.mul(aJn,cp_b)
                RJe=torch.mul(aJe,ce)

                RJ=torch.cat((RJp,RJe),dim=1)
                #print('new z+e RJ:', torch.min(RJ[:,1]))

            except:
                cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,se),dim=1),
                                                     torch.cat((Wp,Wn,weights),dim=0),
                                                     bias=None,stride=stride,padding=padding,
                                                     groups=3,output_padding=(1,1),dilation=dilation)
                
                size=int(cconc.shape[1]/3)
                cp_a=cconc[:,0:size,:,:]
                cp_b=cconc[:,size:2*size,:,:]
                ce=cconc[:,2*size:,:,:]

                cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
                cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
                ce=ce.view(batchSize,numOutputs,ce.shape[-3],ce.shape[-2],ce.shape[-1])

                #step 4- Multiplication with input

                RJp=torch.mul(aJp,cp_a)+torch.mul(aJn,cp_b)
                RJe=torch.mul(aJe,ce)

                RJ=torch.cat((RJp,RJe),dim=1)
                #print('new z+e RJ:', torch.min(RJ[:,1]))

    elif(rule=='AB'):
        if((alpha+beta)!=1):
            raise ValueError('Ensure alpha+beta=1')
        #positive and negative parts
        Wp,Wn=PosNeg(weights)
        if(bias is not None):
            Bp,Bn=PosNeg(bias)
            zeros=torch.zeros(Bp.shape).type_as(Bp)
            biases=torch.cat((Bp,zeros,Bn,zeros),dim=0)
        else:
            biases=None
        aJp,aJn=PosNeg(aJ)
        #step 1- Forward pass
        #skip step 1 if layer input and weights are all positive, and bias is None
        if(torch.min(aJ)>=0 and torch.min(weights)>=0 and biases is None):
            aKp,aKn=PosNeg(aK)
        else:
            #aKconc in the channels dimension: aKp_a,aKp_b,aKn_a,aKn_b
            aKconc=nn.functional.conv2d(torch.cat((aJp,aJn,aJn,aJp),dim=1),
                                        torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                        bias=biases,
                                        stride=stride,padding=padding,
                                        groups=4,dilation=dilation)
            size=int(aKconc.shape[1]/4)
            aKp=aKconc[:,0:size,:,:]+aKconc[:,size:2*size,:,:]
            aKn=aKconc[:,2*size:3*size,:,:]+aKconc[:,3*size:4*size,:,:]
            
        aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aKn=aKn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
        #step 2- Division and stabilizer
        zp=aKp+stabilizer(aK=aKp,e=e,sign=1)
        sp=torch.div(rK,zp)
        zn=aKn+stabilizer(aK=aKn,e=e,sign=-1)
        sn=torch.div(rK,zn)
        
        #step 3- Transpose pass
        sp=sp.view(batchSize*numOutputs,sp.shape[-3],sp.shape[-2],sp.shape[-1])
        sn=sn.view(batchSize*numOutputs,sn.shape[-3],sn.shape[-2],sn.shape[-1])
        
        aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
        try:
            cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                 torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                 bias=None,stride=stride,padding=padding,
                                                 groups=4,dilation=dilation)
            size=int(cconc.shape[1]/4)
            cp_a=cconc[:,0:size,:,:]
            cp_b=cconc[:,size:2*size,:,:]
            cn_a=cconc[:,2*size:3*size,:,:]
            cn_b=cconc[:,3*size:4*size,:,:]

            cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

            #step 4- Multiplication with input
            RJ=alpha*(torch.mul(aJp,cp_a)+\
                      torch.mul(aJn,cp_b))
            RJ=RJ+beta*(torch.mul(aJn,cn_a)+\
                        torch.mul(aJp,cn_b))
            
        except:
            cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                 torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                 bias=None,stride=stride,padding=padding,
                                                 groups=4,output_padding=(1,1),dilation=dilation)
            
            size=int(cconc.shape[1]/4)
            cp_a=cconc[:,0:size,:,:]
            cp_b=cconc[:,size:2*size,:,:]
            cn_a=cconc[:,2*size:3*size,:,:]
            cn_b=cconc[:,3*size:4*size,:,:]

            cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

            #step 4- Multiplication with input
            RJ=alpha*(torch.mul(aJp,cp_a)+\
                      torch.mul(aJn,cp_b))
            RJ=RJ+beta*(torch.mul(aJn,cn_a)+\
                        torch.mul(aJp,cn_b))
        if(len(RJ.shape)<5):
            RJ=RJ.unsqueeze(1)
            
            
    else:
        raise ValueError('only Epsilon (e), Alpha Beta (AB), and z+ rules implemented')
        
    return RJ

def LRPConvReLU(layer,rK,aJ,aK,e,rule='e',alpha=2,beta=-1,
               weights=None,bias=None,stride=None,padding=None,dilation=1):
    #returns: RJ, relevance at input of layer L
    #layer: layer L throgh which we propagate relevance, convolutional
    #e: LRP-e term. Use e=0 for LRP0
    #RK: relevance at layer L output
    #aJ: values at layer L input
    #aK: activations before ReLU
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    #weights, bias, stride and padding: layer parameters, overwritten if layer is not None
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    if (layer is not None):
        weights=layer.weight
        stride=layer.stride
        padding=layer.padding
        dilation=layer.dilation
        if(layer.bias is not None):
            bias=layer.bias.detach()
        else:
            bias=None
    
    if (rule=='e' or rule=='z+'):
        if (rule=='z+'):
            if(torch.min(aJ)<0):
                raise ValueError('negative aJ elements in input and z+ rule, use AB rule with A=1, B=0')
            weights=torch.max(weights,torch.zeros(weights.shape).type_as(weights))
            if(bias is not None):
                bias=torch.max(bias,torch.zeros(bias.shape).type_as(bias))
            
        #if rule=='e':
        #    #print('aK0',aK[0,0,0])
        if ((bias is not None and globals.detach) or rule=='z+'):
            aK=nn.functional.conv2d(aJ,weight=weights,bias=bias,
                                    stride=stride,padding=padding,
                                    dilation=dilation)
            #if rule=='e':
            #    #print('aK1',aK[0,0,0])
        aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

        z=aK+stabilizer(aK=aK,e=e)
        #element-wise inversion:s
        s=torch.div(rK,z)
        #shape: batch,o,k

        AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)        
        
        s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

        if isinstance(stride, int):
            flag=(stride==1)
        else:
            flag=(stride[0]==1)
            
        if flag:
            c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                             stride=stride,padding=padding,
                                             dilation=dilation,output_padding=0)
            c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

            RJ=torch.mul(AJ,c)
            
        else:
            try:
                c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                                 stride=stride,padding=padding,
                                                 dilation=dilation,output_padding=(1,1))
                c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

                RJ=torch.mul(AJ,c)

            except:
                c=nn.functional.conv_transpose2d(s,weight=weights,bias=None,
                                                 stride=stride,padding=padding,
                                                 output_padding=0,
                                                 dilation=dilation)
                c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

                RJ=torch.mul(AJ,c)
    elif(rule=='AB'):
        if((alpha+beta)!=1):
            raise ValueError('Ensure alpha+beta=1')
        #positive and negative parts
        Wp,Wn=PosNeg(weights)
        if(bias is not None):
            Bp,Bn=PosNeg(bias)
            zeros=torch.zeros(Bp.shape).type_as(Bp)
            biases=torch.cat((Bp,zeros,Bn,zeros),dim=0)
        else:
            biases=None
        aJp,aJn=PosNeg(aJ)
        #step 1- Forward pass
        #skip step 1 if layer input and weights are all positive, and bias is None
        if(torch.min(aJ)>=0 and torch.min(weights)>=0 and biases is None):
            aKp,aKn=PosNeg(aK)
        else:
            #aKconc in the channels dimension: aKp_a,aKp_b,aKn_a,aKn_b
            aKconc=nn.functional.conv2d(torch.cat((aJp,aJn,aJn,aJp),dim=1),
                                        torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                        bias=biases,
                                        stride=stride,padding=padding,
                                        groups=4,dilation=dilation)
            size=int(aKconc.shape[1]/4)
            aKp=aKconc[:,0:size,:,:]+aKconc[:,size:2*size,:,:]
            aKn=aKconc[:,2*size:3*size,:,:]+aKconc[:,3*size:4*size,:,:]
            
        aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aKn=aKn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
        #step 2- Division and stabilizer
        zp=aKp+stabilizer(aK=aKp,e=e,sign=1)
        sp=torch.div(rK,zp)
        zn=aKn+stabilizer(aK=aKn,e=e,sign=-1)
        sn=torch.div(rK,zn)
        
        #step 3- Transpose pass
        sp=sp.view(batchSize*numOutputs,sp.shape[-3],sp.shape[-2],sp.shape[-1])
        sn=sn.view(batchSize*numOutputs,sn.shape[-3],sn.shape[-2],sn.shape[-1])
        
        aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
        if isinstance(stride, int):
            flag=(stride==1)
        else:
            flag=(stride[0]==1)
        if flag:
            cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                 torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                 bias=None,stride=stride,padding=padding,
                                                 groups=4,dilation=dilation)
            size=int(cconc.shape[1]/4)
            cp_a=cconc[:,0:size,:,:]
            cp_b=cconc[:,size:2*size,:,:]
            cn_a=cconc[:,2*size:3*size,:,:]
            cn_b=cconc[:,3*size:4*size,:,:]

            cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

            #step 4- Multiplication with input
            RJ=alpha*(torch.mul(aJp,cp_a)+\
                      torch.mul(aJn,cp_b))
            RJ=RJ+beta*(torch.mul(aJn,cn_a)+\
                        torch.mul(aJp,cn_b))
            
        else:        
            try:
                cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                     torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                     bias=None,stride=stride,padding=padding,
                                                     groups=4,dilation=dilation,output_padding=(1,1))
                size=int(cconc.shape[1]/4)
                cp_a=cconc[:,0:size,:,:]
                cp_b=cconc[:,size:2*size,:,:]
                cn_a=cconc[:,2*size:3*size,:,:]
                cn_b=cconc[:,3*size:4*size,:,:]

                cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
                cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
                cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
                cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

                #step 4- Multiplication with input
                RJ=alpha*(torch.mul(aJp,cp_a)+\
                          torch.mul(aJn,cp_b))
                RJ=RJ+beta*(torch.mul(aJn,cn_a)+\
                            torch.mul(aJp,cn_b))

            except:
                cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                     torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                     bias=None,stride=stride,padding=padding,
                                                     groups=4,output_padding=0,dilation=dilation)

                size=int(cconc.shape[1]/4)
                cp_a=cconc[:,0:size,:,:]
                cp_b=cconc[:,size:2*size,:,:]
                cn_a=cconc[:,2*size:3*size,:,:]
                cn_b=cconc[:,3*size:4*size,:,:]

                cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
                cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
                cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
                cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

                #step 4- Multiplication with input
                RJ=alpha*(torch.mul(aJp,cp_a)+\
                          torch.mul(aJn,cp_b))
                RJ=RJ+beta*(torch.mul(aJn,cn_a)+\
                            torch.mul(aJp,cn_b))
                
        if(len(RJ.shape)<5):
            RJ=RJ.unsqueeze(1)
            
            
    elif(rule=='z+e'):
        numOutputs=int(numOutputs/2)
        if torch.min(aJ)>=0:
            RJp=LRPConvReLU(layer,rK[:,:numOutputs,:,:,:],aJ,aK,e,rule='z+',
                                weights=weights,bias=bias,stride=stride,
                                padding=padding,dilation=dilation)
        else:
            RJp=LRPConvReLU(layer,rK[:,:numOutputs,:,:,:],aJ,aK,e,rule='AB',alpha=1,beta=0,
                                weights=weights,bias=bias,stride=stride,
                                padding=padding,dilation=dilation)
        RJe=LRPConvReLU(layer,rK[:,numOutputs:,:,:,:],aJ,aK,e,rule='e',
                        weights=weights,bias=bias,stride=stride,
                        padding=padding,dilation=dilation)
        RJ=torch.cat((RJp,RJe),dim=1)
            
            
    else:
        raise ValueError('only Epsilon (e), Alpha Beta (AB), and z+ rules implemented')
        
    return RJ

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

def LRPPool2d(layer,rK,aJ,aK,e,adaptative=False,rule='e',
              alpha=2,beta=-1):
    #relevance propagation through average pooling
    #adaptative: For adaptative average pooling
    #layer: pooling layer
    #e:e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer output
    #aJ: pooling layer inputs
    #aK: activations after pooling
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
        
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=rK.shape[2]
    
    if(adaptative):
        #https://stackoverflow.com/questions/58692476/what-is-adaptive-average-pooling-and-how-does-it-work
        #kernel_size=(int(aJ.shape[-2]/layer.output_size[0]),int(aJ.shape[-1]/layer.output_size[1]))
        #stride=kernel_size
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
    
    #tmp=torch.nn.functional.conv2d(aJ,weights,stride=stride)
    #print(aK.shape,tmp.shape)
    #print(aK[0,0])
    #print(tmp[0,0])
    
    
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
                   weights=weights,bias=None,stride=stride,padding=padding)

    return RJ

def LRPSum(rK,aJ,aK,e,rule='e',alpha=2,beta=-1):
    #relevance propagation through torch.sum in the spacial dimensions
    #e:e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer output
    #aJ: pooling layer inputs
    #aK: activations after pooling
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
        
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
    
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
               weights=weights,bias=None,stride=stride,padding=padding)

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
        layerBias=layerBias.clone().detach()
    
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
    
def FuseIN(layerWeights, IN, aKConv,Ch0=0,Ch1=None,layerBias=None,
           bias=True,INbeforeReLU=True):
    #returns parameters of convolution fused with instance normalization
    #IN:batch normalization layer
    #layerWeights: convolutional layer wieghts
    #layerBias: convolutional layer bias
    #aKConv: IN input
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the convolutional layer
    #bias: allows returning of bias of equivalent convolution
    #INbeforeReLU: true if IN is placed before the layer activation
    
    if(layerBias is not None):
        layerBias=layerBias.clone().detach()
    
    if(Ch1==IN.running_var.shape[0]):
        Ch1=None
        
    #norm over spatial dimensions, leave batch and channel
    if (IN.training or (not IN.track_running_stats)):
        mean=torch.mean(aKConv,dim=(-2,-1)).detach()
        var=torch.var(aKConv,dim=(-2,-1)).detach()
        std=torch.sqrt(var+IN.eps)
    else:
        mean=IN.running_mean[Ch0:Ch1].detach().unsqueeze(0)
        var=IN.running_var[Ch0:Ch1].detach().unsqueeze(0)
        std=torch.sqrt(var+IN.eps)    
        
    #multiplicative factor for each channel, caused by IN:
    if(IN.weight is None):
        std=std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights=layerWeights/std
    else:
        #y/rho:
        INweights=torch.div(IN.weight[Ch0:Ch1],std)
        #add 3 dimensions (in channels, width and height) in IN weights 
        #to match convolutional weights shape:
        if(INbeforeReLU):
            INweights=INweights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shp=layerWeights.shape
            INweights=INweights.repeat(1,shp[-3],shp[-2],shp[-1])
        else:#align with output channels dimension
            INweights=INweights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            shp=layerWeights.shape
            INweights=INweights.repeat(shp[0],1,shp[2],shp[3])
        #w.y/rho:
        weights=torch.mul(INweights,layerWeights)
    
    if (not bias):
        return weights
    
    if(bias):
        if(layerBias is None):
            layerBias=torch.zeros((weights.shape[0])).type_as(layerWeights)
        if(IN.weight is None and IN.bias is None):
            biases=layerBias*0
        else:
            if(INbeforeReLU):
                biases=layerBias-mean
                biases=torch.div(biases,std)
                biases=torch.mul(IN.weight[Ch0:Ch1],biases)
                biases=biases+IN.bias[Ch0:Ch1].detach()
            else:
                biases=torch.mul(IN.weight[Ch0:Ch1],mean)
                biases=torch.div(biases,std)
                biases=IN.bias[Ch0:Ch1].detach()-biases
                biases=biases.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                shp=layerWeights.shape
                biases=biases.repeat(shp[0],1,shp[2],shp[3])
                biases=torch.mul(biases,layerWeights)
                biases=biases.sum([1,2,3])
                biases=biases+layerBias
                

        return (weights, biases)

def LRPPool2dBNReLU(layer,BN,rK,aJ,aK,e,aKPool,Ch0=0,Ch1=None,
                    rule='e',alpha=2,beta=-1):
    #propagates relevance through average pooling followed by batch normalization and ReLU
    #layer: pooling layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L pooling input
    #aK: activations before ReLU
    #aKPool: pooling output
    #BN: batch normalization layer
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the pooling layer
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
        
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
    
    
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
               weights=weights,bias=biases,stride=stride,padding=layer.padding)

    return RJ

def LRPConvBNReLU(layer,BN,rK,aJ,aK,e,aKConv,Ch0=0,Ch1=None,rule='e',alpha=2,beta=-1):
    #used to propagate relevance through the sequence: Convolution, Batchnorm, ReLU
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN
    #aKConv: convolution output
    #BN: batch normalization layer
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the concolutional layer
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    
    aK=aK[:,Ch0:Ch1,:,:]
    rK=rK[:,:,Ch0:Ch1,:,:]
    
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    
    weights,biases=FuseBN(layerWeights=layer.weight, BN=BN, aKConv=aKConv,
                   Ch0=Ch0,Ch1=Ch1,layerBias=layer.bias,
                   bias=True)
    
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
                   weights=weights,bias=biases,stride=layer.stride,
                   padding=layer.padding,dilation=layer.dilation)
          
    return RJ

def LRPElementWiseSUM(rK,a,b,e,rule):
    #LRP rule for element-wise sum between tensors
    #rK: relevance at sum output
    #e: LRP-e term. Use e=0 for LRP0
    #rule: e, z+ or z+e
    #a and b: first and second sum inputs
    #return: rJx,rJy, the relevances for a and b, respectively
    
    if (rule=='e' or rule=='z+'):
        #add classes dimension
        x=a.unsqueeze(1).repeat(1,rK.shape[1],1,1,1).clone()
        y=b.unsqueeze(1).repeat(1,rK.shape[1],1,1,1).clone()
        
        if rule=='z+':
            x=torch.max(x,torch.zeros(x.shape).type_as(x))
            y=torch.max(y,torch.zeros(y.shape).type_as(y))

        rJx=torch.div(x,(x+y+stabilizer(x+y,e)))
        rJy=torch.div(y,(x+y+stabilizer(x+y,e)))
        rJx=torch.mul(rJx,rK)
        rJy=torch.mul(rJy,rK)
        
    elif rule=='z+e':
        NumOutputs=int(rK.shape[1]/2)
        rKp=rK[:,:NumOutputs,:,:,:]
        rKe=rK[:,NumOutputs:,:,:,:]
        rJxp,rJyp=LRPElementWiseSUM(rKp,a,b,e,rule='z+')
        rJxe,rJye=LRPElementWiseSUM(rKe,a,b,e,rule='e')
        rJx=torch.cat((rJxp,rJxe),dim=1)
        rJy=torch.cat((rJyp,rJye),dim=1)
        
    else:
        raise ValueError('rule not implemented for element-wise sum')
        
        
    return rJx,rJy


    

def MultiBlockPoolBNReLU(layer,BN,rK,aJ,e,aKPool,aK,Ch0=0,Ch1=None,
                         rule='e',alpha=2,beta=-1):
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
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    rK,aK=rK[:],aK[:]
    for i,r in enumerate(rK,0):
        if(Ch1==rK[i].shape[2]):
            TCh1=None
        else: 
            TCh1=Ch1
        rK[i]=rK[i][:,:,Ch0:TCh1,:,:]
    rK=torch.cat(rK,dim=2)
    
    for i,a in enumerate(aK,0):
        if(Ch1==aK[i].shape[1]):
            TCh1=None
        else: 
            TCh1=Ch1
        aK[i]=aK[i][:,Ch0:TCh1,:,:]
    aK=torch.cat(aK,dim=1)
    
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
        
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
               weights=weights,bias=biases,stride=stride,padding=layer.padding)

    return RJ


def MultiLayerConvBNReLU(layer,BN,rK,aK,aJ,e,aKConv,Ch0=0,Ch1=None,
                         rule='e',alpha=2,beta=-1):
    #propagates relevance through last convolution in the dense layers in dense blocks
    #layer: convolutional layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L convolution input
    #aK: activations after BN, list (one for each BN layer)
    #aKConv: convolution output
    #BN: batch normalization layer, list of first BN operations in layers connecting to the convolution in "layer"
    #Ch0 and Ch1: delimits the batch normalization channels that originate from the concolutional layer
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    rK,aK=rK[:],aK[:]
    for i,r in enumerate(rK,0):
        if(Ch1==rK[i].shape[2]):
            TCh1=None
        else: 
            TCh1=Ch1
        rK[i]=rK[i][:,:,Ch0:TCh1,:,:]
    rK=torch.cat(rK,dim=2)
    
    #concatenate ak in channels dimension:
    for i,a in enumerate(aK,0):
        if(Ch1==aK[i].shape[1]):
            TCh1=None
        else: 
            TCh1=Ch1
        aK[i]=aK[i][:,Ch0:TCh1,:,:]
    aK=torch.cat(aK,dim=1)
    
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
        
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
               weights=weights,bias=biases,stride=layer.stride,padding=layer.padding,
                  dilation=layer.dilation)    

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
    
    if(layer.bias is None and globals.detach):
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
    
    if(layer.bias is None and globals.detach):
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
    
    if (BN.bias is not None and globals.detach):
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
        cPos=l*nn.functional.conv_transpose2d(s,weight=WPos,bias=None,stride=layer.stride,
                                              padding=layer.padding,output_padding=op)
        cNeg=h*nn.functional.conv_transpose2d(s,weight=WNeg,bias=None,stride=layer.stride,
                                              padding=layer.padding,output_padding=op)

        c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        cPos=cPos.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        cNeg=cNeg.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1) 
        R0=torch.mul(AJ,c)-cPos-cNeg
        
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

    if(torch.min(R0<0)):
        print('negatives in R0')
    
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
    BPos=torch.max(layer.bias.detach(),torch.zeros(biases.shape).type_as(rK))
    BNeg=torch.min(layer.bias.detach(),torch.zeros(biases.shape).type_as(rK))
        
    #propagation:
    aKPos=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK)*l,weight=WPos,
                                bias=BPos*l,stride=layer.stride,padding=layer.padding)
    aKNeg=nn.functional.conv2d(torch.ones(aJ.shape).type_as(rK)*h,weight=WNeg,
                                bias=BNeg*h,stride=layer.stride,padding=layer.padding)
    
    if (layer.bias is not None and globals.detach):
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
        cPos=l*nn.functional.conv_transpose2d(s,weight=WPos,bias=None,stride=layer.stride,
                                              padding=layer.padding,output_padding=op)
        cNeg=h*nn.functional.conv_transpose2d(s,weight=WNeg,bias=None,stride=layer.stride,
                                              padding=layer.padding,output_padding=op)

        c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        cPos=cPos.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        cNeg=cNeg.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])
        AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1) 
        R0=torch.mul(AJ,c)-cPos-cNeg
        
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

def LRPMaxPool2d(layer,rK,aJ,aK,e,rule='e',alpha=2,beta=-1):
    #propagates relevance through max pooling (preceded by ReLU) or max pool+ReLU 
    #layer: pooling layer throgh which we propagate relevance
    #e: LRP-e term. Use e=0 for LRP0
    #rK: relevance at layer L ReLU output
    #aJ: values at layer L pooling input
    #aK: activations before ReLU
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    
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
        
    if(rule=='e' or rule=='z+' or rule=='z+e'):
        if (rule=='z+' or rule=='z+e'):
            if(torch.min(aJ)<0):
                raise ValueError('negative aJ elements in MaxPool input and z+ rule')
            
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
        #unpool
        try:
            c=nn.functional.max_unpool2d(s,indices=indexes,kernel_size=layer.kernel_size,
                                         stride=layer.stride,padding=layer.padding,
                                         output_size=(s.shape[0],aJ.shape[1],
                                                      aJ.shape[2],aJ.shape[3]))
        except:
            with torch.cuda.amp.autocast(enabled=False):
                c=nn.functional.max_unpool2d(s.float(),indices=indexes,kernel_size=layer.kernel_size,
                                         stride=layer.stride,padding=layer.padding,
                                         output_size=(s.shape[0],aJ.shape[1],
                                                      aJ.shape[2],aJ.shape[3]))
        #reshape:   
        c=c.view(numOutputs,batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
        #change to batches, classes again:
        c=c.permute(1,0,2,3,4)                   

        AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)  

        RJ=torch.mul(AJ,c)
        
    elif(rule=='AB'):
        if((alpha+beta)!=1):
            raise ValueError('Ensure alpha+beta=1')
        #positive and negative parts
        aJp,aJn=PosNeg(aJ)
        #step 1:
        aKp,aKn=PosNeg(aK)
        aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aKn=aKn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        #step 2:
        zp=aKp+stabilizer(aK=aKp,e=e)
        zn=aKp+stabilizer(aK=aKn,e=e)
        
        sp=torch.div(rK,zp)
        sn=torch.div(rK,zn)
        
        #step 3:
        #Unpool: (propagate relevance through Maxpool)
        #begins with output classes dimension, then batch dimension
        sp=sp.permute(1,0,2,3,4)
        sn=sn.permute(1,0,2,3,4)
        #change to classes,batches:
        sp=sp.reshape(numOutputs*batchSize,sp.shape[-3],sp.shape[-2],sp.shape[-1])
        sn=sn.reshape(numOutputs*batchSize,sn.shape[-3],sn.shape[-2],sn.shape[-1])
        indexes=indexes.repeat(numOutputs,1,1,1)
        #unpool:
        try:
            cp=nn.functional.max_unpool2d(sp,indices=indexes,
                                          kernel_size=layer.kernel_size,
                                         stride=layer.stride,padding=layer.padding,
                                         output_size=(sp.shape[0],aJ.shape[1],
                                                      aJ.shape[2],aJ.shape[3]))
            cn=nn.functional.max_unpool2d(sn,indices=indexes,
                                          kernel_size=layer.kernel_size,
                                         stride=layer.stride,padding=layer.padding,
                                         output_size=(sn.shape[0],aJ.shape[1],
                                                      aJ.shape[2],aJ.shape[3]))
        except:
            with torch.cuda.amp.autocast(enabled=False):
                cp=nn.functional.max_unpool2d(sp.float(),indices=indexes,
                                              kernel_size=layer.kernel_size,
                                         stride=layer.stride,padding=layer.padding,
                                         output_size=(sp.shape[0],aJ.shape[1],
                                                      aJ.shape[2],aJ.shape[3]))
                cn=nn.functional.max_unpool2d(sn.float(),indices=indexes,
                                              kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(sn.shape[0],aJ.shape[1],
                                                          aJ.shape[2],aJ.shape[3]))
        #reshape:   
        cp=cp.view(numOutputs,batchSize,cp.shape[-3],cp.shape[-2],cp.shape[-1])
        cn=cn.view(numOutputs,batchSize,cn.shape[-3],cn.shape[-2],cn.shape[-1])
        #change to batches, classes again:
        cp=cp.permute(1,0,2,3,4)
        cn=cn.permute(1,0,2,3,4)
        
        #step 4:
        aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        RJ=alpha*(torch.mul(aJp.squeeze(1),cp.squeeze(1)))
        RJ=RJ+beta*(torch.mul(aJn.squeeze(1),cn.squeeze(1)))
        if(len(RJ.shape)<5):
            RJ=RJ.unsqueeze(1)
            
        
    else:
        raise ValueError('Rule must be e, z+, z+e or AB')
    #if(torch.min(RJ)<0):
    #    print('negatives')
    return RJ

def MultiBlockMaxPoolBNReLU(layer,BN,rK,aJ,aK,e,aKPool,Ch0=0,Ch1=None,
                            rule='e',alpha=2,beta=-1):
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
    #rule: 'e': epsilon; 'AB': alpha beta
    #alpha and beta: weights for LRP-AlphaBeta rule
    rK,aK=rK[:],aK[:]
    
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
    
    #concatenate ak in channels dimension:
    for i,a in enumerate(aK,0):
        if(Ch1==aK[i].shape[1]):
            TCh1=None
        else: 
            TCh1=Ch1
        aK[i]=aK[i][:,Ch0:TCh1,:,:]
    aK=torch.cat(aK,dim=1)
        
        
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

    if (rule=='e' or rule=='z+'):
        if (rule=='z+'):
            if(torch.min(aJ)<0):
                raise ValueError('negative aJ elements in MaxPool input and z+ rule')
            weights=torch.max(weights,torch.zeros(weights.shape).type_as(weights))
            biases=torch.max(biases,torch.zeros(biases.shape).type_as(biases))
            
        if ((BN[0].bias is not None) or rule=='z+'):
            aK=nn.functional.conv2d(aKPool,weights,biases)

        aK=aK.unsqueeze(1).repeat(1,numOutputs,1,1,1)

        z=aK+stabilizer(aK=aK,e=e)
        #element-wise inversion:s
        s=torch.div(rK,z)
        #shape: batch,o,k

        s=s.view(batchSize*numOutputs,s.shape[-3],s.shape[-2],s.shape[-1])

        #transpose conv: (propagate relevance through BN)
        try:
            c=nn.functional.conv_transpose2d(s,weight=weights,stride=stride)
            c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

            #Unpool: (propagate relevance through Maxpool)
            #begins with output classes dimension, then batch dimension
            c=c.permute(1,0,2,3,4)
            #change to classes,batches:
            c=c.reshape(numOutputs*batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
            indexes=indexes.repeat(numOutputs,1,1,1)
            #unpool:
            try:
                c=nn.functional.max_unpool2d(c,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(c.shape[0],aJ.shape[1],
                                                          aJ.shape[2],aJ.shape[3]))
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    c=nn.functional.max_unpool2d(c.float(),indices=indexes,
                                                 kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(c.shape[0],aJ.shape[1],
                                                          aJ.shape[2],aJ.shape[3]))
            
            #reshape:   
            c=c.view(numOutputs,batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
            #change to batches, classes again:
            c=c.permute(1,0,2,3,4)

            AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)  

            RJ=torch.mul(AJ,c)
            
        except:
            c=nn.functional.conv_transpose2d(s,weight=weights,stride=stride,
                                             output_padding=(1,1))
            c=c.view(batchSize,numOutputs,c.shape[-3],c.shape[-2],c.shape[-1])

            #Unpool: (propagate relevance through Maxpool)
            #begins with output classes dimension, then batch dimension
            c=c.permute(1,0,2,3,4)
            #change to classes,batches:
            c=c.reshape(numOutputs*batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
            indexes=indexes.repeat(numOutputs,1,1,1)
            #unpool:
            try:
                c=nn.functional.max_unpool2d(c,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(c.shape[0],aJ.shape[1],
                                                          aJ.shape[2],aJ.shape[3]))
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    c=nn.functional.max_unpool2d(c.float(),indices=indexes,
                                                 kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(c.shape[0],aJ.shape[1],
                                                          aJ.shape[2],aJ.shape[3]))
            #reshape:   
            c=c.view(numOutputs,batchSize,c.shape[-3],c.shape[-2],c.shape[-1])
            #change to batches, classes again:
            c=c.permute(1,0,2,3,4)

            AJ=aJ.unsqueeze(1).repeat(1,numOutputs,1,1,1)  

            RJ=torch.mul(AJ,c)
        
    elif(rule=='z+e'):
        numOutputs=int(numOutputs/2)
        if(torch.min(aJ)<0):
            raise ValueError('negative aJ elements in MaxPool input and z+ rule')
        weights_p=torch.max(weights,torch.zeros(weights.shape).type_as(weights))
        biases_p=torch.max(biases,torch.zeros(biases.shape).type_as(biases))
            
        AK=nn.functional.conv2d(torch.cat((aKPool,aKPool),dim=1),
                                torch.cat((weights_p,weights),dim=0),
                                torch.cat((biases_p,biases),dim=0),
                                stride=stride,groups=2)
        
        size=int(AK.shape[1]/2)
        aKp=AK[:,:size,:,:].unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aKe=AK[:,size:,:,:].unsqueeze(1).repeat(1,numOutputs,1,1,1)


        zp=aKp+stabilizer(aK=aKp,e=e)
        ze=aKe+stabilizer(aK=aKe,e=e)
        #element-wise inversion:s
        sp=torch.div(rK[:,:numOutputs,:,:,:],zp)
        se=torch.div(rK[:,numOutputs:,:,:,:],ze)
        #shape: batch,o,k

        #transpose conv BN:
        sp=sp.view(batchSize*numOutputs,sp.shape[-3],sp.shape[-2],sp.shape[-1])
        se=se.view(batchSize*numOutputs,se.shape[-3],se.shape[-2],se.shape[-1])
        try:
            c=nn.functional.conv_transpose2d(torch.cat((sp,se),dim=1),
                                                 torch.cat((weights_p,weights),dim=0),
                                                 bias=None,stride=stride,
                                                 groups=2)
            size=int(c.shape[1]/2) 
            cp=c[:,:size,:,:]
            ce=c[:,size:,:,:]
            cp=cp.view(batchSize,numOutputs,cp.shape[-3],cp.shape[-2],cp.shape[-1])
            ce=ce.view(batchSize,numOutputs,ce.shape[-3],ce.shape[-2],ce.shape[-1])

            #Unpool: (propagate relevance through Maxpool)
            #begins with output classes dimension, then batch dimension
            cp=cp.permute(1,0,2,3,4)
            ce=ce.permute(1,0,2,3,4)
            #change to classes,batches:
            cp=cp.reshape(numOutputs*batchSize,cp.shape[-3],cp.shape[-2],cp.shape[-1])
            ce=ce.reshape(numOutputs*batchSize,ce.shape[-3],ce.shape[-2],ce.shape[-1])

            indexes=indexes.repeat(numOutputs,1,1,1)
            #unpool:
            try:
                cp=nn.functional.max_unpool2d(cp,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                ce=nn.functional.max_unpool2d(ce,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(ce.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    cp=nn.functional.max_unpool2d(cp.float(),indices=indexes,
                                                  kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    ce=nn.functional.max_unpool2d(ce.float(),indices=indexes,
                                                  kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(ce.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            #reshape:   
            cp=cp.view(numOutputs,batchSize,cp.shape[-3],cp.shape[-2],cp.shape[-1])
            ce=ce.view(numOutputs,batchSize,ce.shape[-3],ce.shape[-2],ce.shape[-1])
            #change to batches, classes again:
            cp=cp.permute(1,0,2,3,4)
            ce=ce.permute(1,0,2,3,4)

            #combine two maps in the classes dimension:
            c=torch.cat((cp,ce), dim=1)

            AJ=aJ.unsqueeze(1).repeat(1,2*numOutputs,1,1,1)  

            RJ=torch.mul(AJ,c)
            
        except:
            c=nn.functional.conv_transpose2d(torch.cat((sp,se),dim=1),
                                                 torch.cat((weights_p,weights),dim=0),
                                                 bias=None,stride=stride,
                                                 groups=2,output_padding=(1,1))
            size=int(c.shape[1]/2) 
            cp=c[:,:size,:,:]
            ce=c[:,size:,:,:]
            cp=cp.view(batchSize,numOutputs,cp.shape[-3],cp.shape[-2],cp.shape[-1])
            ce=ce.view(batchSize,numOutputs,ce.shape[-3],ce.shape[-2],ce.shape[-1])

            #Unpool: (propagate relevance through Maxpool)
            #begins with output classes dimension, then batch dimension
            cp=cp.permute(1,0,2,3,4)
            ce=ce.permute(1,0,2,3,4)
            #change to classes,batches:
            cp=cp.reshape(numOutputs*batchSize,cp.shape[-3],cp.shape[-2],cp.shape[-1])
            ce=ce.reshape(numOutputs*batchSize,ce.shape[-3],ce.shape[-2],ce.shape[-1])

            indexes=indexes.repeat(numOutputs,1,1,1)
            #unpool:
            try:
                cp=nn.functional.max_unpool2d(cp,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                ce=nn.functional.max_unpool2d(ce,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(ce.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    cp=nn.functional.max_unpool2d(cp.float(),indices=indexes,
                                                  kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    ce=nn.functional.max_unpool2d(ce.float(),indices=indexes,
                                                  kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(ce.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            #reshape:   
            cp=cp.view(numOutputs,batchSize,cp.shape[-3],cp.shape[-2],cp.shape[-1])
            ce=ce.view(numOutputs,batchSize,ce.shape[-3],ce.shape[-2],ce.shape[-1])
            #change to batches, classes again:
            cp=cp.permute(1,0,2,3,4)
            ce=ce.permute(1,0,2,3,4)

            #combine two maps in the classes dimension:
            c=torch.cat((cp,ce), dim=1)

            AJ=aJ.unsqueeze(1).repeat(1,2*numOutputs,1,1,1)  

            RJ=torch.mul(AJ,c)
        
    elif(rule=='AB'):
        if((alpha+beta)!=1):
            raise ValueError('Ensure alpha+beta=1')
        #positive and negative parts
        Wp,Wn=PosNeg(weights)
        Bp,Bn=PosNeg(biases)
        zeros=torch.zeros(Bp.shape).type_as(Bp)
        biases=torch.cat((Bp,zeros,Bn,zeros),dim=0)
        aKPoolp,aKPooln=PosNeg(aKPool)
        aJp,aJn=PosNeg(aJ)
        
        #step 1- Forward pass
        #aKconc in the channels dimension: aKp_a,aKp_b,aKn_a,aKn_b
        aKconc=nn.functional.conv2d(torch.cat((aKPoolp,aKPooln,aKPooln,aKPoolp),dim=1),
                                    torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                    bias=biases,
                                    groups=4)
        size=int(aKconc.shape[1]/4)
        aKp=aKconc[:,0:size,:,:]+aKconc[:,size:2*size,:,:]
        aKn=aKconc[:,2*size:3*size,:,:]+aKconc[:,3*size:4*size,:,:]
            
        aKp=aKp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        aKn=aKn.unsqueeze(1).repeat(1,numOutputs,1,1,1)
        
        #step 2- Division and stabilizer
        zp=aKp+stabilizer(aK=aKp,e=e)
        sp=torch.div(rK,zp)
        zn=aKn+stabilizer(aK=aKn,e=e)
        sn=torch.div(rK,zn)
        
        #step 3- Transpose pass
        sp=sp.view(batchSize*numOutputs,sp.shape[-3],sp.shape[-2],sp.shape[-1])
        sn=sn.view(batchSize*numOutputs,sn.shape[-3],sn.shape[-2],sn.shape[-1])
        
        try:
            cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                 torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                 bias=None,stride=stride,
                                                 groups=4)
            size=int(cconc.shape[1]/4)
            cp_a=cconc[:,0:size,:,:]
            cp_b=cconc[:,size:2*size,:,:]
            cn_a=cconc[:,2*size:3*size,:,:]
            cn_b=cconc[:,3*size:4*size,:,:]

            cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

            #Unpool: (propagate relevance through Maxpool)
            cp_a=cp_a.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])
            indexes=indexes.repeat(numOutputs,1,1,1)
            #unpool:
            try:
                cp_a=nn.functional.max_unpool2d(cp_a,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp_a.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                cp_b=nn.functional.max_unpool2d(cp_b,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp_b.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                cn_a=nn.functional.max_unpool2d(cn_a,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cn_a.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                cn_b=nn.functional.max_unpool2d(cn_b,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cn_b.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    cp_a=nn.functional.max_unpool2d(cp_a.float(),indices=indexes,
                                                     kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cp_a.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    cp_b=nn.functional.max_unpool2d(cp_b.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cp_b.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    cn_a=nn.functional.max_unpool2d(cn_a.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cn_a.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    cn_b=nn.functional.max_unpool2d(cn_b.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cn_b.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            #reshape:   
            cp_a=cp_a.view(numOutputs,batchSize,
                           cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1]).permute(1,0,2,3,4)
            cp_b=cp_b.view(numOutputs,batchSize,
                           cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1]).permute(1,0,2,3,4)
            cn_a=cn_a.view(numOutputs,batchSize,
                           cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1]).permute(1,0,2,3,4)
            cn_b=cn_b.view(numOutputs,batchSize,
                           cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1]).permute(1,0,2,3,4)

            #step 4- Multiplication with input
            aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1,1,1)

            RJ=alpha*(torch.mul(aJp.squeeze(1),cp_a.squeeze(1))+\
                      torch.mul(aJn.squeeze(1),cp_b.squeeze(1)))
            RJ=RJ+beta*(torch.mul(aJn.squeeze(1),cn_a.squeeze(1))+\
                        torch.mul(aJp.squeeze(1),cn_b.squeeze(1)))
            
        except:
            cconc=nn.functional.conv_transpose2d(torch.cat((sp,sp,sn,sn),dim=1),
                                                 torch.cat((Wp,Wn,Wp,Wn),dim=0),
                                                 bias=None,stride=stride,
                                                 groups=4,output_padding=(1,1))
            
            size=int(cconc.shape[1]/4)
            cp_a=cconc[:,0:size,:,:]
            cp_b=cconc[:,size:2*size,:,:]
            cn_a=cconc[:,2*size:3*size,:,:]
            cn_b=cconc[:,3*size:4*size,:,:]

            cp_a=cp_a.view(batchSize,numOutputs,cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.view(batchSize,numOutputs,cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.view(batchSize,numOutputs,cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.view(batchSize,numOutputs,cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])

            #Unpool: (propagate relevance through Maxpool)
            cp_a=cp_a.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1])
            cp_b=cp_b.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1])
            cn_a=cn_a.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1])
            cn_b=cn_b.permute(1,0,2,3,4).reshape(numOutputs*batchSize,
                                                 cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1])
            indexes=indexes.repeat(numOutputs,1,1,1)
            #unpool:
            try:
                cp_a=nn.functional.max_unpool2d(cp_a,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp_a.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                cp_b=nn.functional.max_unpool2d(cp_b,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cp_b.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                cn_a=nn.functional.max_unpool2d(cn_a,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cn_a.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                cn_b=nn.functional.max_unpool2d(cn_b,indices=indexes,kernel_size=layer.kernel_size,
                                             stride=layer.stride,padding=layer.padding,
                                             output_size=(cn_b.shape[0],
                                                          aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            except:
                with torch.cuda.amp.autocast(enabled=False):
                    cp_a=nn.functional.max_unpool2d(cp_a.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cp_a.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    cp_b=nn.functional.max_unpool2d(cp_b.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cp_b.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    cn_a=nn.functional.max_unpool2d(cn_a.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cn_a.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
                    cn_b=nn.functional.max_unpool2d(cn_b.float(),indices=indexes,
                                                    kernel_size=layer.kernel_size,
                                                 stride=layer.stride,padding=layer.padding,
                                                 output_size=(cn_b.shape[0],
                                                              aJ.shape[1],aJ.shape[2],aJ.shape[3]))
            #reshape:   
            cp_a=cp_a.view(numOutputs,batchSize,
                           cp_a.shape[-3],cp_a.shape[-2],cp_a.shape[-1]).permute(1,0,2,3,4)
            cp_b=cp_b.view(numOutputs,batchSize,
                           cp_b.shape[-3],cp_b.shape[-2],cp_b.shape[-1]).permute(1,0,2,3,4)
            cn_a=cn_a.view(numOutputs,batchSize,
                           cn_a.shape[-3],cn_a.shape[-2],cn_a.shape[-1]).permute(1,0,2,3,4)
            cn_b=cn_b.view(numOutputs,batchSize,
                           cn_b.shape[-3],cn_b.shape[-2],cn_b.shape[-1]).permute(1,0,2,3,4)

            #step 4- Multiplication with input
            aJp=aJp.unsqueeze(1).repeat(1,numOutputs,1,1,1)
            aJn=aJn.unsqueeze(1).repeat(1,numOutputs,1,1,1)

            RJ=alpha*(torch.mul(aJp.squeeze(1),cp_a.squeeze(1))+\
                      torch.mul(aJn.squeeze(1),cp_b.squeeze(1)))
            RJ=RJ+beta*(torch.mul(aJn.squeeze(1),cn_a.squeeze(1))+\
                        torch.mul(aJp.squeeze(1),cn_b.squeeze(1)))
            
        if(len(RJ.shape)<5):
            RJ=RJ.unsqueeze(1)
            
        
    else:
        raise ValueError('Rule must be e, z+ or AB')

    return RJ

def LRPBNReLU(BN,rK,aJ,aK,e,rule='e',alpha=2,beta=-1):
    #size of batch dimension (0):
    batchSize=rK.shape[0]
    #size of classes dimension (1):
    numOutputs=rK.shape[1]
    channels=rK.shape[2]
    
    #identity weights for convolution (to get BN weights):
    stride=(1,1)
    bias=torch.zeros(channels).type_as(rK)
    weights=torch.zeros(channels,channels,1,1).type_as(rK)
    for i in range(channels):
        weights[i,i,:,:]=torch.ones(1,1).type_as(rK)
        
    #get equivalent convolution
    w,b=FuseBN(layerWeights=weights,layerBias=bias, BN=BN, aKConv=aJ,bias=True)
    
    RJ=LRPConvReLU(layer=None,rK=rK,aJ=aJ,aK=aK,e=e,rule=rule,alpha=alpha,beta=beta,
                   weights=w,bias=b,stride=stride,padding=(0,0))
    
    return RJ

#Auxiliary functions:

def AppendOutput(self,input,output):
    #forward hook to save layer output
    
    globals.X.append(output)
    
def AppendInput(self,input,output):
    #forward hook to save layer input
    
    globals.XI.append(input[0])
    
def AppendBoth(self,input,output):
    #forward hook to save layer input
    globals.X.append(output)
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


def ChangeInplace(m: torch.nn.Module):
    #function to remove inplace operations

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

class LN (nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer=layer
        
    def forward(self, rK, aJ, aK):
        y=f.LRPConvReLU(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y

def ChangeNorm(module,normLayer,oldName=''):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            if normLayer=='instanceNorm':
                new_layer = nn.InstanceNorm2d(child.num_features)
                #default:affine false, produces 0 bias dnn
            elif normLayer=='instanceNormAffine':
                new_layer = nn.InstanceNorm2d(child.num_features,affine=True)
            elif normLayer=='instanceNormTrack':
                new_layer = nn.InstanceNorm2d(child.num_features,affine=False,
                                              track_running_stats=True)
            elif normLayer=='layerNorm':
                #print(oldName+'.'+name)
                
                if oldName+'.'+name=='features.norm0':
                    shape=(64,112,112)
                elif 'denseblock1' in oldName+'.'+name or 'transition1' in oldName+'.'+name:
                    shape=(child.num_features,56,56)
                elif 'denseblock2' in oldName+'.'+name or 'transition2' in oldName+'.'+name:
                    shape=(child.num_features,28,28)
                elif 'denseblock3' in oldName+'.'+name or 'transition3' in oldName+'.'+name:
                    shape=(child.num_features,14,14)
                elif 'denseblock4' in oldName+'.'+name or (oldName+'.'+name)=='features.norm5':
                    shape=(child.num_features,7,7)
                else:
                    #shape=(1,1,1)
                    raise ValueError('Layer norm conversion only implemented for densenet121')
                #if oldName+'.'+name in list(shapes.keys()):
                #    shape=shapes[oldName+'.'+name]
                #else:
                #    shape=(1,1,1)
                new_layer = nn.LayerNorm(shape)
            else:
                raise ValueError('Unrecognized normLayer')
            setattr(module, name, new_layer)
        else:
            if oldName=='':
                ChangeNorm(child,normLayer,oldName=name)
            else:
                ChangeNorm(child,normLayer,oldName=oldName+'.'+name)
                
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        try:
            self.input = input[0].clone()
        except:
            #maxpool
            self.input = (input[0][0].clone(),input[0][1].clone())
        try:
            self.output = output.clone()
        except:
            #maxpool
            self.output = (output[0].clone(),output[1].clone())
    def close(self):
        self.hook.remove()
        
class HookRelevance():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output.clone()
    def close(self):
        self.hook.remove()
        
class HookFwd():
    def __init__(self, module,mode):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.mode=mode
    def hook_fn(self, module, input, output):
        if self.mode=='output':
            self.x = output#.clone()
        elif self.mode=='input':
            self.x = input[0]#.clone()
        else:
            raise ValueError('Unrecognized mode in hook')
    def close(self):
        self.hook.remove()
        
def InsertFSPHooksDense(m: torch.nn.Module,name):
    #Function to insert multiple forward hooks in the classifier, for later use in the LRP block
    #m: classifier
    
    children = dict(m.named_children())
    output = {}
    if (('LRPlayer' in name) or ('LRPTransition' in name)):
        return HookRelevance(m)
    else:
        for name, child in children.items():
            try:
                output[name] = InsertFSPHooksDense(child,name)
            except TypeError:
                output[name] = InsertFSPHooksDense(child,name)
    return output

def getRelevance(d,x,name):
    for k, v in d.items():
        if isinstance(v, dict):
            getRelevance(v,x,k)
        else:
            try:
                x[name+k]=v.output
            except:
                x[name+k]=v.x
            
def InsertIO(m: torch.nn.Module):
    #Function to insert multiple forward hooks in the classifier, for later use in the LRP block
    #m: classifier
    
    children = dict(m.named_children())
    output = {}
    if children == {}:
        #m.register_forward_hook(AppendBoth)
        #l=globals.LayerIndex
        #globals.LayerIndex=globals.LayerIndex+1
        #return (m,l)
        return (m,Hook(m))
    else:
        for name, child in children.items():
            try:
                output[name] = InsertIO(child)
            except TypeError:
                output[name] = InsertIO(child)
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

def ChangeRule(m: torch.nn.Module,rule):
    #function to change all rules in the network
    #m: ISNet
    
    children = dict(m.named_children())
    output = {}
    if hasattr(m,'rule'):
        m.rule=rule
    if children == {}:
        return (m)
    else:
        for name, child in children.items():
            try:
                output[name] = ChangeRule(child,rule)
            except TypeError:
                output[name] = ChangeRule(child,rule)
    return output

def ChangeMultiple(m: torch.nn.Module,multiple):
    #function to change "multiple" argument in the network
    #m: ISNet
    
    children = dict(m.named_children())
    output = {}
    if hasattr(m,'multiple'):
        m.multiple=multiple
    if children == {}:
        return (m)
    else:
        for name, child in children.items():
            try:
                output[name] = ChangeMultiple(child,multiple)
            except TypeError:
                output[name] = ChangeMultiple(child,multiple)
    return output

def ChangePositive(m: torch.nn.Module,positive):
    #function to change "positive" argument in the network
    #m: ISNet
    
    children = dict(m.named_children())
    output = {}
    if hasattr(m,'multiple'):
        m.positive=positive
    if children == {}:
        return (m)
    else:
        for name, child in children.items():
            try:
                output[name] = ChangePositive(child,positive)
            except TypeError:
                output[name] = ChangePositive(child,positive)
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
    #resets all global variables
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
    model=ISNet.classifierDNN
    remove_all_forward_hooks(model)
    return model
