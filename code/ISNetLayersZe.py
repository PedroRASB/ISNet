import torch
import torch.nn as nn
import ISNetFunctionsZe as f
import LRPDenseNetZe as LRPDenseNet
import globalsZe as globals
import re


#Transform function in ISNetFunctions in DNN layers:
class LRPDenseReLU (nn.Module):
    def __init__(self, layer,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward(self, rK, aJ, aK):
        y=f.LRPDenseReLU(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y
    
class LRPConvReLU (nn.Module):
    def __init__(self, layer,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward(self, rK, aJ, aK):
        y=f.LRPConvReLU(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y
    
class LRPOutput (nn.Module):
    def __init__(self,layer,multiple,positive=False,rule='e',ignore=None,amplify=1,highest=False,
                 randomLogit=False):
        super().__init__()
        self.layer=layer
        self.multiple=multiple
        self.positive=positive
        self.rule=rule
        self.ignore=ignore
        self.amplify=amplify
        self.highest=highest
        self.randomLogit=randomLogit
        
    def forward(self, aJ,y,label=None):
        return f.LRPOutput(layer=self.layer, aJ=aJ,
                           y=y,multiple=self.multiple,positive=self.positive,rule=self.rule,
                           ignore=self.ignore,amplify=self.amplify,highest=self.highest,label=label,
                           randomLogit=self.randomLogit)
    

    
class LRPSelectiveOutput (nn.Module):
    def __init__(self,layer,e,rule,highest,amplify=1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.rule=rule
        self.highest=highest#single heatmap, for highest logit
        self.amplify=amplify
        
    def forward(self, aJ, aK, label=None):
        return f.LRPClassSelectiveOutputLayer(layer=self.layer, aJ=aJ, aK=aK, e=self.e,
                                              highest=self.highest,amplify=self.amplify,
                                              label=label)
        
class LRPPool2d (nn.Module):
    def __init__(self,layer,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward(self, rK, aJ, aK):
        y= f.LRPPool2d(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y
    
class LRPAdaptativePool2d (nn.Module):
    def __init__(self,layer,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward(self, rK, aJ,aK):
        y= f.LRPPool2d(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,
                       adaptative=True,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y
    
class LRPSum (nn.Module):
    def __init__(self,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward(self,rK,aJ,aK):
        y= f.LRPSum(rK=rK,aJ=aJ,aK=aK,e=self.e,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y
    
class LRPElementWiseSum (nn.Module):
    def __init__(self,e,rule):
        super().__init__()
        self.e=e
        self.rule=rule
        #if self.rule!='e':
        #    raise ValueError('LRPElementWiseSum can only be implemented for e rule')
        
    def forward(self,rK,a,b):
        y= f.LRPElementWiseSUM(rK,a,b,self.e,self.rule)
        return y  
            
class LRPLogSumExpPool(nn.Module):
    def __init__(self,e,lse_r=6):
        super().__init__()
        self.e=e
        self.lse_r=lse_r
        
    def forward(self, rK, aJ):
        y=f.LRPLogSumExpPool(rK, aJ, self.lse_r , self.e)
        return y
    
class LRPPool2dBNReLU (nn.Module):
    def __init__(self,layer,BN,e,rule,Ch0=0,Ch1=None,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        

    def forward (self,rK, aJ, aKConv):
        y= f.LRPPool2dBNReLU(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,e=self.e,
                                 aKConv=aKConv,
                                 Ch0=self.Ch[0],Ch1=self.Ch[1],rule=self.rule,
                                 alpha=self.alpha,beta=self.beta)
        return y

class MultiBlockPoolBNReLU (nn.Module):
    def __init__(self,layer,BN,e,rule,Ch0=0,Ch1=None,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward (self, rK, aJ, aK, aKPool):
        y= f.MultiBlockPoolBNReLU(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,e=self.e,
                                  aKPool=aKPool,
                                  Ch0=self.Ch[0],Ch1=self.Ch[1],rule=self.rule,
                                  alpha=self.alpha,beta=self.beta)
        return y
    
class MultiBlockMaxPoolBNReLU (nn.Module):
    def __init__(self,layer,BN,e,rule,Ch0=0,Ch1=None,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward (self, rK, aJ, aK, aKPool):
        y= f.MultiBlockMaxPoolBNReLU(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,e=self.e,
                                      aKPool=aKPool,
                                      Ch0=self.Ch[0],Ch1=self.Ch[1],rule=self.rule,
                                      alpha=self.alpha,beta=self.beta)
        return y
    
class MultiLayerConvBNReLU (nn.Module):
    def __init__(self,layer,BN,e,rule,Ch0=0,Ch1=None,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward (self,rK,aJ,aKConv,aK):                         
        y= f.MultiLayerConvBNReLU(layer=self.layer,BN=self.BN,rK=rK,aK=aK,aJ=aJ,e=self.e,
                                      aKConv=aKConv,
                                      Ch0=self.Ch[0],Ch1=self.Ch[1],rule=self.rule,
                                      alpha=self.alpha,beta=self.beta)
        return y
    
class LRPConvBNReLU (nn.Module):
    def __init__(self,layer,BN,e,rule,Ch0=0,Ch1=None,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward (self,rK,aJ,aKConv,aK):         
        y= f.LRPConvBNReLU(layer=self.layer,BN=self.BN,rK=rK,aK=aK,aJ=aJ,e=self.e,
                               aKConv=aKConv,
                               Ch0=self.Ch[0],Ch1=self.Ch[1],rule=self.rule,
                               alpha=self.alpha,beta=self.beta)
        return y

class LRPBNReLU (nn.Module):
    def __init__(self,BN,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.BN=BN
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward (self,rK,aJ,aK):         
        y= f.LRPBNReLU(BN=self.BN,rK=rK,aK=aK,aJ=aJ,e=self.e,
                       rule=self.rule,
                       alpha=self.alpha,beta=self.beta)
        return y
    
class w2RuleInput (nn.Module):
    def __init__(self,layer,e):
        super().__init__()
        self.e=e
        self.layer=layer
        
    def forward (self,rK,aJ,aK):
        y= f.w2RuleInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e)
        return y
    
class w2BNRuleInput (nn.Module):
    def __init__(self,layer,BN,e):
        super().__init__()
        self.e=e
        self.BN=BN
        
    def forward (self,rK,aJ,aKConv,aK):
        y= f.w2BNRuleInput(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,
                               e=self.e,aKConv=aKConv)
        return y
    
class ZbRuleConvBNInput (nn.Module):
    def __init__(self,layer,BN,e,l=0,h=1,op=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aKConv,aK):
        try:
            y= f.ZbRuleConvBNInput(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,
                                       aKConv=aKConv,e=self.e,
                                       l=self.l,h=self.h)  
        except:#legacy, if self.op is missing
            y= f.ZbRuleConvBNInput(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,
                                       aKConv=aKConv,e=self.e,
                                       l=self.l,h=self.h)  
        return y
    
class ZbRuleConvInput (nn.Module):
    def __init__(self,layer,e,l=0,h=1,op=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aK):
        try:
            y= f.ZbRuleConvInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                                 e=self.e,l=self.l,h=self.h,op=self.op)
        except:#legacy, if self.op is missing
            y= f.ZbRuleConvInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                                 e=self.e,l=self.l,h=self.h)
        return y
    
class ZbRuleDenseInput (nn.Module):
    def __init__(self,layer,e,l=0,h=1,op=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aK):
        y= f.ZbRuleDenseInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                             e=self.e,l=self.l,h=self.h)
        return y
    
class LRPMaxPool (nn.Module):
    def __init__(self,layer,e,rule,alpha=2,beta=-1):
        super().__init__()
        self.e=e
        self.layer=layer
        self.rule=rule
        self.alpha=alpha
        self.beta=beta
        
    def forward (self,rK,aJ,aK):
        y= f.LRPMaxPool2d(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,rule=self.rule,
                         alpha=self.alpha,beta=self.beta)
        return y

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#DenseNet specific modules:


    
class LRPDenseLayer (nn.Module):
    def __init__(self,DenseLayerIdx,DenseBlockIdx,model,e,rule):
        #propagates relevance through a dense layer inside dense block
        #DenseLayerIdx: index of dense layer in block
        #DenseBlockIdx: Dense block index
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e term
        super(LRPDenseLayer,self).__init__()
        #Ch0 and Ch1: channels of the dense layer output in the BNs
        DenseBlock=model['features']['denseblock'+str(DenseBlockIdx)]
        self.DenseLayer=DenseBlock['denselayer'+str(DenseLayerIdx)]
        
        Ch0=self.DenseLayer['norm1'][0].num_features
        Ch1=Ch0+self.DenseLayer['conv2'][0].weight.shape[0]
        
        #BNs: all batch normalizations that take the DenseLayer output
        BNs=[]
        #indexes of batchnorm outputs in the global fdw:
        self.BNOut=[]
        #indexes (negative) of relevances from posterior layers in block, starting from last:
        self.Rs=[]
        #transition layer:
        if ('transition'+str(DenseBlockIdx) in model['features']):
            #not last denseblock
            BNs.append(model['features']['transition'+str(DenseBlockIdx)]['norm'][0])
            self.BNOut.append(model['features']['transition'+str(DenseBlockIdx)]['norm'][1])
        else: 
            #last denseblock
            BNs.append(model['features']['norm5'][0])
            self.BNOut.append(model['features']['norm5'][1])
        #iterate from last layer in the DenseBlock (BN order will match self.R order):
        j=(len(DenseBlock)-DenseLayerIdx)*(-1)
        self.Rs.append(j-1)#transition layer
        for key,layer in reversed(list(DenseBlock.items())):
            if(int(key[-1])>DenseLayerIdx):
                BNs.append(layer['norm1'][0])
                self.BNOut.append(layer['norm1'][1])
                self.Rs.append(j)
                j=j+1
                
        #initialize layers:
        #through batchnorms and ReLUs in posterior layers and conv2 in this layer
        self.L2=MultiLayerConvBNReLU(layer=self.DenseLayer['conv2'][0],
                                     BN=BNs,e=e,Ch0=Ch0,Ch1=Ch1,rule=rule)
        #through conv1, norm2 and relu2:
        self.L1=LRPConvBNReLU(layer=self.DenseLayer['conv1'][0],BN=self.DenseLayer['norm2'][0],
                                 e=e,rule=rule)
        
    def forward (self,X,R):
        #X:list of all layer activations
        #R:list of relevances after dense layers, starting from last DNN layer
        
        #through batchnorms and ReLUs in posterior layers and conv2 in this layer:
        #get relevances from posterior layers
        rK=[]
        for i in self.Rs:
            rK.append(R[i])
        #get activations from posterior layers:
        aK=[]
        for i in self.BNOut:
            aK.append(X[i])
        #get activations for current layer:
        aKConv=X[self.DenseLayer['conv2'][1]]
        aJ=X[self.DenseLayer['relu2'][1]]     
        #propagate relevance:
        r=self.L2(rK=rK,aJ=aJ,aKConv=aKConv,aK=aK)
        #through conv1, norm2 and relu2:
        aJ=X[self.DenseLayer['relu1'][1]]
        aKConv=X[self.DenseLayer['conv1'][1]]
        aK=X[self.DenseLayer['norm2'][1]]
        r=self.L1(rK=r,aJ=aJ,aKConv=aKConv,aK=aK)
        
        return r

class LRPDenseBlock (nn.ModuleDict):
    def __init__(self,DenseBlockIdx,model,e,rule):
        #propagates relevance through a dense block
        #DenseBlockIdx: index of dense block in the DenseNet
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e term
        
        super(LRPDenseBlock,self).__init__()
        DenseBlock=model['features']['denseblock'+str(DenseBlockIdx)]
        if ('transition'+str(DenseBlockIdx) in model['features']):
            Transition=model['features']['transition'+str(DenseBlockIdx)]
            
        #initialize all dense layers:
        for i in reversed(list(range(len(DenseBlock)))):
            layer=LRPDenseLayer(DenseLayerIdx=i+1,DenseBlockIdx=DenseBlockIdx,
                                model=model,e=e,rule=rule)
            self.add_module('LRPlayer%d' % (i + 1), layer)
            
    def forward(self,X,r):
        #X:list of all layer activations
        #r:relevance before next block/transition first convolution

        R=[r]
        #propagate relevance through layer, from last in block
        for name,layer in self.items():
            R.append(layer(X=X,R=R))   
        #return all dense layers relevances+transition
        return R
    
class LRPDenseNetClassifier(nn.Module):
    def __init__(self,model,e,rule,multiple,positive,ignore,selective=False,highest=False,
                 amplify=1,randomLogit=False):
        #propagates relevance through the DenseNet last layers (after dense blocks)
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e term
        #ignore: lsit with classes which will not suffer attention control
        #selective: uses class selective propagation. 
        
        super(LRPDenseNetClassifier,self).__init__()

        #dense layer:
        try:
            classifier=model['classifier'][0]
        except:
            #model with dropout
            classifier=model['classifier']['1'][0]
        #num_features: number of channels after adaptative pool
        self.num_features=classifier.in_features
        
        if selective:
            self.SelectiveOut=LRPSelectiveOutput(classifier,e,rule,highest,amplify=amplify)
        else:
            #initialize output LRP layer:
            self.Out=LRPOutput(layer=classifier,multiple=multiple,positive=positive,
                               rule=rule,ignore=ignore,highest=highest,
                              amplify=amplify,randomLogit=randomLogit)
            #initialize LRP layer for classifier:
            self.Dense=LRPDenseReLU(classifier,e,rule)
        
        #pooling layer:
        pool=model['features']['AdaptPool'][0]
        self.PoolInput=model['features']['fReLU'][1]
        self.PoolOutput=model['features']['AdaptPool'][1]
        #initialize LRP layer for pooling:
        self.AdpPool=LRPAdaptativePool2d(pool,e=e,rule=rule)
        self.multiple=multiple
        self.rule=rule
        self.selective=selective
        self.highest=highest
        
    def forward(self,X,XI,y,label=None):
        #X:list of all layer activations 
        #XI: inputs of last layer
        #y: classifier outputs
        
        B=y.shape[0]#batch size
        if self.rule=='z+e':
            C=2
        elif self.multiple:
            C=y.shape[-1]
        elif not self.multiple:
            C=1
        else:
            raise ValueError('Invalid argument selection')
            
        if self.selective:
            R=self.SelectiveOut(aJ=XI[0],aK=y,label=label)
            R=R.view(B,C,self.num_features,1,1)
        else:
            #y:DNN output
            R=self.Out(aJ=XI[0],y=y,label=label)
            R=self.Dense(rK=R,aJ=XI[0],aK=y)
            R=R.view(B,C,self.num_features,1,1)
            
        R=self.AdpPool(rK=R,aJ=X[self.PoolInput],aK=X[self.PoolOutput])
        return R
        
class LRPDenseNetInitial(nn.Module):
    def __init__(self,model,e,rule,Zb=True):
        #propagates relevance through the DenseNet first layers (before dense blocks)
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        
        super(LRPDenseNetInitial,self).__init__()
        #all layers in the first dense block:
        BNs=[]
        self.BNOut=[]
        if ('transition1' in model['features']):
            BNs.append(model['features']['transition1']['norm'][0])
            self.BNOut.append(model['features']['transition1']['norm'][1])
        else:
            BNs.append(model['features']['norm5'][0])
            self.BNOut.append(model['features']['norm5'][1])
        self.Rs=[]
        DenseBlock=model['features']['denseblock1']
        self.features=model['features']
        #from last dense layer:
        j=len(DenseBlock)*(-1)
        self.Rs.append(j-1)#transition layer
        for key,layer in reversed(list(DenseBlock.items())):
            BNs.append(layer['norm1'][0])
            self.BNOut.append(layer['norm1'][1])
            self.Rs.append(j)
            j=j+1
            
        #init pool:
        self.pool=MultiBlockMaxPoolBNReLU(layer=model['features']['pool0'][0],BN=BNs,
                                       Ch1=model['features']['norm0'][0].num_features,
                                       e=e,rule=rule)
        if(Zb):
            self.conv=ZbRuleConvBNInput(layer=model['features']['conv0'][0],
                                        BN=model['features']['norm0'][0],
                                        e=e,l=0,h=1)
        else:
            self.conv=LRPConvBNReLU(layer=model['features']['conv0'][0],
                                           BN=model['features']['norm0'][0],
                                           e=e,rule=rule)
    def forward(self,X,x,r):
        #X:list of all layer activations 
        #x: model input images
        #r:relevances from first dense block
        
        rK=[]
        for i in self.Rs:
            rK.append(r[i])
        aK=[]
        for i in self.BNOut:
            aK.append(X[i])
        #get activations for current layer:
        aKPool=X[self.features['pool0'][1]]
        aJ=X[self.features['relu0'][1]] 
        R=self.pool(rK=rK, aJ=aJ, aKPool=aKPool, aK=aK)
        R=self.conv(rK=R,aJ=x,aKConv=X[self.features['conv0'][1]],
                    aK=X[self.features['norm0'][1]])
        return R
        
class LRPDenseNetTransition(nn.Module):
    def __init__(self,TransitionIdx,model,e,rule):
        #propagates relevance through transition layer
        #TransitionIdx: index of transiton layer
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e term
        
        super(LRPDenseNetTransition,self).__init__()
        
        #all layers in the first dense block:
        BNs=[]
        self.BNOut=[]
        nextT='transition'+str(TransitionIdx+1)
        self.current='transition'+str(TransitionIdx)
        if (nextT in model['features']):
            BNs.append(model['features'][nextT]['norm'][0])
            self.BNOut.append(model['features'][nextT]['norm'][1])
        else:
            BNs.append(model['features']['norm5'][0])
            self.BNOut.append(model['features']['norm5'][1])
        self.Rs=[]
        #next dense block:
        DenseBlock=model['features']['denseblock'+str(TransitionIdx+1)]
        self.features=model['features']
        #from last dense layer:
        j=len(DenseBlock)*(-1)
        self.Rs.append(j-1)#next transition layer
        for key,layer in reversed(list(DenseBlock.items())):
            BNs.append(layer['norm1'][0])
            self.BNOut.append(layer['norm1'][1])
            self.Rs.append(j)
            j=j+1
        #init pool:
        self.pool=MultiBlockPoolBNReLU(layer=model['features'][self.current]['pool'][0],BN=BNs,
                                       Ch1=model['features'][self.current]['conv'][0].out_channels,
                                       e=e,rule=rule)
        self.conv=LRPConvReLU(layer=model['features'][self.current]['conv'][0],
                                e=e,rule=rule)
        
    def forward(self,X,r):
        #X:list of all layer activations 
        #r:relevances from next dense block
        
        rK=[]
        for i in self.Rs:
            rK.append(r[i])
        aK=[]
        for i in self.BNOut:
            aK.append(X[i])
        #get activations for current layer:
        aKPool=X[self.features[self.current]['pool'][1]]
        aJ=X[self.features[self.current]['relu2'][1]] 
        R=self.pool(rK=rK, aJ=aJ, aKPool=aKPool, aK=aK)
        R=self.conv(rK=R,aJ=X[self.features[self.current]['relu'][1]],
                    aK=X[self.features[self.current]['conv'][1]])
        return R        
        
class _LRPDenseNet(nn.ModuleDict):
    def __init__(self,DenseNet,e,rule,multiple,positive,ignore=None,Zb=True,selective=False,
                 highest=False,amplify=1,detach=True,features='fReLU',
                 storeRelevance=False,FSP=False,randomLogit=False):
        #LRP Block: Propagates relevance through DenseNet
        #network: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #rule: LRP rule, e or z+e
        #ignore: lsit with classes which will not suffer attention control
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: set to true for selective ISNet
        #amplify: multiplier to LRP heatmap
        #multiple: multiple heatmaps per sample, use for Original ISNet
        #detach: detach biases from graph when propagating relevances
        #FSP: full signal penalization in LDS
        #storeRelevance: stores intermediate heatmaps for  LDS
        #randomLogit: Stochastic ISNet
        #features: deprecated
        
        super(_LRPDenseNet,self).__init__()
        
        #check for inconsistent parameters:
        if (randomLogit and selective):
            raise ValueError('Set randomLogit or selective, not both')
        if(not torch.backends.cudnn.deterministic):
            raise ValueError('Please set torch.backends.cudnn.deterministic=True')
        if(rule=='AB' or rule=='z+' or rule=='composite'):
            print('ATTENTION: Using '+rule+' rule.')
            print('AB and z+ rule should not be used for background relevance minimization. Use them for vizualization only.')
        if(multiple and rule=='z+e'):
            raise ValueError('Please set multiple=False with z+e rule')
        if(selective and rule=='z+e'):
            raise ValueError('Selective should not be used with z+e rules')
        if (not multiple and not selective and rule=='e'):
            print('ATTENTION: multiple and selective set to false with e rule. For ISNet training, please use multiple=True selective=False rule=e (original ISNet), or multiple=False selective=True rule=e (selective isnet), or multiple=False selective=False rule=z+e (isnet z+e)')

        #register hooks:
        f.resetGlobals()
        model=f.InsertHooks(DenseNet)
        
        try:
            classifier=model['classifier'][0]
        except:
            #model with dropout
            classifier=model['classifier']['1'][0]
        classifier.register_forward_hook(f.AppendInput)
        
        #classifier and last layers:
        composite=False
        last_e=e
        if rule=='composite':
            composite=True
            rule='e'
            last_e=1e-6#LRP-0 like
        self.add_module('LRPFinalLayers',
                        LRPDenseNetClassifier(model=model,e=last_e,rule=rule,multiple=multiple,
                                             positive=positive,ignore=ignore,selective=selective,
                                             highest=highest,amplify=amplify,
                                             randomLogit=randomLogit))
        changeRule=0
        if composite:
            for key in reversed(list(model['features'])):
                if ('denseblock' in key):
                    maximum=int(key[-1])
                    break
            changeRule=int(maximum/2)
                    
        #DenseBlocks and transitions:
        for key in reversed(list(model['features'])):
            if ('denseblock' in key):
                block=int(key[-1])
                self.add_module('LRPDenseBlock%d' % (block),
                                LRPDenseBlock(block,model=model,e=e,rule=rule))
            if ('transition' in key):
                trans=int(key[-1])
                if (composite and trans==changeRule):
                    rule='AB'
                self.add_module('LRPTransition%d' % (trans),
                                LRPDenseNetTransition(trans,model=model,e=e,rule=rule))
        
        #initial layers:
        self.add_module('LRPInitialLayers',
                        LRPDenseNetInitial(model=model,e=e,Zb=Zb,rule=rule))
        
        globals.detach=detach
        self.features=model['features'][features][1]
        
        #register hooks to store relevance
        if not FSP and storeRelevance:
            self.storedRelevance={}
            self.storedRelevance['LRPFinalLayers']=f.HookRelevance(self.LRPFinalLayers)
            for name, module in self.named_modules():
                if 'LRPTransition' in name and '.' not in name:
                    self.storedRelevance[name]=f.HookRelevance(module)
        elif FSP:
            self.storedRelevance=f.InsertFSPHooksDense(self,'LRPBlock')
            self.storedRelevance['LRPFinalLayers']=f.HookRelevance(self.LRPFinalLayers)
            self.storedRelevance['LRPInitialLayers']=f.HookRelevance(self.LRPInitialLayers.pool)
        
    def forward(self,x,y,returnFeatures=False,LRPFeatures=False,label=None):
        #x: model input
        #y: classifier output
        
        R=self.LRPFinalLayers(X=globals.X,XI=globals.XI,y=y,label=label)
        featuresLRP=R.clone()
        for name,layer in self.items():
            if ((name != 'LRPFinalLayers') and (name != 'LRPInitialLayers')
                 and (name != 'features')):
                R=layer(X=globals.X,r=R)
        R=self.LRPInitialLayers(X=globals.X,x=x,r=R)

        featureMaps=globals.X[self.features]
            
        #clean global variables:
        globals.X=[]
        globals.XI=[]
        
        if returnFeatures and LRPFeatures:
            return R,featureMaps,featuresLRP
        elif LRPFeatures:
            return R,featuresLRP
        elif returnFeatures:
            return R,featureMaps
        else:
            return R

class IsDense(nn.Module):
    def __init__(self,DenseNet,e=1e-2,heat=True,ignore=None,
                 Zb=True,rule='e',multiple=True,positive=False,
                 selective=False,highest=False,amplify=1,detach=True,
                 randomLogit=False):
        #Creates ISNet based on DenseNet
        #DenseNet: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #heat: allows relevance propagation and heatmap creation. If False,
        #no signal is propagated through LRP block.
        #rule: LRP rule, choose e, z+, AB or z+e. For background relevance 
        #minimization use either e or z+e.
        #multiple: whether to produce a single heatmap or one per class
        #positive: whether to make output relevance positive or not
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: makes selective rule with a single heatmap
        
        super (IsDense,self).__init__()
        self.DenseNet=DenseNet
        self.LRPBlock=_LRPDenseNet(self.DenseNet,e=e,Zb=Zb,rule=rule,
                                   multiple=multiple,positive=positive,
                                   selective=selective,
                                   ignore=ignore,highest=highest,
                                   amplify=amplify,detach=detach,
                                   randomLogit=randomLogit)
        self.heat=heat

    def forward(self,x):
        #x:input
        
        y=self.DenseNet(x)
        if(self.heat):
            R=self.LRPBlock(x=x,y=y)
            return y,R
        else: 
            globals.X=[]
            globals.XI=[]
            return y

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#Resnet specific modules:
from collections import OrderedDict

class IgnoreIndexes(nn.Module):   
    #special layer to ignore indexes of previous max pooling layer
    def __init__(self):
        super(IgnoreIndexes, self).__init__()
    def forward(self, x):
        return(x[0])
    
class LRPResNetBottleneck (nn.Module):
    def __init__(self,model,e,rule):
        #model: bottleneck
        super(LRPResNetBottleneck ,self).__init__()
        
        self.bottleneck=model
        
        self.LRPSum=LRPElementWiseSum(e,rule=rule)
        
        #check for downsampling:
        if 'downsample' in self.bottleneck:
            self.LRPDownsample=LRPConvBNReLU(layer=self.bottleneck['downsample']['0'][0],
                                               BN=self.bottleneck['downsample']['1'][0],
                                               e=e,
                                               rule=rule)
            
        self.LRPc3=LRPConvBNReLU(layer=self.bottleneck['conv3'][0],
                                   BN=self.bottleneck['bn3'][0],
                                   e=e,
                                   rule=rule)
        self.LRPc2=LRPConvBNReLU(layer=self.bottleneck['conv2'][0],
                                   BN=self.bottleneck['bn2'][0],
                                   e=e,
                                   rule=rule)
        self.LRPc1=LRPConvBNReLU(layer=self.bottleneck['conv1'][0],
                                   BN=self.bottleneck['bn1'][0],
                                   e=e,
                                   rule=rule)
        
    def forward(self,rK):
        if 'downsample' not in self.bottleneck:#identity shotcut
            RShortcut,R=self.LRPSum(rK,self.bottleneck['conv1'][1].input,
                                    self.bottleneck['bn3'][1].output)
        else:
            RShortcut,R=self.LRPSum(rK,self.bottleneck['downsample']['1'][1].output,
                                    self.bottleneck['bn3'][1].output)
            RShortcut=self.LRPDownsample(RShortcut,
                                         aJ=self.bottleneck['downsample']['0'][1].input,
                                         aKConv=self.bottleneck['downsample']['0'][1].output,
                                         aK=self.bottleneck['downsample']['1'][1].output)

        R=self.LRPc3 (R,aJ=self.bottleneck['conv3'][1].input,
                      aKConv=self.bottleneck['conv3'][1].output,
                      aK=self.bottleneck['bn3'][1].output)
        
        R=self.LRPc2 (R,aJ=self.bottleneck['conv2'][1].input,
                      aKConv=self.bottleneck['conv2'][1].output,
                      aK=self.bottleneck['bn2'][1].output)
        
        R=self.LRPc1 (R,aJ=self.bottleneck['conv1'][1].input,
                      aKConv=self.bottleneck['conv1'][1].output,
                      aK=self.bottleneck['bn1'][1].output)

        R=R+RShortcut
        
        #print(torch.sum(R[:,0]))
        
        return R
    
class LRPResNetBasicBlock (nn.Module):
    def __init__(self,model,e,rule):
        #model: bottleneck
        super(LRPResNetBasicBlock ,self).__init__()
        
        self.block=model
        
        self.LRPSum=LRPElementWiseSum(e,rule=rule)
        
        #check for downsampling:
        if (list(self.block.keys())[0]=='conv1'):#conv bn relu
            if 'downsample' in self.block:
                self.LRPDownsample=LRPConvBNReLU(layer=self.block['downsample']['0'][0],
                                                   BN=self.block['downsample']['1'][0],
                                                   e=e,
                                                   rule=rule)

            self.LRPc2=LRPConvBNReLU(layer=self.block['conv2'][0],
                                       BN=self.block['bn2'][0],
                                       e=e,
                                       rule=rule)
            self.LRPc1=LRPConvBNReLU(layer=self.block['conv1'][0],
                                       BN=self.block['bn1'][0],
                                       e=e,
                                       rule=rule)
            
        elif (list(self.block.keys())[0]=='bn1'):#bn relu conv
            if 'downsample' in self.block:
                self.LRPDownsample=LRPConvReLU(layer=self.block['downsample']['0'][0],
                                               e=e,rule=rule)
            self.LRPc2=LRPConvReLU(layer=self.block['conv2'][0],
                                   e=e,rule=rule)
            self.LRPc1=LRPConvBNReLU(layer=self.block['conv1'][0],
                                       BN=self.block['bn2'][0],
                                       e=e,rule=rule)
            self.LRPbn=LRPBNReLU(BN=self.block['bn1'][0],
                                 e=e,rule=rule)
            
        else:
            raise ValueError('Unrecognized block configuration')
        
        
    def forward(self,rK):
        
        if (list(self.block.keys())[0]=='conv1'):
            if 'downsample' not in self.block:#identity shotcut
                RShortcut,R=self.LRPSum(rK,self.block['conv1'][1].input,
                                        self.block['bn2'][1].output)
            else:
                RShortcut,R=self.LRPSum(rK,self.block['downsample']['1'][1].output,
                                        self.block['bn2'][1].output)
                RShortcut=self.LRPDownsample(RShortcut,
                                             aJ=self.block['downsample']['0'][1].input,
                                             aKConv=self.block['downsample']['0'][1].output,
                                             aK=self.block['downsample']['1'][1].output)

            R=self.LRPc2 (R,aJ=self.block['conv2'][1].input,
                          aKConv=self.block['conv2'][1].output,
                          aK=self.block['bn2'][1].output)


            R=self.LRPc1 (R,aJ=self.block['conv1'][1].input,
                          aKConv=self.block['conv1'][1].output,
                          aK=self.block['bn1'][1].output)

            
        
        else:
            
            
            if 'downsample' not in self.block:#identity shotcut
                RShortcut,R=self.LRPSum(rK,self.block['bn1'][1].input,
                                        self.block['conv2'][1].output)
            else:
                RShortcut,R=self.LRPSum(rK,self.block['downsample']['0'][1].output,
                                        self.block['conv2'][1].output)
                #print('shortcut')
                RShortcut=self.LRPDownsample(RShortcut,
                                             aJ=self.block['downsample']['0'][1].input,
                                             aK=self.block['downsample']['0'][1].output)
                
            if torch.isnan(R).any():
                    print('nan LRPshortcut')
            #print(self.block)
            #print('LRPc2')
            R=self.LRPc2 (R,aJ=self.block['conv2'][1].input,
                          aK=self.block['conv2'][1].output)
            if torch.isnan(R).any():
                    print('nan LRPc2')
            #print('LRPc1')
            R=self.LRPc1 (R,aJ=self.block['conv1'][1].input,
                          aKConv=self.block['conv1'][1].output,
                          aK=self.block['bn2'][1].output)
            if torch.isnan(R).any():
                    print('nan LRPc1')
            #print('LRPbn')
            R=self.LRPbn (R,aJ=self.block['bn1'][1].input,
                            aK=self.block['bn1'][1].output)
            if torch.isnan(R).any():
                    print('nan LRPbn')
        #print(torch.sum(R[:,1]))
        R=R+RShortcut
        return R
            
class _LRPResNetLayer (nn.ModuleDict):
    def __init__(self,model,idx,e,rule):
        #idx: index of the resnet layer
        super(_LRPResNetLayer ,self).__init__()
        layer=model['layer'+str(idx)]
        bottle=False
        if 'conv3' in list(layer['0'].keys()):
            bottle=True
            
        for bottleneck in reversed(list(layer.keys())):
            if bottle:
                self.add_module('LRPBottleneck'+bottleneck,
                                LRPResNetBottleneck(layer[bottleneck],e,rule))
            else:
                self.add_module('LRPBasicBlock'+bottleneck,
                                LRPResNetBasicBlock(layer[bottleneck],e,rule))

    def forward(self,rK):
        for name,layer in self.items():
            #print(name)
            rK=layer(rK)
        return rK
    
class LRPResNetClassifier (nn.Module):
    def __init__(self,model,e,rule,selective,highest,multiple,positive,ignore=None,amplify=1,
                 randomLogit=False):
        super(LRPResNetClassifier ,self).__init__()
        try:
            #model with dropout
            classifier=model['fc']['1'][0]
            self.clsIO=model['fc']['1'][1]
        except:
            classifier=model['fc'][0]
            self.clsIO=model['fc'][1]

        #num_features: number of channels after adaptative pool
        self.num_features=classifier.in_features
        
        if selective:
            if ignore is not None:
                raise ValueError('ignore not implemented for selective rule')
            self.SelectiveOut=LRPSelectiveOutput(classifier,e,rule,highest,amplify=amplify)
        else:
            #initialize output LRP layer:
            self.Out=LRPOutput(layer=classifier,multiple=multiple,positive=positive,
                               rule=rule,ignore=ignore,highest=highest,
                               amplify=amplify,randomLogit=randomLogit)
            #initialize LRP layer for classifier:
            self.Dense=LRPDenseReLU(classifier,e,rule)
        
        #pooling layer:
        pool=model['avgpool'][0]
        self.poolIO=model['avgpool'][1]
        #initialize LRP layer for pooling:
        self.AdpPool=LRPAdaptativePool2d(pool,e=e,rule=rule)
        
        if ('bn2' in list(model.keys())):
            self.bnReLU=LRPBNReLU(BN=model['bn2'][0],e=e,rule=rule)
        
        self.multiple=multiple
        self.rule=rule
        self.selective=selective
        self.highest=highest
        self.model=model
        
    def forward(self,LRPFeatures=False,label=None):
        #X:list of all layer activations 
        #XI: inputs of last layer
        #y: classifier outputs
        
        B=self.clsIO.output.shape[0]#batch size
        
        if self.rule=='z+e':
            C=2
        elif self.multiple:
            C=self.clsIO.output.shape[-1]
        elif not self.multiple:
            C=1
        else:
            raise ValueError('Invalid argument selection')
            
        if self.selective:
            R=self.SelectiveOut(aJ=self.clsIO.input,aK=self.clsIO.output,label=label)
            R=R.view(B,C,self.num_features,1,1)
            if torch.isnan(R).any():
                print('nan selective out')
                print('x=',self.clsIO.input,'y=',self.clsIO.output)
        else:
            #y:DNN output
            #print(label)
            R=self.Out(aJ=self.clsIO.input,y=self.clsIO.output,label=label)
            R=self.Dense(rK=R,aJ=self.clsIO.input,aK=self.clsIO.output)
            R=R.view(B,C,self.num_features,1,1)
            if torch.isnan(R).any():
                print('nan non-selective out')
                print('x=',self.clsIO.input,'y=',self.clsIO.output)
            
        R=self.AdpPool(rK=R,aJ=self.poolIO.input,aK=self.poolIO.output)
        Rfeat=R.clone()
        if torch.isnan(R).any():
            print('nan adpPool')
        if ('bn2' in list(self.model.keys())):
            R=self.bnReLU(rK=R,aJ=self.model['bn2'][1].input,
                          aK=self.model['bn2'][1].output)
            if torch.isnan(R).any():
                print('nan bnrelu final')
        if LRPFeatures:
            return R,Rfeat
        else:
            return R
        

class _LRPResNetBackbone (nn.ModuleDict):
    def __init__(self,model,e,Zb,rule,amplify=1):
        super(_LRPResNetBackbone ,self).__init__()
        
        backbone=model
        
        #all resnet layers (with bottlenecks inside)
        for layer in reversed(list(backbone.keys())):
            if ('layer' in layer and 'fa_layer' not in layer):
                self.add_module('LRP'+layer,
                                _LRPResNetLayer(model,int(layer[-1]),e,rule))
        
        #initial layers
        if 'maxpool' in list(model.keys()):
            self.add_module('LRPMaxpool',
                             LRPMaxPool(backbone['maxpool']['maxpool'][0],e,rule=rule))
        if not Zb:
            self.add_module('LRPFirstConv',
                            LRPConvBNReLU(layer=backbone['conv1'][0],
                                            BN=backbone['bn1'][0],
                                            e=e,
                                            rule=rule))
        else:
            self.add_module('LRPFirstConv',
                            ZbRuleConvBNInput(layer=backbone['conv1'][0],
                                              BN=backbone['bn1'][0],
                                              e=e,l=0,h=1))
            
        self.backbone=backbone
        self.amplify=amplify
    
    def forward(self,rK):
        if self.amplify>1:
            rK=rK*self.amplify
        #all bottlenecks:
        for name,layer in self.items():
            if ('LRPlayer' in name):
                #print(name)
                rK=layer(rK)
                
                if self.amplify>1:
                    rK=rK*self.amplify
                    
                if torch.isnan(rK).any():
                    print('nan layer: '+name)
                #print(name,rK.shape)
                #print(name,torch.min(rK[:,1]))
        #initial layers:
        if 'maxpool' in list(self.backbone.keys()):
            rK=self.LRPMaxpool(rK,aJ=self.backbone['maxpool']['maxpool'][1].input,
                               aK=self.backbone['maxpool']['maxpool'][1].output)
            if self.amplify>1:
                rK=rK*self.amplify
                
        rK=self.LRPFirstConv(rK,aJ=self.backbone['conv1'][1].input,
                            aKConv=self.backbone['conv1'][1].output,
                            aK=self.backbone['bn1'][1].output)
        if self.amplify>1:
            rK=rK*self.amplify
        
        return rK       


class _LRPResNet (nn.ModuleDict):
    def __init__(self,network,e,Zb,rule,selective,highest,multiple,positive=False,ignore=None,
                 amplify=1,detach=True,storeRelevance=False,FSP=False,randomLogit=False):
        #LRP Block: Propagates relevance through ResNet
        #network: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #rule: LRP rule, e or z+e
        #ignore: lsit with classes which will not suffer attention control
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: set to true for selective ISNet
        #amplify: multiplier to LRP heatmap
        #multiple: multiple heatmaps per sample, use for Original ISNet
        #detach: detach biases from graph when propagating relevances
        #FSP: full signal penalization in LDS
        #storeRelevance: stores intermediate heatmaps for  LDS
        #randomLogit: Stochastic ISNet
        
        super(_LRPResNet ,self).__init__()
        #check for inconsistent parameters:
        if (randomLogit and selective):
            raise ValueError('Set randomLogit or selective, not both')
        if(not torch.backends.cudnn.deterministic):
            raise ValueError('Please set torch.backends.cudnn.deterministic=True')
        if (selective and rule=='z+e'):
            raise ValueError('set selective=True and e rule, or z+e rule')
        if (selective and (multiple or not highest or positive)):
            print('For ISNet training with selective rule, set multiple=False, highest=True and positive=False')
        if (not multiple and not selective and rule=='e'):
            print('ATTENTION: multiple and selective set to false with e rule. For ISNet training, please use multiple=True selective=False rule=e (original ISNet), or multiple=False selective=True rule=e (selective isnet), or multiple=False selective=False rule=z+e (isnet z+e)')
        
        #register hooks:
        self.model=f.InsertIO(network)
        
        self.add_module('LRPClassifier',
                        LRPResNetClassifier(self.model,e,rule,
                                            selective,highest,multiple,positive,ignore,
                                            amplify=amplify,randomLogit=randomLogit))
        
        self.add_module('LRPBackbone',
                        _LRPResNetBackbone(self.model,e,Zb,rule))
        
        globals.detach=detach
        print('detach is: ',globals.detach)
        
        #register hooks to store relevance
        if storeRelevance and FSP:
            self.storedRelevance={}
            self.storedRelevance['LRPClassifier']=f.HookRelevance(self.LRPClassifier)
            for name, module in self.LRPBackbone.named_modules():
                if ((('LRPBottleneck' in name) or ('LRPBasicBlock' in name)) \
                    and name.count('.')==1):
                    self.storedRelevance[name]=f.HookRelevance(module)
        elif storeRelevance and not FSP:
            #penalize input of resnet stages 2, 3, 4, and output layers
            self.storedRelevance={}
            self.storedRelevance['LRPClassifier']=f.HookRelevance(self.LRPClassifier)
            for name, module in self.LRPBackbone.named_modules():
                if (('LRPlayer' in name) and ('LRPlayer1' not in name) \
                    and name.count('.')==0):
                    self.storedRelevance[name]=f.HookRelevance(module)
                    
            print('Stored Relevance:',list(self.storedRelevance.keys()))
        
    def forward(self,x,y=None,returnFeatures=False,LRPFeatures=False,label=None):
        R=self.LRPClassifier(label=label)
        if LRPFeatures:
            R,featuresLRP=self.LRPClassifier(LRPFeatures,label=label)
        if torch.isnan(R).any():
            print('nan last layers')
        #print('classifier',R.shape)
        R=self.LRPBackbone(R)
        if returnFeatures:
            featureMaps=self.model['avgpool'][1].input
        
        if returnFeatures and LRPFeatures:
            return R,featureMaps,featuresLRP
        elif LRPFeatures:
            return R,featuresLRP
        elif returnFeatures:
            return R,featureMaps
        else:
            return R
        
class LRPResNet (nn.Module):
    def __init__(self,network,heat,e,Zb,rule,selective,highest,multiple,positive,ignore=None,
                 amplify=1,detach=True,randomLogit=False):
        #LRP wrapper: network and block
        #network: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #rule: LRP rule, e or z+e
        #ignore: lsit with classes which will not suffer attention control
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: set to true for selective ISNet
        #amplify: multiplier to LRP heatmap
        #multiple: multiple heatmaps per sample, use for Original ISNet
        #detach: detach biases from graph when propagating relevances
        #randomLogit: Stochastic ISNet
        
        super(LRPResNet ,self).__init__()
        
        #remove inplace
        f.ChangeInplace(network) #no need, using .clone() in hooks
        
        #get MaxPool indexes:
        if hasattr(network, 'maxpool'):
            network.maxpool.return_indices=True
            network.maxpool=nn.Sequential(OrderedDict([('maxpool',network.maxpool),
                                                       ('special',IgnoreIndexes())]))
        
        self.resnet=network
        self.LRPBlock=_LRPResNet(self.resnet,e,Zb,rule,selective,highest,multiple,positive,ignore,
                                amplify=amplify,detach=detach,randomLogit=randomLogit)
        
        self.heat=heat
        
    def forward(self,x):
        y=self.resnet(x)
        
        if not self.heat:
            return y
        
        R=self.LRPBlock(x)
        return y,R
        
        
        
#functions for standard PyTorch sequential blocks and simple sequences of layers

class _LRPSequential (nn.ModuleDict):
    def __init__(self,network,e,Zb,rule,selective,highest,multiple,
                 detach=True,storeRelevance=False,amplify=1, inputShape=None,
                 preFlattenShape=None,classifierPresent=True,randomLogit=False):
        #LRP Block: Propagates relevance through arbitrary nn.Sequential
        #network: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #rule: LRP rule, e or z+e
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: set to true for selective ISNet
        #amplify: multiplier to LRP heatmap
        #multiple: multiple heatmaps per sample, use for Original ISNet
        #detach: detach biases from graph when propagating relevances
        #storeRelevance: stores intermediate heatmaps for  LDS
        #randomLogit: Stochastic ISNet
        #preFlattenShape: sginal shape before flatten, not needed for VGG
        #inputShape: input shape
        #classifierPresent: informs whether a fully-connected classifier is present in the DNN
        
        super(_LRPSequential ,self).__init__()
        #check for inconsistent parameters:
        if(not torch.backends.cudnn.deterministic):
            raise ValueError('Please set torch.backends.cudnn.deterministic=True')
        if (selective and rule=='z+e'):
            raise ValueError('set selective=True and e rule, or z+e rule')
        if (selective and (multiple or not highest)):
            print('For ISNet training with selective rule, set multiple=False and highest=True')
        if (not multiple and not selective and rule=='e'):
            print('ATTENTION: multiple and selective set to false with e rule. For ISNet training, please use multiple=True selective=False rule=e (original ISNet), or multiple=False selective=True rule=e (selective isnet), or multiple=False selective=False rule=z+e (isnet z+e)')
        
        self.multiple=multiple
        self.rule=rule
        self.selective=selective
        self.highest=highest
        globals.detach=detach
        self.classifierPresent=classifierPresent
        
        for i,layer in enumerate(network,0):
            name=layer.__class__.__name__
            if name=='MaxPool2d':
                network[i].return_indices=True
                network[i]=nn.Sequential(OrderedDict([('maxpool',network[i]),
                                                      ('special',IgnoreIndexes())]))
        
        #register hooks:
        self.model=f.InsertIO(network)
        #print(self.model)
        #print(self.model[list(self.model.keys())[-1]])
        
        if classifierPresent:
            #last layer:
            classifier=self.model[list(self.model.keys())[-1]][0]
            if selective:
                self.add_module('LRPSelectiveOut',
                                LRPSelectiveOutput(classifier,e,rule,highest,amplify=amplify))
            else:
                #initialize output LRP layer:
                self.add_module('LRPOut',
                                LRPOutput(layer=classifier,multiple=multiple,
                                   rule=rule,highest=highest,amplify=amplify,
                                    randomLogit=randomLogit))
                #initialize LRP layer for classifier:
                self.add_module('LRPOutDense',
                                LRPDenseReLU(classifier,e,rule))
        
        chain=[]
        indices=[]
        for layer in reversed(list(self.model.keys())):
            if classifierPresent and layer==list(self.model.keys())[-1]:#classifier
                continue
                
            try:
                name=self.model[layer][0].__class__.__name__
            except:
                name='MaxPool'
                
            if (name=='Dropout' or name=='Identity'):
                continue
                
            chain.append(name)
            indices.append(layer)
            
            if chain==['MaxPool']:
                self.add_module('LRPMaxPool'+indices[-1],
                                LRPMaxPool(self.model[indices[0]]['maxpool'][0],e,rule))
                chain=[]
                indices=[]
            elif chain==['AvgPool2d']:
                self.add_module('LRPAvgPool'+indices[-1],
                                LRPPool2d(self.model[indices[0]][0],e,rule))
                chain=[]
                indices=[]
            elif chain==['AvgPool2d']:
                self.add_module('LRPAdpAvgPool'+indices[-1],
                                LRPAdaptativePool2d(self.model[indices[0]][0],e,rule))
                chain=[]
                indices=[]
            elif chain==['Flatten']:
                if indices==['0']:
                    size=inputShape
                else:
                    size=preFlattenShape
                self.add_module('Unflatten'+indices[-1],
                                torch.nn.Unflatten(dim=-1, unflattened_size=size))
                chain=[]
                indices=[]
            
            elif ((indices[-1]=='0' or indices[-1]=='1') and Zb):
                if list(reversed(chain))==['Linear','ReLU']:
                    self.add_module('LRPZbDense'+indices[-1],
                                    ZbRuleDenseInput(self.model[indices[-1]][0],e,l=0,h=1))
                    chain=[]
                    indices=[] 
                elif list(reversed(chain))==['Conv2d','ReLU']:
                    self.add_module('LRPZbConv'+indices[-1],
                                    ZbRuleConvInput(self.model[indices[-1]][0],e,l=0,h=1))
                    chain=[]
                    indices=[] 
                    
                elif list(reversed(chain))==['Conv2d','BatchNorm2d','ReLU']:
                    self.add_module('LRPZbConvBN'+indices[-1],
                                    ZbRuleConvBNInput(self.model[indices[-1]][0],
                                                      self.model[indices[-2]][0],
                                                      e,l=0,h=1))
                    chain=[]
                    indices=[] 
                
                
            elif list(reversed(chain))==['Linear','ReLU']:
                self.add_module('LRPDense'+indices[-1],
                                LRPDenseReLU(self.model[indices[-1]][0],e,rule))
                chain=[]
                indices=[]
                
            elif list(reversed(chain))==['Conv2d','ReLU']:
                self.add_module('LRPConv'+indices[-1],
                                LRPConvReLU(self.model[indices[-1]][0],e,rule))
                chain=[]
                indices=[]
                
            elif list(reversed(chain))==['Conv2d','BatchNorm2d','ReLU']:
                self.add_module('LRPConvBN'+indices[-1],
                                LRPConvBNReLU(self.model[indices[-1]][0],
                                              self.model[indices[-2]][0],
                                              e,rule))
                chain=[]
                indices=[]
                
        if len(chain)>0:
            raise ValueError('Unrecognized sequence:',list(reversed(chain)))
                
        
                
        #register hooks to store relevance
        if storeRelevance:
            self.storedRelevance={}
            for name,layer in self.items():
                if ('LRP' in name):
                    self.storedRelevance[name]=f.HookRelevance(module)
                    
    def forward(self,x,y,R=None,label=None):
        #y=self.model[list(self.model.keys())[-1]][1].output
        B=y.shape[0]#batch size
        
        if self.rule=='z+e':
            C=2
        elif self.multiple:
            C=y.shape[-1]
        elif not self.multiple:
            C=1
        else:
            raise ValueError('Invalid argument selection')
            
        if self.classifierPresent:
            if self.selective:
                R=self.LRPSelectiveOut(aJ=self.model[list(self.model.keys())[-1]][1].input,aK=y,
                                       label=label)
            else:
                R=self.LRPOut(aJ=self.model[list(self.model.keys())[-1]][1].input,y=y,label=label)
                R=self.LRPOutDense(rK=R,aJ=self.model[list(self.model.keys())[-1]][1].input,aK=y)

        for name,layer in self.items():
            if 'Unflatten' in name:
                R=layer(R)
            if 'LRPMaxPool' in name:
                R=layer(rK=R,
                         aJ=self.model[str(re.findall(r'\d+',name)[-1])]['maxpool'][1].input, 
                         aK=self.model[str(re.findall(r'\d+',name)[-1])]['maxpool'][1].output)
            if ('LRP' in name and 'Out' not in name and 'MaxPool' not in name\
               and 'BN' not in name):
                R=layer(rK=R,
                         aJ=self.model[str(re.findall(r'\d+',name)[-1])][1].input, 
                         aK=self.model[str(re.findall(r'\d+',name)[-1])][1].output)
            elif 'BN'in name:
                R=layer(rK=R,
                         aJ=self.model[str(re.findall(r'\d+',name)[-1])][1].input,
                         aKConv=self.model[str(re.findall(r'\d+',name)[-1])][1].output,
                         aK=self.model[str(int(re.findall(r'\d+',name)[-1])+1)][1].output)
        return R
                    
class LRPSequential (nn.Module):
    def __init__(self,network,heat,e,Zb,rule,selective,highest,multiple,amplify=1,detach=True, 
                 inputShape=None,preFlattenShape=None,randomLogit=False):
        super(LRPSequential ,self).__init__()
        
        #remove inplace
        f.ChangeInplace(network) #no need, using .clone() in hooks
        
        self.NN=network
        self.LRPBlock=_LRPSequential(self.NN,e,Zb,rule,selective,highest,
                                     multiple,amplify=amplify,detach=detach,
                                     inputShape=inputShape,preFlattenShape=preFlattenShape,
                                     randomLogit=randomLogit)
        
        self.heat=heat
        
    def forward(self,x):
        y=self.NN(x)
        
        if not self.heat:
            return y
        
        R=self.LRPBlock(x,y)
        return y,R
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class _LRPVGG (nn.Module):
    def __init__(self,network,e,Zb,rule,selective,highest,multiple,
                 detach=True,storeRelevance=False,amplify=1, inputShape=None,
                 preFlattenShape=None,randomLogit=False):
        #LRP Block: Propagates relevance through VGG
        #network: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #rule: LRP rule, e or z+e
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: set to true for selective ISNet
        #amplify: multiplier to LRP heatmap
        #multiple: multiple heatmaps per sample, use for Original ISNet
        #detach: detach biases from graph when propagating relevances
        #storeRelevance: stores intermediate heatmaps for  LDS
        #randomLogit: Stochastic ISNet
        #preFlattenShape: sginal shape before flatten, not needed for VGG
        #inputShape: input shape
        
        super(_LRPVGG ,self).__init__()
        
        if (randomLogit and selective):
            raise ValueError('Set randomLogit or selective, not both')
        
        self.LRPClassifier=_LRPSequential(network.classifier,e,Zb=False,rule=rule,
                                        selective=selective,highest=highest,
                                     multiple=multiple,amplify=amplify,detach=detach,
                                     inputShape=inputShape,preFlattenShape=preFlattenShape,
                                       classifierPresent=True,randomLogit=randomLogit)
        #self.num_features=network.classifier[0].in_features
        self.Pool=f.InsertIO(network.avgpool)
        #print('pool model:',self.Pool)
        self.LRPPool=LRPAdaptativePool2d(self.Pool[0],e,rule)
        self.LRPFeatures=_LRPSequential(network.features,e,Zb,rule=rule,
                                        selective=selective,highest=highest,
                                     multiple=multiple,amplify=amplify,detach=detach,
                                     inputShape=inputShape,preFlattenShape=preFlattenShape,
                                       classifierPresent=False)
        #print(self.LRPFeatures)
        print(self.LRPClassifier)
        
        if storeRelevance:
            #store the relevances after each MaxPool
            self.storedRelevance={}
            self.storedRelevance['LRPFinalPool']=f.HookRelevance(self.LRPPool)
            names=[]
            for name, module in self.LRPFeatures.named_modules():
                if ('.' not in name and name!=''):
                    names.append(name)
            save=[]
            for i,name in enumerate(names,0):
                #we want the relevance at the input of the layer after the maxpool
                #ignore maxpool right before classifier
                if 'LRPMaxPool' in name and i!=0:
                    #print(name,names[i-1])
                    save.append(names[i-1])
            for name, module in self.LRPFeatures.named_modules():
                if (name in save):
                    self.storedRelevance[name]=f.HookRelevance(module)
                    
        
    def forward(self,x,y,label=None):
        R=self.LRPClassifier(x,y,label)
        
        B=y.shape[0]
        if self.LRPClassifier.rule=='z+e':
            C=2
        elif self.LRPClassifier.multiple:
            C=y.shape[-1]
        elif not self.LRPClassifier.multiple:
            C=1
        else:
            raise ValueError('Invalid argument selection')
            
        R=R.view(B,C,self.Pool[1].output.shape[-3],self.Pool[1].output.shape[-2],
                 self.Pool[1].output.shape[-1])
        
        #print(R.shape)
        #print(self.Pool[1].input.shape,self.Pool[1].output.shape)
        R=self.LRPPool(R,
                       aJ=self.Pool[1].input,
                       aK=self.Pool[1].output)
        
        R=self.LRPFeatures(x,y,R=R)
        
        return R
    
class LRPVGG (nn.Module):
    def __init__(self,network,heat,e,Zb,rule,selective,highest,multiple,amplify=1,detach=True, 
                 inputShape=None,preFlattenShape=None,randomLogit=False):
        #LRP Block: Propagates relevance through VGG
        #network: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #rule: LRP rule, e or z+e
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #highest: set to true for selective ISNet
        #amplify: multiplier to LRP heatmap
        #multiple: multiple heatmaps per sample, use for Original ISNet
        #detach: detach biases from graph when propagating relevances
        #randomLogit: Stochastic ISNet
        #preFlattenShape: sginal shape before flatten, not needed for VGG
        #inputShape: input shape
        super(LRPVGG ,self).__init__()
        
        #remove inplace
        f.ChangeInplace(network) #no need, using .clone() in hooks
        
        self.NN=network
        self.LRPBlock=_LRPVGG(self.NN,e,Zb,rule,selective,highest,
                             multiple,amplify=amplify,detach=detach,
                             inputShape=inputShape,preFlattenShape=preFlattenShape,
                             randomLogit=randomLogit)
        
        self.heat=heat
        
    def forward(self,x):
        y=self.NN(x)
        
        if not self.heat:
            return y
        
        R=self.LRPBlock(x,y)
        return y,R
        
        
        
        

