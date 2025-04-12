import torch
import torch.nn as nn
import ISNetFunctions as f
import LRPDenseNet
import globals
from collections import OrderedDict
import re

#Transform function in ISNetFunctions in DNN layers:

class LRPDenseReLU (nn.Module):
    def __init__(self, layer,e):
        super().__init__()
        self.e=e
        self.layer=layer
        
        
    def forward(self, rK, aJ, aK):
        y=f.LRPDenseReLU(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e)
        return y
    
class LRPConvReLU (nn.Module):
    def __init__(self, layer,e):
        super().__init__()
        self.e=e
        self.layer=layer
        
    def forward(self, rK, aJ, aK):
        y=f.LRPConvReLU(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e)
        return y
    
class LRPOutput (nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y):
        return f.LRPOutput(y=y)

class LRPPool2d (nn.Module):
    def __init__(self,layer,e):
        super().__init__()
        self.e=e
        self.layer=layer
        
    def forward(self, rK, aJ, aK):
        y= f.LRPPool2d(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e)
        return y
    
class LRPAdaptativePool2d (nn.Module):
    def __init__(self,layer,e):
        super().__init__()
        self.e=e
        self.layer=layer
        
    def forward(self, rK, aJ,aK):
        y= f.LRPPool2d(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e,
                       adaptative=True)
        return y
    
class LRPSum (nn.Module):
    def __init__(self,e):
        super().__init__()
        self.e=e
    def forward(self,rK,aJ,aK):
        y= f.LRPSum(rK=rK,aJ=aJ,aK=aK,e=self.e)
        return y
    
class LRPPool2dBNReLU (nn.Module):
    def __init__(self,layer,BN,e,Ch0=0,Ch1=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        

    def forward (self,rK, aJ, aKConv):
        y= f.LRPPool2dBNReLU(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,e=self.e,
                                 aKConv=aKConv,
                                 Ch0=self.Ch[0],Ch1=self.Ch[1])
        return y

class MultiBlockPoolBNReLU (nn.Module):
    def __init__(self,layer,BN,e,Ch0=0,Ch1=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        
    def forward (self, rK, aJ, aK, aKPool):
        y= f.MultiBlockPoolBNReLU(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,e=self.e,
                                  aKPool=aKPool,
                                  Ch0=self.Ch[0],Ch1=self.Ch[1])
        return y
    
class MultiBlockMaxPoolBNReLU (nn.Module):
    def __init__(self,layer,BN,e,Ch0=0,Ch1=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        
    def forward (self, rK, aJ, aK, aKPool):
        y= f.MultiBlockMaxPoolBNReLU(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,e=self.e,
                                      aKPool=aKPool,
                                      Ch0=self.Ch[0],Ch1=self.Ch[1])
        return y
    
class MultiLayerConvBNReLU (nn.Module):
    def __init__(self,layer,BN,e,Ch0=0,Ch1=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        
    def forward (self,rK,aJ,aKConv,aK):                         
        y= f.MultiLayerConvBNReLU(layer=self.layer,BN=self.BN,rK=rK,aK=aK,aJ=aJ,e=self.e,
                                      aKConv=aKConv,
                                      Ch0=self.Ch[0],Ch1=self.Ch[1])
        return y
    
class LRPConvBNReLU (nn.Module):
    def __init__(self,layer,BN,e,Ch0=0,Ch1=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        
    def forward (self,rK,aJ,aKConv,aK):         
        y= f.LRPConvBNReLU(layer=self.layer,BN=self.BN,rK=rK,aK=aK,aJ=aJ,e=self.e,
                               aKConv=aKConv,
                               Ch0=self.Ch[0],Ch1=self.Ch[1])
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
    
class LRPMaxPool (nn.Module):
    def __init__(self,layer,e):
        super().__init__()
        self.e=e
        self.layer=layer
    def forward (self,rK,aJ,aK):
        y= f.LRPMaxPool2d(layer=self.layer,rK=rK,aJ=aJ,aK=aK,e=self.e)
        return y
    
class LRPBNConvReLU (nn.Module):
    def __init__(self,layer,BN,e,Ch0=0,Ch1=None):
        super().__init__()
        self.e=e
        self.layer=layer
        self.BN=BN
        self.Ch=[Ch0,Ch1]
        
    def forward (self,rK,aJ,aK):         
        y= f.LRPBNConvReLU(layer=self.layer,BN=self.BN,rK=rK,aK=aK,
                           aJ=aJ,e=self.e,
                           Ch0=self.Ch[0],Ch1=self.Ch[1])
        return y
    
#DenseNet specific modules:
    
class LRPDenseLayer (nn.Module):
    def __init__(self,DenseLayerIdx,DenseBlockIdx,model,e):
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
                                     BN=BNs,e=e,Ch0=Ch0,Ch1=Ch1)
        #through conv1, norm2 and relu2:
        self.L1=LRPConvBNReLU(layer=self.DenseLayer['conv1'][0],BN=self.DenseLayer['norm2'][0],
                                 e=e)
        
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
    def __init__(self,DenseBlockIdx,model,e):
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
                                model=model,e=e)
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
    def __init__(self,model,e):
        #propagates relevance through the DenseNet last layers (after dense blocks)
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e term
        
        super(LRPDenseNetClassifier,self).__init__()
        
        #initialize output LRP layer:
        self.Out=LRPOutput()
        #dense layer:
        try:
            classifier=model['classifier'][0]
        except:
            #model with dropout
            classifier=model['classifier']['1'][0]
        #num_features: number of channels after adaptative pool
        self.num_features=classifier.in_features
        #initialize LRP layer for classifier:
        self.Dense=LRPDenseReLU(classifier,e)
        #pooling layer:
        pool=model['features']['AdaptPool'][0]
        self.PoolInput=model['features']['fReLU'][1]
        self.PoolOutput=model['features']['AdaptPool'][1]
        #initialize LRP layer for pooling:
        self.AdpPool=LRPAdaptativePool2d(pool,e=e)
        
    def forward(self,X,XI,y):
        #X:list of all layer activations 
        #XI: inputs of last layer
        #y: classifier outputs
        
        B=y.shape[0]#batch size
        C=y.shape[-1]#classes
        #y:DNN output
        R=self.Out(y)
        R=self.Dense(rK=R,aJ=XI[0],aK=y)
        R=R.view(B,C,self.num_features,1,1)
        R=self.AdpPool(rK=R,aJ=X[self.PoolInput],aK=X[self.PoolOutput])
        return R
        
class LRPDenseNetInitial(nn.Module):
    def __init__(self,model,e,Zb=True):
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
                                       e=e)
        if(Zb):
            self.conv=ZbRuleConvBNInput(layer=model['features']['conv0'][0],
                                        BN=model['features']['norm0'][0],
                                        e=e,l=0,h=1)
        else:
            self.conv=LRPConvBNReLU(layer=model['features']['conv0'][0],
                                           BN=model['features']['norm0'][0],
                                           e=e)
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
    def __init__(self,TransitionIdx,model,e):
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
                                       e=e)
        self.conv=LRPConvReLU(layer=model['features'][self.current]['conv'][0],
                                e=e)
        
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
    def __init__(self,DenseNet,e,Zb=True):
        #LRP Block: Propagates relevance through DenseNet
        #DenseNet: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        
        super(_LRPDenseNet,self).__init__()
        
        if(not torch.backends.cudnn.deterministic):
            raise Exception('Please set torch.backends.cudnn.deterministic=True') 
        
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
        self.add_module('LRPFinalLayers',
                        LRPDenseNetClassifier(model=model,e=e))
        
        #DenseBlocks and transitions:
        for key in reversed(list(model['features'])):
            if ('denseblock' in key):
                block=int(key[-1])
                self.add_module('LRPDenseBlock%d' % (block),
                                LRPDenseBlock(block,model=model,e=e))
            if ('transition' in key):
                trans=int(key[-1])
                self.add_module('LRPTransition%d' % (trans),
                                LRPDenseNetTransition(trans,model=model,e=e))
        
        #initial layers:
        self.add_module('LRPInitialLayers',
                        LRPDenseNetInitial(model=model,e=e,Zb=Zb))
        
    def forward(self,x,y):
        #x: model input
        #y: classifier output
        
        R=self.LRPFinalLayers(X=globals.X,XI=globals.XI,y=y)
        for name,layer in self.items():
            if ((name != 'LRPFinalLayers') and (name != 'LRPInitialLayers')):
                R=layer(X=globals.X,r=R)
        R=self.LRPInitialLayers(X=globals.X,x=x,r=R)

        #clean global variables:
        globals.X=[]
        globals.XI=[]
        
        return R

class IsDense(nn.Module):
    def __init__(self,DenseNet,e=1e-2,heat=True,
                 Zb=True):
        #Creates ISNet based on DenseNet
        #DenseNet: classifier
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #heat: allows relevance propagation and heatmap creation. If False,
        #no signal is propagated through LRP block.
        
        super (IsDense,self).__init__()
        self.DenseNet=DenseNet
        self.LRPBlock=_LRPDenseNet(self.DenseNet,e=e,Zb=Zb)
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

        
        
        
        
        
        
#functions for standard PyTorch sequential blocks and simple sequences of layers

class _LRPSequential (nn.ModuleDict):
    def __init__(self,network,e,Zb,
                 inputShape=None,
                 preFlattenShape=None,classifierPresent=True):
        
        super(_LRPSequential ,self).__init__()
        if(not torch.backends.cudnn.deterministic):
            raise ValueError('Please set torch.backends.cudnn.deterministic=True')
        
        self.classifierPresent=classifierPresent
        
        for i,layer in enumerate(network,0):
            name=layer.__class__.__name__
            if name=='MaxPool2d':
                network[i].return_indices=True
                network[i]=nn.Sequential(OrderedDict([('maxpool',network[i]),
                                                      ('special',IgnoreIndexes())]))
        
        #register hooks:
        self.model=f.InsertIO(network)
        
        if classifierPresent:
            #last layer:
            classifier=self.model[list(self.model.keys())[-1]][0]
           
            self.add_module('LRPOut',
                            LRPOutput())
            #initialize LRP layer for classifier:
            self.add_module('LRPOutDense',
                            LRPDenseReLU(classifier,e))
        
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
                                LRPMaxPool(self.model[indices[0]]['maxpool'][0],e))
                chain=[]
                indices=[]
            elif chain==['AvgPool2d']:
                self.add_module('LRPAvgPool'+indices[-1],
                                LRPPool2d(self.model[indices[0]][0],e))
                chain=[]
                indices=[]
            elif chain==['AvgPool2d']:
                self.add_module('LRPAdpAvgPool'+indices[-1],
                                LRPAdaptativePool2d(self.model[indices[0]][0],e))
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
                                LRPDenseReLU(self.model[indices[-1]][0],e))
                chain=[]
                indices=[]
                
            elif list(reversed(chain))==['Conv2d','ReLU']:
                self.add_module('LRPConv'+indices[-1],
                                LRPConvReLU(self.model[indices[-1]][0],e))
                chain=[]
                indices=[]
                
            elif list(reversed(chain))==['Conv2d','BatchNorm2d','ReLU']:
                self.add_module('LRPConvBN'+indices[-1],
                                LRPConvBNReLU(self.model[indices[-1]][0],
                                              self.model[indices[-2]][0],
                                              e))
                chain=[]
                indices=[]
                
        if len(chain)>0:
            raise ValueError('Unrecognized sequence:',list(reversed(chain)))

                    
    def forward(self,x,y,R=None):
        B=y.shape[0]#batch size
        C=y.shape[-1]
            
        if self.classifierPresent:
            R=self.LRPOut(y=y)
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
    def __init__(self,network,heat,e,Zb, 
                 inputShape=None,preFlattenShape=None):
        super(LRPSequential ,self).__init__()
        
        #remove inplace
        f.ChangeInplace(network)
        
        self.NN=network
        self.LRPBlock=_LRPSequential(self.NN,e,Zb,
                                     inputShape=inputShape,preFlattenShape=preFlattenShape)
        
        self.heat=heat
        
    def forward(self,x):
        y=self.NN(x)
        
        if not self.heat:
            return y
        
        R=self.LRPBlock(x,y)
        return y,R
            
        
        
        
        
        
        
        
        
        
        
        
        
        
# Functions for ISNets with VGG backbones        
        
class _LRPVGG (nn.Module):
    def __init__(self,network,e,Zb,inputShape=None,
                 preFlattenShape=None):
        #LRP block for VGG network
        super(_LRPVGG ,self).__init__()
        
        
        self.LRPClassifier=_LRPSequential(network.classifier,e,Zb=False,
                                     inputShape=inputShape,preFlattenShape=preFlattenShape,
                                       classifierPresent=True)
        self.Pool=f.InsertIO(network.avgpool)
        self.LRPPool=LRPAdaptativePool2d(self.Pool[0],e)
        self.LRPFeatures=_LRPSequential(network.features,e,Zb,
                                     inputShape=inputShape,preFlattenShape=preFlattenShape,
                                       classifierPresent=False)
        
    def forward(self,x,y):
        R=self.LRPClassifier(x,y)
        
        B=y.shape[0]
        C=y.shape[-1]
        
        R=R.view(B,C,self.Pool[1].output.shape[-3],self.Pool[1].output.shape[-2],
                 self.Pool[1].output.shape[-1])
        
        
        R=self.LRPPool(R,
                       aJ=self.Pool[1].input,
                       aK=self.Pool[1].output)
        
        R=self.LRPFeatures(x,y,R=R)
        
        return R
    
class LRPVGG (nn.Module):
    def __init__(self,network,heat,e,Zb, 
                 inputShape=None,preFlattenShape=None):
        #VGG-based ISNet, network must be torchvision.models.vgg...
        super(LRPVGG ,self).__init__()
        
        #remove inplace
        f.ChangeInplace(network) #no need, using .clone() in hooks
        
        self.NN=network
        self.LRPBlock=_LRPVGG(self.NN,e,Zb,
                             inputShape=inputShape,preFlattenShape=preFlattenShape)
        
        self.heat=heat
        
    def forward(self,x):
        y=self.NN(x)
        
        if not self.heat:
            return y
        
        R=self.LRPBlock(x,y)
        return y,R
    
class IgnoreIndexes(nn.Module):   
    #special layer to ignore indexes of previous max pooling layer
    def __init__(self):
        super(IgnoreIndexes, self).__init__()
    def forward(self, x):
        return(x[0])