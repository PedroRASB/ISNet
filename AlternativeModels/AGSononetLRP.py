#perform LRP thourgh AG-Sononet
import torch
import torch.nn as nn
import ISNetLayers as LRP
import ISNetFunctions as f
import globals

class LRPFinalLayers(nn.Module):
    def __init__(self,model,e):
        #perform LRP through last layers of AG-sononet
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e e value        
        super(LRPFinalLayers, self).__init__()
        self.Out=LRP.LRPOutput()
        self.Dense=LRP.LRPDenseReLU(layer=model['classifier'][0],e=e)
        self.Sum2=LRP.LRPSum(e=e)
        self.Sum1=LRP.LRPSum(e=e)
        self.GlobalPool=LRP.LRPAdaptativePool2d(layer=model['adpPool'][0],e=e)
        self.GlobalPoolInput=model['conv5']['conv2']['2'][1]
        self.GlobalPoolOutput=model['adpPool'][1]
        self.AttGateSum2Out=model['gateSum2'][1]
        self.AttGateSum2In=model['gate2'][1]
        self.AttGateSum1Out=model['gateSum1'][1]
        self.AttGateSum1In=model['gate1'][1]
        
    def forward(self,y,X,XI):
        r=self.Out(y)
        r=self.Dense(rK=r,aJ=XI[0],aK=y)
        
        rOriginal=self.GlobalPool(rK=r[:,:,X[self.AttGateSum1In].shape[1]+\
                                       X[self.AttGateSum2In].shape[1]:]\
                                  .unsqueeze(-1).unsqueeze(-1),
                                  aJ=X[self.GlobalPoolInput],
                                  aK=X[self.GlobalPoolOutput])
        #relevance at output of attention blocks:
        rAtt2=self.Sum2(rK=r[:,:,X[self.AttGateSum1In].shape[1]:\
                            X[self.AttGateSum1In].shape[1]+\
                            X[self.AttGateSum2In].shape[1]],
                        aJ=X[self.AttGateSum2In],aK=X[self.AttGateSum2Out])
        
        rAtt1=self.Sum1(rK=r[:,:,:X[self.AttGateSum1In].shape[1]],
                        aJ=X[self.AttGateSum1In],aK=X[self.AttGateSum1Out])
        return rOriginal, rAtt2, rAtt1
        
class LRPConvBlock(nn.Module):
    def __init__(self,model,blockIdx,e,first=False):
        #Perform LRP though convolutional block
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e e value
        #blockIdx: identifies the block
        #first: identifies the first block        
        super(LRPConvBlock, self).__init__()
        self.block=model['conv'+str(blockIdx)]
        if (len(self.block)==3):
            self.Conv3=LRP.LRPConvBNReLU(layer=self.block['conv3']['0'][0],
                                         BN=self.block['conv3']['1'][0],e=e)
        self.Conv2=LRP.LRPConvBNReLU(layer=self.block['conv2']['0'][0],
                                         BN=self.block['conv2']['1'][0],e=e)
        self.first=first
        if (not first):
            self.blockInput=model['ignore'+str(blockIdx-1)][1]
            self.Conv1=LRP.LRPConvBNReLU(layer=self.block['conv1']['0'][0],
                                         BN=self.block['conv1']['1'][0],e=e)
        else:
            self.Conv1=LRP.ZbRuleConvBNInput(layer=self.block['conv1']['0'][0],
                                         BN=self.block['conv1']['1'][0],e=e)
    def forward(self,r,X,x=None):
        if (len(self.block)==3):
            r=self.Conv3(rK=r,aJ=X[self.block['conv2']['2'][1]],
                         aKConv=X[self.block['conv3']['0'][1]],
                         aK=X[self.block['conv3']['1'][1]])
        r=self.Conv2(rK=r,aJ=X[self.block['conv1']['2'][1]],
                         aKConv=X[self.block['conv2']['0'][1]],
                         aK=X[self.block['conv2']['1'][1]])
        
        if(self.first):#first block
            BlkIn=x
        else:
            BlkIn=X[self.blockInput]
            
        r=self.Conv1(rK=r,aJ=BlkIn,
                         aKConv=X[self.block['conv1']['0'][1]],
                         aK=X[self.block['conv1']['1'][1]])
        return r
    
class LRPPool(nn.Module):
    def __init__(self,model,poolIdx,e):
        #Perform LRP though max-pool
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e e value
        #poolIdx: identifies the pooling layer
        super(LRPPool, self).__init__()
        self.pool=LRP.LRPMaxPool(layer=model['maxpool'+str(poolIdx)][0],e=e)
        self.poolIn=model['conv'+str(poolIdx)]\
        [list(model['conv'+str(poolIdx)].keys())[-1]]['2'][1]
        self.poolOut=model['maxpool'+str(poolIdx)][1]
    def forward(self,r,X):
        r=self.pool(rK=r,aJ=X[self.poolIn],aK=X[self.poolOut])
        return r
        
        
class _LRPAGSononet(nn.ModuleDict):
    def __init__(self,AGSononet,e):
        #Perform LRP though AG-Sononet
        #e: LRP-e e value
        #AGSononet: AG-Sononet
        super(_LRPAGSononet, self).__init__()
        #register hooks:
        f.resetGlobals()
        model=f.InsertHooks(AGSononet)
        AGSononet.classifier.register_forward_hook(f.AppendInput)
        
        #classifier and global pooling/sum:
        self.add_module('LRPFinal',LRPFinalLayers(model=model, e=e))
        
        #convolutional blocks and maxpool:
        for key in reversed(list(model)):
            if('conv' in key):
                if(key[-1]!='1'):
                    self.add_module('LRP'+key,LRPConvBlock(model=model,blockIdx=int(key[-1]),e=e,
                                                               first=False))
                else:
                    self.add_module('LRP'+key,LRPConvBlock(model=model,blockIdx=int(key[-1]),e=e,
                                                               first=True))
            if('maxpool' in key):
                self.add_module('LRP'+key,LRPPool(model=model,poolIdx=int(key[-1]),e=e))
                
    def forward(self,x,y,a1,a2):
        r,rAtt2,rAtt1=self.LRPFinal(y=y,X=globals.X,XI=globals.XI)
        r=self.LRPconv5(r=r,X=globals.X)
        r=self.LRPmaxpool4(r=r,X=globals.X)
        r=rAtt2+r #connection to attention gate 2
        r=self.LRPconv4(r=r,X=globals.X)
        r=self.LRPmaxpool3(r=r,X=globals.X)
        r=rAtt1+r #connection to attention gate 1
        r=self.LRPconv3(r=r,X=globals.X)
        r=self.LRPmaxpool2(r=r,X=globals.X)
        r=self.LRPconv2(r=r,X=globals.X)
        r=self.LRPmaxpool1(r=r,X=globals.X)
        r=self.LRPconv1(r=r,X=globals.X,x=x)
        return r

class LRPAGSononet(nn.Module):
    def __init__(self,AGSononet,e=1e-2,heat=True):
        #AG-Sononet+LRP block to create explanation heatmaps
        #attention: this model was not tested as an ISNet (i.e., minimizing heatmap loss)
        #heat: allows the creatiion of heatmaps
        #e: LRP-e e term
        #AGSononet: AG-Sononet
        super(LRPAGSononet, self).__init__()
        self.AGSononet=AGSononet
        f.ChangeInplace(self.AGSononet)
        self.LRPBlock=_LRPAGSononet(AGSononet=self.AGSononet,e=e)
        self.heat=heat

    def forward(self,x):
        #clean global variables:
        globals.X=[]
        globals.XI=[]
        
        if (not self.heat):
            self.AGSononet.returnCoef=False
            y=self.AGSononet(x)
            return y
        
        if (self.heat):
            self.AGSononet.returnCoef=True
            y,a1,a2=self.AGSononet(x)
            R=self.LRPBlock(x=x,y=y,a1=a1,a2=a2)
            return y,R
                          
                          