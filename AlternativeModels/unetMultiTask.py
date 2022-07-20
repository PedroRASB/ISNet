"""
Inspired by the U-Net implementation from https://github.com/jvanvugt/pytorch-unet,
released under the MIT License, copied below:

MIT License

Copyright (c) 2018 Joris

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
import ISNetLayers as LRP
import ISNetFunctions as f
import globals

class MTUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        ClassificationClasses=3,
        ReturnMasks=True,
        ClassifierFeatures=1024*7*7,
        bias=True
    ):
        """
        Multi-task U-Net
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            ClassificationClasses (int): number of output classes for segmentation head
            ReturnMasks (bool): if True, model returns classification and segmentation outputs, 
            otherwise, it returns only classification outputs
            ClassifierFeatures (int): number of elements before first 
                                      dense layer in classification head
            bias (bool): allows convolution bias
        """
        super(MTUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        
        #Down path:
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if(i!=depth-1):
                pool=True
            else:
                pool=False
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm,bias=bias,
                              max_pool=pool)
            )
            prev_channels = 2 ** (wf + i)

        #Classification head:
        self.classification_head = ClassificationHead(ClassifierFeatures=ClassifierFeatures,
                                                      ClassificationClasses=ClassificationClasses)
        
        #Up Path:
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        
        self.ReturnMasks=ReturnMasks
        self.batch_norm=batch_norm
        self.padding=padding
        
        

    def forward(self, x):
        #Down path:
        blocks = []
        for i, down in enumerate(self.down_path):
            if i != len(self.down_path) - 1:
                prePool,x=down(x)
                blocks.append(prePool)
            else:
                x=down(x)
                
        #Classification head:
        c=self.classification_head(x)
        
        if(self.ReturnMasks):
            #Up path
            for i, up in enumerate(self.up_path):
                x = up(x, blocks[-i - 1])
            return c,self.last(x)
        
        else:
            return c
        
class IgnoreIndexes(nn.Module):   
    #special layer to ignore indexes of previous max pooling layer
    def __init__(self):
        super(IgnoreIndexes, self).__init__()
    def forward(self, x):
        return(x[0])

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, max_pool,bias=True):
        #U-Net convolutional Block (contracting path)
        #in_size: input channels
        #out_size: output channels
        #padding (bool): convolutional padding, if true, concolutions preserve feature map dimensions
        #batch_norm (bool): includes batch normalization before ReLU if True
        #max_pool (bool): adds max pooling at the end of the block
        #bias (bool): allows bias in convolutions
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding),
                              bias=bias))
        if (batch_norm):
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding),
                              bias=bias))
        if (batch_norm):
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())
            
        self.block = nn.Sequential(*block)
        
        if max_pool:
            self.pool=nn.Sequential(nn.MaxPool2d(2,return_indices=True),
                                    IgnoreIndexes())

    def forward(self, x):
        out = self.block(x)
        if (hasattr(self,'pool')):
            pout=self.pool(out)
            return out,pout
        else:
            return out
    
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        #U-Net Block in expanding path
        #in_size: input channels
        #out_size: output channels
        #up_mode: one of 'upconv' or 'upsample'.
        #padding (bool): convolutional padding, if true, concolutions preserve feature map dimensions
        #batch_norm (bool): includes batch normalization before ReLU if True
        
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, max_pool=False)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
    
class ClassificationHead(nn.Module):
    def __init__(self,ClassifierFeatures,ClassificationClasses):
        #Classification head attached to the bottleneck feature representation
        #ClassifierFeatures: number of elements before first dense layer
        #ClassificationClasses: number of classes
        super(ClassificationHead, self).__init__()
        block=[]
        block.append(nn.AdaptiveAvgPool2d(output_size=(7, 7)))
        block.append(nn.Flatten())
        block.append(nn.Linear(in_features=ClassifierFeatures, 
                               out_features=4096, bias=True))
        block.append(nn.ReLU())
        block.append(nn.Dropout(p=0.5, inplace=False))
        block.append(nn.Linear(in_features=4096, 
                               out_features=4096, bias=True))
        block.append(nn.ReLU())
        block.append(nn.Dropout(p=0.5, inplace=False))
        block.append(nn.Linear(in_features=4096, 
                               out_features=ClassificationClasses, bias=True))
        self.block=nn.Sequential(*block)
        
    def forward(self, x):
        out=self.block(x)
        return out

    
class LRPHead1(nn.Module):
    def __init__(self,model,e):
        #Performs LRP through classification head
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e e value
        super(LRPHead1, self).__init__()
        self.Out=LRP.LRPOutput()
        head=model['classification_head']
        self.Dense3=LRP.LRPDenseReLU(layer=head['block']['8'][0],e=e)
        self.Dense2=LRP.LRPDenseReLU(layer=head['block']['5'][0],e=e)
        self.Dense1=LRP.LRPDenseReLU(layer=head['block']['2'][0],e=e)
        self.AvgPool=LRP.LRPAdaptativePool2d(layer=head['block']['0'][0],e=e)
        self.head=head
        lastBlock=model['down_path'][list(model['down_path'].keys())[-1]]['block']
        self.HeadInput=lastBlock[list(lastBlock.keys())[-1]][1]
        self.num_features=lastBlock['4'][0].num_features
        
    def forward(self,X,XI,y):
        B=y.shape[0]#batch size
        C=y.shape[-1]#classes
        #y:DNN output
        R=self.Out(y)
        R=self.Dense3(rK=R,aJ=XI[0],aK=y)
        R=self.Dense2(rK=R,aJ=X[self.head['block']['4'][1]],aK=X[self.head['block']['5'][1]])
        R=self.Dense1(rK=R,aJ=X[self.head['block']['1'][1]],aK=X[self.head['block']['2'][1]])
        R=R.view(B,C,self.num_features,7,7)
        R=self.AvgPool(rK=R,aJ=X[self.HeadInput],aK=X[self.head['block']['0'][1]])
        return R
    
class LRPConvBlockBN(nn.Module):
    def __init__(self,model,blockIdx,e):
        #Performs LRP through U-Net convolutional block (contracting path)
        #model: dictionary with all layers and fdw indexes (net.layers)
        #e: LRP-e e value
        #blockIdx: identifies the block
        super(LRPConvBlockBN, self).__init__()
        
        self.block=model['down_path'][str(blockIdx)]
        
        if ('pool' in self.block):
            self.Pool=LRP.LRPMaxPool(layer=self.block['pool']['0'][0],
                                     e=e)
            
        self.Conv2=LRP.LRPConvBNReLU(layer=self.block['block']['3'][0],
                                     BN=self.block['block']['4'][0],e=e)
        if (blockIdx>0):
            self.Conv1=LRP.LRPConvBNReLU(layer=self.block['block']['0'][0],
                                         BN=self.block['block']['1'][0],e=e)
            self.FirstBlock=False
            self.Conv1Input=model['down_path'][str(blockIdx-1)]\
                            ['pool']['1'][1]
        else:
            self.Conv1=LRP.ZbRuleConvBNInput(layer=self.block['block']['0'][0],
                                             BN=self.block['block']['1'][0],e=e,
                                             op=0)
            self.FirstBlock=True
            
        
    
    def forward(self,R,X,x):
        if (hasattr(self,'Pool')):
            R=self.Pool(rK=R,aJ=X[self.block['block']['5'][1]],aK=X[self.block['pool']['0'][1]])
            
        R=self.Conv2(rK=R,aJ=X[self.block['block']['2'][1]],aK=X[self.block['block']['4'][1]],
                     aKConv=X[self.block['block']['3'][1]])
        
        if(not self.FirstBlock):
            R=self.Conv1(rK=R,aJ=X[self.Conv1Input],aK=X[self.block['block']['1'][1]],
                         aKConv=X[self.block['block']['0'][1]])
        else:
            R=self.Conv1(rK=R,aJ=x,aK=X[self.block['block']['1'][1]],
                         aKConv=X[self.block['block']['0'][1]])
        return R
    
class LRPUNetClassifier(nn.ModuleDict):
    def __init__(self,MTUNet,e):
        #Performs LRP through the multi-task U-Net classification path
        #e: LRP-e e term
        #MTUNet: Multi-task U-Net
        super(LRPUNetClassifier, self).__init__()
        
        if(not torch.backends.cudnn.deterministic):
            raise Exception('Please set torch.backends.cudnn.deterministic=True') 
        
        #register hooks:
        f.resetGlobals()
        model=f.InsertHooks(MTUNet)
        #print(model)
        MTUNet.classification_head.block[-1].register_forward_hook(f.AppendInput)
        
        #output layers (classificaiton head):
        self.add_module('ClassificaitonHead',LRPHead1(model=model,e=e))
        for key in reversed(list(model['down_path'])):
            block=int(key[-1])
            self.add_module('LRPConvBlock%d' % (block),
                            LRPConvBlockBN(model=model,blockIdx=block,e=e))
            
    def forward(self,x,y):
        R=self.ClassificaitonHead(X=globals.X,XI=globals.XI,y=y)
        for name,layer in self.items():
            if ('LRPConvBlock' in name):
                R=layer(R=R,X=globals.X,x=x)
        return R

class LRPMTUNet(nn.Module):
    def __init__(self,MTUNet,e=1e-2,heat=True):
        #multi-task U-Net+LRP block to create explanation heatmaps
        #attention: this model was not tested as an ISNet (i.e., minimizing heatmap loss)
        #heat: allows the creatiion of heatmaps
        #e: LRP-e e term
        #MTUNet: Multi-task U-Net
        super(LRPMTUNet, self).__init__()
        if (not MTUNet.batch_norm and not MTUNet.padding):
                raise ValueError('LRP not implemented for no padding or no batch norm')
        self.MTUNet=MTUNet
        self.LRPBlock=LRPUNetClassifier(MTUNet=self.MTUNet,e=e)
        self.heat=heat
    def forward(self,x):
        #clean global variables:
        globals.X=[]
        globals.XI=[]
        if (self.MTUNet.ReturnMasks):
            y,masks=self.MTUNet(x)
            output=(y,masks)
        else:
            y=self.MTUNet(x)
            output=(y)
        if(self.heat):
            maps=self.LRPBlock(x=x,y=y)
            output=(*output,maps)
        #clean global variables:
        globals.X=[]
        globals.XI=[]
        
        return output
