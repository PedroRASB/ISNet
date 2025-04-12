import torch
import torch.nn.functional as F
import torch.nn as nn
import globalsZe as globals
import warnings
from collections import OrderedDict
from torch.autograd import Variable
import torch.autograd as ag
import warnings
import copy


class ISNetFlex(nn.Module):
    def __init__(self,model=None,architecture='densenet121',dropout=False,classes=1,
                 e=1e-2,Zb=True,multiple=True,
                 randomLogit=False,selective=False,selectiveEps=1e-2,
                 HiddenLayerPenalization=False,
                 pretrained=False,VGGRemoveLastMaxpool=False,
                 explainLabels=False):
        """
        Wrapper to apply LRP-Flex to arbitrary ReLU-based DNNs.
        
        Forward pass returns logits and LRP heatmaps.
        
        Args:
            model: arbitrary PyTorch DNN to be converted, not restricted to resnet densenet or VGG.
            If None, model is as pre-defined backbone, according to architecture parameter
            architecture: pre-defined backbone architecture name. Either densenets, resnets or vggs
            dropout: if True, adds dropout to pre-defined backbone
            classes: number of output neurons in pre-defined backbones
            e: LRP epsilon parameter
            Zb: if True, uses LRP-Zb in the first DNN layer
            multiple: if True, produces a single heatmap per sample (Faster ISNet). If False,
            creates one heatmap per class per sample (original ISNet)
            randomLogit: if True, stochastically selects a single logit to be explained (Stochastic
            ISNet)
            selective: if True, explains the softmax-based quantity Eta, instead of a logit
            (Selective ISNet)
            selectiveEps: epsilon used for stabilizing the relevance pass through softmax (see 
            Selective ISNet Section in paper)
            HiddenLayerPenalization: LRP Deep Supervision
            pretrained: if True, pre-defined backbone is downloaded pre-trained (ImageNet)
            VGGRemoveLastMaxpool: if True, removes last maxPool from VGG pre-defined backbone
            explainLabels: used for explanation only, not for ISNet training. If True, 
            heatmaps explain the classes indicated by the argument labels in the forward pass
        """
        super(ISNetFlex,self).__init__()
        if model is None:
            self.model=getBackbone(architecture,dropout,classes,pretrained=pretrained,
                                  VGGRemoveLastMaxpool=VGGRemoveLastMaxpool)
        else:
            self.model=model
            print('Using custom model. Ensure that ReLU layers are not re-utilized inside the model and avoid in-place operations')
        self.model=CleanNet(self.model)
        self.model=RemoveInplace(self.model,architecture)
        self.hooksModel=InsertReLUHooks(self.model)
        if HiddenLayerPenalization:
            self.HLPHooksFlex=LRPFlexStoreActivations(self.model)
        
        globals.LRP=False
        
        self.HLP=HiddenLayerPenalization
        self.randomLogit=randomLogit
        self.selective=selective
        self.selectiveEps=selectiveEps        
        self.multiple=multiple
        self.Zb=Zb
        globals.e=e
        self.explainLabels=explainLabels
        self.maximum=[]
        self.selected=[]
        
        tmp=0
        tmp2=0
        if self.selective:
            tmp+=1
        if self.randomLogit:
            tmp+=1
            tmp2+=1
        if self.multiple:
            tmp+=1
            tmp2+=1
        if self.explainLabels:
            tmp2+=1
        if tmp>1:
            print('selective, randomLogit and multiple:',selective,
                  randomLogit,multiple)
            raise ValueError('Choose multiple, selective OR randomLogit')
        if tmp2>1:
            print('explainLabels, randomLogit and multiple:',explainLabels,
                  randomLogit,multiple)
            raise ValueError('Choose explainLabels, selective OR randomLogit')
        
        if Zb:
            self.SetZbLayer()
            
        #print(self.model)
        
    def SetZbLayer(self):
        """
        Sets LRP-Zb model for DNN first layer.
        """
        self.firstLayer=[]
        flag=False
        for layer in self.model.children():
            if layer.__class__.__name__=='Sequential':
                for layer in layer.children():
                    self.firstLayer.append(layer)
                    if isinstance(layer, torch.nn.ReLU):
                        flag=True
                        break
            else:
                self.firstLayer.append(layer)
            if flag:
                break
            if isinstance(layer, torch.nn.ReLU):
                break
        
        chain=[]
        for module in self.firstLayer:
            chain.append(module.__class__.__name__)
            
        #LRP block for first layer:
        if (chain==['Conv2d','ReLU'] or chain==['Conv2d']):
            self.ZbModule=ZbRuleConvInput(self.firstLayer[0],l=0,h=1)
        elif chain==['Conv2d','BatchNorm2d','ReLU']:
            self.ZbModule=ZbRuleConvBNInput(self.firstLayer[0],
                                            self.firstLayer[1],
                                            l=0,h=1)
        elif chain==['Linear','ReLU']:
            self.ZbModule=ZbRuleDenseInput(self.firstLayer[0],l=0,h=1)
        else:
            #print(self.model)
            print(chain)
            raise ValueError ('Unrecognized first layer sequence for Zb rule')
        self.chain=chain
        #add hook to save first ReLU output, to get relevance after first layer:
        self.firstLayerOutput=SaveLRP(self.firstLayer[-1])
        self.firstLayerLinearOutput=SaveLRP(self.firstLayer[0])
        if 'BatchNorm2d' in self.chain:
            self.firstLayerBNOutput=SaveLRP(self.firstLayer[1])
        print(self.firstLayer)
        
                
    def forward(self,x,runLRPFlex=False,labels=None):
        """
        Forward pass.
        
        Args:
            x: input batch
            runLRPFlex: if True, produces LRP heatmap
            labels: optional, labels for x
            
        Return value:
            outputs dictionary with logits and LRP heatmap (if runLRPFlex is True)
        """
        outputs={'output':None,
                 'LRPFlex':None}
        
        if runLRPFlex:
            x=Variable(x, requires_grad=True)
        #run
        globals.LRP=runLRPFlex
        outputs['output']=self.model(x)
        globals.LRP=False
        
        if (self.training and outputs['output'].dtype!=torch.float32):
            raise ValueError('Support for half-precision not implemented, notice we manipulate the gradients and generate them inside forward')
            
        #get LRP Flex output
        if runLRPFlex:
            if not self.multiple:
                self.maximum=[]#shows for which batch elements we propagated the maximum logit
                if self.randomLogit:
                    indices=RandomLogitIndices(outputs['output'],maximum=self.maximum)
                    #print('self.maximum:',self.maximum)
                elif self.selective and not self.explainLabels:
                    indices=torch.argmin(outputs['output'],dim=-1)
                elif self.explainLabels:
                    #select label logit
                    if len(labels.shape)==1:
                        indices=labels
                    elif len(labels.shape)==2:
                        if labels.shape[-1]==1:
                            indices=labels.squeeze(-1)
                        else:
                            #multi-label problem
                            indices=torch.argmax(labels.float()+\
                                                 torch.normal(mean=0*labels.float(),
                                                              std=0.001),dim=-1)
                            #weak random noise is added to make argmax randomply select one of the 
                            #labels that are 1
                    else:
                        raise ValueError('Unrecognized label format')
                else:
                    raise ValueError('Unrecognized mode')
                self.selected=indices
                quantity=[]
                for i,val in enumerate(indices,0):
                    if self.selective:
                        quantity.append(torch.softmax(outputs['output'],dim=-1)[i,val])
                    else:
                        quantity.append(outputs['output'][i,val])
                quantity=torch.stack(quantity,0)
                if self.selective:
                    with torch.cuda.amp.autocast(enabled=False):
                        quantity=quantity.float()
                        #quantity=torch.clamp(quantity,min=1e-7,max=1-1e-7)
                        quantity=torch.log(quantity+self.selectiveEps)-\
                        torch.log(1-quantity+self.selectiveEps)
                quantity=quantity.sum(0)
                #sum because batch size should not change LRP maps
                LRP=self.LRPFlex(quantity,x,
                                 retain_graph=self.training,
                                 create_graph=self.training)#run lrp
                
            else:
                tmp=[]
                for i in list(range(outputs['output'].shape[-1])):
                    #print(i)
                    if self.selective:
                        quantity=torch.softmax(outputs['output'],dim=-1)[:,i]
                    else:
                        quantity=outputs['output'][:,i]#one logit at a time
                    if self.selective:
                        with torch.cuda.amp.autocast(enabled=False):
                            quantity=quantity.float()
                            #quantity=torch.clamp(quantity,min=1e-7,max=1-1e-7)
                            quantity=torch.log(quantity+self.selectiveEps)-\
                            torch.log(1-quantity+self.selectiveEps)
                    quantity=quantity.sum()#sum batch
                    #print(quantity)
                    tmp.append(self.LRPFlex(quantity,x,
                                            retain_graph=True,
                                            create_graph=self.training))
                    #print('multiple alement appended')
                LRP={}
                for key in tmp[0].keys():
                    LRP[key]=torch.cat([d[key] for d in tmp],dim=1)#concatenate in classes dimension
                #print(LRP['input'].shape)
            outputs['LRPFlex']=LRP
        #release memory by cleaning all values saved in hooks for LRP
        try:
            CleanHooks(self.model.LRPBlock.model) 
        except:
            pass
        
        CleanHooks(self.hooksModel) 
        
        #print(module,net.classifier.bias.data)
        
        return outputs
    
    def LRPFlex(self,quantity,x,retain_graph=True,create_graph=True):
        """
        Runs LRP-Flex, creating heatmaps.
        
        Args:
            quantity: quantity to be explained, e.g., logit
            x: input
            retain_graph: if True, retains graph, allowing future backward pass through DNN and
            the creation of other LRP-Flex heatmaps
            create_graph: if True, allows back-propagation through the LRP-Flex procedure
            
        Return value:
            LRP heatmaps
        """       
        if self.Zb:
            y0=self.firstLayerOutput.output
        else:
            y0=x
        
        #quantity: backpropagated quantity
        globals.LRP=(globals.e>0)
        #LRP-0=Gradient*Input
        inpt=[y0]
        names=['input']
        if self.HLP:
            for key in self.HLPHooksFlex:
                inpt.append(self.HLPHooksFlex[key].x)
                names.append(key)
        G=ag.grad(quantity, inpt, retain_graph=retain_graph, 
                  create_graph=create_graph)
        globals.LRP=False
        #print(G)
        GX={} 
        for i,_ in enumerate(inpt,0):
            GX[names[i]]=(torch.nan_to_num(G[i])*inpt[i]).unsqueeze(1)#unsqueeze adds classes dim
            
        
        if self.Zb:
            #get G from first Relu output, multiply by first relu output. Equivalent to
            #getting the LRP relevance at the input of the first RELU, or at the output of the
            #first layer. Use Zb rule to get input relevance.
            if 'BatchNorm2d' not in self.chain:
                GX['input']=self.ZbModule(rK=GX['input'],
                                          aJ=x,aK=self.firstLayerLinearOutput.output)
            else:
                GX['input']=self.ZbModule(rK=GX['input'],aJ=x,
                                          aKConv=self.firstLayerLinearOutput.output,
                                          aK=self.firstLayerBNOutput.output)
        return GX
    
    def returnBackbone(self):
        """
        Returns backbone DNN, without LRP-Flex wrapper.
        """
        model=self.model
        remove_all_forward_hooks(model)
        return model

def CompoundLoss(out,labels,masks=None,
                 tuneCut=False,d=1,dLoss=1,cutFlex=None,cut2Flex=None,
                 A=1,B=1,E=1,
                 alternativeForeground=False,norm=True):
    """
    Function to apply the ISNet loss, with option of LRP deep supervision (LDS).
    
    Args:
        out:DNN output, dictionaty with logits ('output') and LRP heatmaps ('LRPFlex')
        For LDS, heatmaps are another dictionary, with heatmaps for every supervised layer
        labels: labels
        masks: foreground segmentation masks, valued 1 in foreground and 0 in background
        tuneCut: if True, loss outputs total absolute relevance for selection of C1 and C2 
        hyper-parameters, use for non-ISNet only
        d: background loss GWRP exponential decay
        dLoss: LDS exponential decay (GWRP)
        cutFlex,cut2Flex: C1 and C2 hyper-parameters
        A and B: w1 and w2 parameters, weights for the background and foreground loss
        E: parameter of the background loss activation x/(x+E)
        alternativeForeground='hybrid' adds the loss modification in the paper Faster ISNet
        for Background Bias Mitigation on Deep Neural Networks, alternativeForeground='L2'
        uses the standard (original) loss
        norm: 'absRoIMean' uses standard ISNet loss, 'RoIMean' applies the background loss
        normalization step before the absolute value operation
        
     Return Value:
         ISNet loss
     """
    L={'classification':None,
        'LRPFlex':None,
        'mapAbsFlex':None}
    
    outputs=out['output']
    LRPFlex=out['LRPFlex']
    
    if len(labels.shape)>1:
        labels=labels.squeeze(-1)
    if len(labels.shape)>1:
        L['classification']=F.binary_cross_entropy_with_logits(outputs,labels)
    else:
        L['classification']=torch.nn.functional.cross_entropy(outputs,labels)
    
    if LRPFlex is not None:
        if not isinstance(LRPFlex, dict):
            LRPFlex={'input':LRPFlex}
        if not isinstance(cutFlex, dict):
            cutFlex={'input':cutFlex}
            cut2Flex={'input':cut2Flex}
            
        losses=[]
        tune={}
        for key in LRPFlex:
            if tuneCut:
                heatmapLoss,foreg=LRPLossCEValleysGWRP(LRPFlex[key],
                                                          masks,
                                                           A=A,B=B,d=d,
                                                           E=E,
                                                           rule='e',
                                                           tuneCut=tuneCut,
                                                           norm=norm,
                                                           channelGWRP=1,
                                                       alternativeForeground=alternativeForeground)
                losses.append(heatmapLoss)
                tune[key]=foreg

            else:
                heatmapLoss=LRPLossCEValleysGWRP(LRPFlex[key],
                                                    masks,
                                               cut=cutFlex[key],
                                               cut2=cut2Flex[key],
                                               A=A,B=B,d=d,
                                               E=E,
                                               rule='e',
                                               tuneCut=tuneCut,
                                               channelGWRP=1,
                                               norm=norm,
                                                 alternativeForeground=alternativeForeground)
                losses.append(heatmapLoss)
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=GlobalWeightedRankPooling(heatmapLoss,d=dLoss,oneD=True)
        if tuneCut:
            L['mapAbsFlex']=tune
        L['LRPFlex']=heatmapLoss
    return L    
    
def CleanHooks(hooks):
    """
    Frees memory erasing values stored by DenoiseReLU hooks.
    
    Args:
        hooks: dictionaty with all DenoiseReLU hooks
    """
    for key in hooks:
        try:
            hooks[key][1].clean()
        except:
            CleanHooks(hooks[key])
            
def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    """
    Removes all forward hooks in a networks.
    
    Args:
        model: PyTorch network
    """
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)    
                
def LRPFlexStoreActivations(model,layers=None):
    """
    Insert forward hooks, which store the activations for supervised layers in LRP deep supervision.
    
    Args:
        model: PyTorch network
        layers: list of names of supervised layers, optional
        
    Return Value:
        dictionary with layer names as keys and instantiated hooks
    """
    hooks={}
    for name, module in model.named_modules():
        if layers is not None:
            if name in layers:
                hooks[name]=HookFwd(module,mode='output')
        #penalize same relevances as in the LRP block:
        elif ('DenseNet' in model.__class__.__name__):
            if (name=='features.fReLU' or \
                (('transition' in name) and (name.count('.')==2) and ('LRP' not in name) \
                 and name[-4:]=='relu')):
                hooks[name]=HookFwd(module,mode='output')
                #print('inserted hook:',hooks[name].identifier)
        elif ('ResNet' in model.__class__.__name__):
            if ('layer' in name and 'LRP' not in name and (name.count('.')==0)):
                hooks[name]=HookFwd(module,mode='output')
        elif('VGG' in model.__class__.__name__):
            if 'special' in name:#layer after maxpool, which ignores indexes
                hooks[name]=HookFwd(module,mode='output')
        else:
            raise ValueError('Please specify layer names for all intermediate layers optimized with hidden layer penalization, passing a list as the layer variable')
            
    #print('Hooked lrp Relevance at the output of:',hooks.keys())

    return hooks

def RemoveInplace(net,architecture=None):
    """
    Removes inplace operations.
    
    Args:
        net: PyTorch Network
        architecture: name of backbone, if using pre-defined backbone
        
    Return Value:
        neural network without in-place operations
    """
    if (('DenseNet' in net.__class__.__name__) or
        ('ResNet' in net.__class__.__name__)):
        net2=getBackbone(architecture,dropout=DropoutPresent(net),
                         classes=CountClasses(net))
        net2.load_state_dict(net.state_dict())
    else:
        net2=net
        
    ChangeInplace(net2)
    
    return net2

class HookFwd():
    def __init__(self, module,mode):
        """
        Inserts forward hook that saves input or output of PyTorch module.
        """
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
        
def getBackbone(architecture,dropout,classes,pretrained=False,VGGRemoveLastMaxpool=False):
    """
    Creates pre-defined backbone.
    
    Args:
        architecture: pre-defined backbone architecture name. Either densenets, resnets or vggs
        dropout: if True, adds dropout to pre-defined backbone
        classes: number of output neurons in pre-defined backbones
        pretrained: if True, pre-defined backbone is downloaded pre-trained (ImageNet)
        VGGRemoveLastMaxpool: if True, removes last maxPool from VGG pre-defined backbone
        
    Return Value:
        backbone DNN
    """
    if (architecture=='densenet121'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet121(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet161'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet161(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet169'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet169(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet201'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet201(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet264'):
        import LRPDenseNetZe as LRPDenseNet
        if(pretrained):
            raise ValueError('No available pretrained densenet264')
        classifierDNN=LRPDenseNet.densenet264(pretrained=False,
                                              num_classes=classes)
    elif (architecture=='resnet18'):
        import resnet
        classifierDNN=resnet.resnet18(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet34'):
        import resnet
        classifierDNN=resnet.resnet34(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet50'):
        import resnet
        classifierDNN=resnet.resnet50(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet101'):
        import resnet
        classifierDNN=resnet.resnet101(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet152'):
        import resnet
        classifierDNN=resnet.resnet152(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet18FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet18(num_classes=classes)
    elif (architecture=='resnet34FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                     'resnet34', pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet50FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet50(num_classes=classes)
    elif (architecture=='resnet101FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet101(num_classes=classes)
    elif (architecture=='resnet152FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet152(num_classes=classes)
    elif (architecture=='vgg11'):
        self.classifierDNN=torchvision.models.vgg11(pretrained=pretrained,
                                                    num_classes=classes)
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()
    elif (architecture=='vgg11_bn'):
        self.classifierDNN=torchvision.models.vgg11_bn(pretrained=pretrained,
                                                       num_classes=classes)    
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()
    elif (architecture=='vgg16'):
        self.classifierDNN=torchvision.models.vgg16(pretrained=pretrained,
                                                    num_classes=classes) 
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[30]=torch.nn.Identity()
    elif (architecture=='vgg16_bn'):
        self.classifierDNN=torchvision.models.vgg16_bn(pretrained=pretrained,
                                                       num_classes=classes) 
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()
    elif (architecture=='vgg13'):
        self.classifierDNN=torchvision.models.vgg13(pretrained=pretrained,
                                                    num_classes=classes) 
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()
    elif (architecture=='vgg13_bn'):
        self.classifierDNN=torchvision.models.vgg13_bn(pretrained=pretrained,
                                                       num_classes=classes) 
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()
    elif (architecture=='vgg19'):
        self.classifierDNN=torchvision.models.vgg19(pretrained=pretrained,
                                                    num_classes=classes) 
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()

    elif (architecture=='vgg19_bn'):
        self.classifierDNN=torchvision.models.vgg19_bn(pretrained=pretrained,
                                                       num_classes=classes) 
        if VGGRemoveLastMaxpool:
            self.classifierDNN.features[-1]=torch.nn.Identity()
    else:
        raise ValueError('Architecture must be densenet, resnet or VGG')

    if dropout:
        if ('DenseNet' in classifierDNN.__class__.__name__):
            classifierDNN.classifier=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                   classifierDNN.classifier)
        elif ('ResNet' in classifierDNN.__class__.__name__):
            classifierDNN.fc=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                   classifierDNN.fc)
        else:
            raise ValueError('Unrecognized backbone')

    if (('ResNet' in classifierDNN.__class__.__name__) and
        hasattr(classifierDNN, 'maxpool')):
            classifierDNN.maxpool.return_indices=True
            classifierDNN.maxpool=nn.Sequential(
                OrderedDict([('maxpool',classifierDNN.maxpool),
                             ('special',IgnoreIndexes())]))
    return classifierDNN

class IgnoreIndexes(nn.Module):   
    def __init__(self):
        """
        Special layer to ignore indexes of previous max pooling layer.
        """
        super(IgnoreIndexes, self).__init__()
    def forward(self, x):
        return(x[0])

def CountClasses(net):
    """
    Returns number of output neurons in a DNN.
    
    Args:
        net: neural network (PyTorch)
    """
    if ('DenseNet' in net.__class__.__name__):
        try:
            classes=net.classifier.weight.data.shape[0]
        except:#with dropout
            classes=net.classifier[1].weight.data.shape[0]
    elif ('ResNet' in net.__class__.__name__):
        try:
            classes=net.fc.weight.data.shape[0]
        except:
            classes=net.fc[1].weight.data.shape[0]
    else:
        raise ValueError('Classes counter unrecognized backbone')
    return classes

def DropoutPresent(model):
    """
    Checks if network has Dropout layer.
    
    Args:
        model: neural network (PyTorch)
    """
    if isinstance(model, nn.Dropout):
        return True

    for module in model.children():
        if DropoutPresent(module):
            return True

    return False                         
                         
def RandomLogitIndices(outputs,labels=None,maximum=[]):
    """
    Randomly chooses indices of explained logits, for Stochastic ISNet.
    
    Args:
        outputs: logits
        labels: if not None, chooses the indexes according to labels
        maximum: list where function will save which were the maximum logits chosen, optional
        
    Return Value:
        indices for logits to be explained, per batch element
    """
    indices=[]
    for i,_ in enumerate(outputs,0):
        if len(outputs[i].shape)!=1:
            print(outputs[i].shape)
            raise ValueError('Output has more than 2 dimensions')
        if torch.randint(size=(1,),low=0,high=11).item()>5:#50% chance of propagating highest/label logit
            maximum.append(1)
            if labels is None:
                indices.append(torch.argmax(outputs[i],dim=0))
            else:
                try:
                    indices.append(labels[i].item())
                except:
                    if len(labels[i].shape)>1:
                        raise ValueError('Labels has more than 2 dimensions')
                    indices.append(torch.argmax(labels[i],dim=0))
        else:#50% chance of propagating other random logit
            maximum.append(0)
            randChoice=torch.randint(size=(1,),low=0,high=outputs.shape[1]).item()
            while randChoice==torch.argmax(outputs[i],dim=0):
                randChoice=torch.randint(size=(1,),low=0,high=outputs.shape[1]).item()
            indices.append(randChoice)
    #print('maximum in rand logit creation:',maximum)
    return indices
        
                
def CleanNet(network):
    """
    Removes hooks and apendices from DNN.
    
    Args:
        network: PyTorch network
    """
    try:
        net=network.returnBackbone()
    except:
        net=network
    try:
        remove_all_forward_hooks(net,backToo=True)
    except:
        pass
    try:
        delattr(net,'LRPBlock')
    except:
        pass
    return net
    
            
def ChangeInplace(m: torch.nn.Module):
    """
    Function to remove inplace operations.
    
    Args:
        m: PyTorch network
    """
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

class DenoiseReLU():
    def __init__(self, module):
        """
        Hook for ReLU layers. 

        Implements LRP-Flex modified backward pass when the global variable 
        globals.LRP is True. globals.e is the epsilon value in LRP epsilon

        Args:
            module: ReLU layer
        """
        #self.hook = module.register_backward_hook(self.hook_fn)
        self.hook_fwd = module.register_forward_hook(self.hook_f)
        self.hook_back = module.register_full_backward_hook(self.hook_b)
    def hook_b(self, module, grad_input, grad_output):
        #print(grad_input[0].shape,grad_output[0].shape)
        #print('out:',grad_output[0][0,0])
        #print('in:',grad_input[0][0,0])
        #print('Started backward hook ',self.identifier)
        #if self.output is not None:
        #    print('Output is:',self.output[0][0])
        #else:
        #    print('Output is None')
        if globals.LRP:
            if (globals.e>0):
                atStep=self.output/(self.output+globals.e)
                grad=torch.mul(atStep,grad_output[0])
                if torch.isnan(grad).any():
                    warnings.warn('NaN in LRP backpropagated quantity (G), substituted by 0')
                    grad=torch.nan_to_num(grad)#nans become 0, 0/0=0
                grad=(grad,)
                return grad
            elif (globals.e<0):
                raise ValueError('Negative epsilon in LRP')
            elif (globals.e==0):
                pass #do not alter gradient propagation, use LRP-0/gradient*input
        if not globals.multiple:
            self.output = None#clean memory
            #print('I cleaned memory1')
    def hook_f(self, module, input, output):
        if globals.LRP:
            self.output = output#.clone()
            #print('I saved the output, hook identifier:',self.identifier)
            #print('Output is:',self.output[0][0])
        else: 
            self.output = None
            #print('I saved the output, hook identifier:',self.identifier)
            #print('Output is None')
            #print('I erased the output')
    def clean(self):
        self.output = None
    def close(self):
        self.hook_fwd.remove()
        self.hook_back.remove()
        
class SaveLRP():
    def __init__(self, module):
        """
        Hook for layers supervised by Deep LRP Supervision.

        Saves layer's activation and corresponding LRP heatmap.

        Args:
            module: PyTorch layer
        """
        self.hook_fwd = module.register_forward_hook(self.hook_f)
        self.hook_back = module.register_full_backward_hook(self.hook_b)
    def hook_b(self, module, grad_input, grad_output): 
        #save LRP at the layer output
        self.heatmap=torch.mul(grad_output[0],self.output)#.clone()
    def hook_f(self, module, input, output):
        self.output = output#.clone()
    def close(self):
        self.hook_fwd.remove()
        self.hook_back.remove()

def InsertReLUHooks(m: torch.nn.Module,oldName=''):
    """
    Inserts DenoiseReLU hooks in all ReLU layers. 
    
    Hooks implement LRP-Flex modified backward pass when the global variable globals.LRP 
    is True. globals.e is the epsilon value in LRP epsilon
    
    Args:
        m: Pytorch network
        
    Return Value:
        dictionary with layer names as keys and instantiated hooks
    """
    children = dict(m.named_children())
    output = {}
    if children == {}:
        if m.__class__.__name__=='ReLU':
            return (m,DenoiseReLU(m))
        else:
            return None
    else:
        for name, child in children.items():
            if oldName=='':
                tmp=InsertReLUHooks(child,oldName=name)
            else:
                tmp=InsertReLUHooks(child,oldName=oldName+'.'+name)
            if tmp is not None:
                if oldName=='':
                    output[name] = tmp
                else:
                    output[oldName+'.'+name] = tmp
    return output

class ZbRuleConvInput (nn.Module):
    def __init__(self,layer,l=0,h=1,op=None):
        """
        LRP-Block LRP-Zb layer for convolution followed by ReLU.

        Args:
            layer: DNN convolutional input layer
            l: lowest possible input value
            h: highest possible input value
            op: avoids input shape mismatch, optional
        """
        super().__init__()
        self.layer=layer
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aK):
        try:
            y=ZbRuleConvInputRule(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                                 e=globals.e,l=self.l,h=self.h,op=self.op)
        except:#legacy, if self.op is missing
            y=ZbRuleConvInputRule(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                                 e=globals.e,l=self.l,h=self.h)
        return y
    
class ZbRuleDenseInput (nn.Module):
    def __init__(self,layer,l=0,h=1,op=None):
        """
        LRP-Block LRP-Zb layer for dense layer followed by ReLU.

        Args:
            layer: DNN dense input layer
            l: lowest possible input value
            h: highest possible input value
            op: deprecated
        """
        super().__init__()
        self.layer=layer
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aK):
        y=ZbRuleDenseInputRule(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                             e=globals.e,l=self.l,h=self.h)
        return y
    
class ZbRuleConvBNInput (nn.Module):
    def __init__(self,layer,BN,l=0,h=1,op=None):
        """
        LRP-Block LRP-Zb layer for convolution followed by BN and ReLU.

        Args:
            layer: DNN convolutional input layer
            BN: following batch normalization layer
            l: lowest possible input value
            h: highest possible input value
            op: deprecated
        """
        super().__init__()
        self.layer=layer
        self.BN=BN
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aKConv,aK):
        y=ZbRuleConvBNInputRule(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,
                                   aKConv=aKConv,e=globals.e,
                                   l=self.l,h=self.h)  
        return y
        


def LRPLossCEValleysGWRP (heatmap, mask, cut=1, cut2=25, reduction='mean', 
                         norm='absRoIMean', A=1, B=3, E=1,d=0.9,
                         eps=1e-10,rule='e',tuneCut=False,
                         channelGWRP=1.0,
                         alternativeForeground='L2'):
    """
    ISNet Loss Function.
    
    Args:
        heatmap: heatmaps to be optimized
        mask: foreground segmentation masks, should be 1 in foreground and 0 in background
        cut1 and cut2: C1 and C2 parameters, should define natural range of total absolute 
        heatmap relevance
        reduction: how the loss is reduced
        norm: 'absRoIMean' uses standard ISNet loss, 'RoIMean' applies the background loss 
        normalization step before the absolute value operation
        A and B: w2 and w2 parameters, weights for the foreground and background loss
        E: parameter of the background loss activation x/(x+E)
        d: GWRP exponential decay parameter
        channelGWRP: exponential decay of 1D GWRP to aggregate heatmap losses of each channel 
        in LDS, set to 1 in paper (simple mean)
        alternativeForeground='hybrid' adds the loss modification in the paper Faster ISNet for
        Background Bias Mitigation on Deep Neural Networks, alternativeForeground='L2' uses the
        standard (original) loss
    """
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
    
        
    #print(mask.shape,heatmap.shape)
        
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
        
        #gradient is correct
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
    
    


def ZbRuleConvBNInputRule(layer,BN,rK,aJ,aK,aKConv,e,l=0,h=1,Zb0=False):
    """
    Propagates relevance through the sequence: Convolution, Batchnorm, ReLU using Zb rule.
    
    LRP-Block function.
    
    Args:
        l and h: minimum and maximum allowed pixel values
        layer: convolutional layer throgh which we propagate relevance
        e: LRP-e term. Use e=0 for LRP0
        rK: relevance at layer L ReLU output
        aJ: values at layer L convolution input
        aK: activations after BN
        aKConv: convolution output
        BN: batch normalization layer
        Zb0: removes stabilizer term
        
    Return Value:
        LRP relevance at DNN input
    """
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

def ZbRuleConvInputRule(layer,rK,aJ,aK,e,l=0,h=1,Zb0=False):
    """
    Propagates relevance through the sequence: Convolution, ReLU using Zb rule.
    
    LRP-Block function.
    
    Args:
        l and h: minimum and maximum allowed pixel values
        layer: convolutional layer throgh which we propagate relevance
        e: LRP-e term. Use e=0 for LRP0
        rK: relevance at layer L ReLU output
        aJ: values at layer L convolution input
        aK: activations after BN
        
    Return Value:
        LRP relevance at DNN input
    """    
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

def ZbRuleDenseInputRule(layer,rK,aJ,aK,e,l=0,h=1):
    """
    Propagates relevance through the sequence: Dense, ReLU using Zb rule.
    
    LRP-Block function.
    
    Args:
        l and h: minimum and maximum allowed pixel values
        layer: convolutional layer throgh which we propagate relevance
        e: LRP-e term. Use e=0 for LRP0
        rK: relevance at layer L ReLU output
        aJ: values at layer L convolution input
        aK: activations after BN
        
    Return Value:
        LRP relevance at DNN input
    """
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

def FuseBN(layerWeights, BN, aKConv,Ch0=0,Ch1=None,layerBias=None,
           bias=True,BNbeforeReLU=True):
    """
    Returns parameters of convolution fused with batch normalization.
    
    Args:
        BN:batch normalization layer
        layerWeights: convolutional layer wieghts
        layerBias: convolutional layer bias
        aKConv: BN input
        Ch0 and Ch1: delimits the batch normalization channels that originate from the 
        convolutional layer
        bias: allows returning of bias of equivalent convolution
        BNbeforeReLU: true if BN is placed before the layer activation
    """    
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
    
def stabilizer(aK,e,sign=None):
    """
    Used for LRP-e, returns terms sign(ak)*e.
    
    Args:
        aK: values to stabilize
        e: LRP-e term 
        sign: 1, -1 or None
    """
    if(sign is None):
        signs=torch.sign(aK)
        #zeros as positives
        signs[signs==0]=1
        signs=signs*e
    else:
        signs=sign*torch.ones(aK.shape).type_as(aK)*e
        
    return signs

    
def GlobalWeightedRankPooling(x,d=0.9,descending=True,oneD=False,rank=True):
    """
    GWRP, performs weighted average in spatial dimensions https://arxiv.org/abs/1809.08264.
    
    Args:
        x: input
        d: rate of weights' exponential decay
        descending: if true elements are ordered in descending order before weighting, 
        if false, in ascending order
        oneD: set to True only for aggregating LDS loss, aggregates one-dimensional
        vectors instead of 2D objects
        rank: allows ordering of elements. If False, elements are aggregated in the provided order
    """
    if len(x.shape)==5 and not oneD:#2D pool
        x=x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3]*x.shape[4])
    elif len(x.shape)!=5 and not oneD:
        raise ValueError('input should have 5 dimensions for GWRP')
    if rank:
        x,_=torch.sort(x,dim=-1,descending=descending)
    weights=torch.tensor([d ** i for i in range(x.shape[-1])]).type_as(x)
    while len(weights.shape)<len(x.shape):
        weights=weights.unsqueeze(0)
    x=torch.mul(x,weights)
    x=x.sum(-1)/weights.sum(-1)
    return x