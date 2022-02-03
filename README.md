Official code for paper "ISNet: Costless and Implicit Image Segmentation for Deep Classifiers, with Application in COVID-19 Detection".

LRPDenseNet.py: code to create a DenseNet, based on original TorchVision model, but  without in place ReLU and with an extra ReLU in transition layers.

ISNetFunctions.py: functions to define heatmap loss and relevance propagation. 

ISNetLayers.py: functions to create an  ISNet.

globals.py: global variables, for skip connections between classifier and LRP block.

TrainedModels: parameters for models trained in the paper.

Defining a DenseNet121 based ISNet:
DenseNet=LRPDenseNet.densenet121(pretrained=False)
#change last layer if needed
net=ISNetLayers.IsDense(DenseNet,heat=True,e=1e-2,device='cuda:0', Zb=True)
