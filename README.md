Official code for paper "ISNet: Costless and Implicit Image Segmentation for Deep Classifiers, with Application in COVID-19 Detection" (https://arxiv.org/abs/2202.00232).

Abstract:
This work proposes a novel deep neural network (DNN) architecture, Implicit Segmentation Neural Network (ISNet), to solve the task of image segmentation followed by classification. It substitutes the common pipeline of two DNNs with a single model. We designed the ISNet for high flexibility and performance: it allows virtually any classification neural network architecture to analyze a common image as if it had been previously segmented. Furthermore, in relation to the unmodified classifier, the ISNet does not cause any increment in computational cost at run-time. We test the architecture with two applications: COVID-19 detection in chest X-rays, and facial attribute estimation. We implement an ISNet based on a DenseNet121 classifier, and compare the model to a U-net (performing lung/face segmentation) followed by a DenseNet121, and to a standalone DenseNet121. The new architecture matched the other DNNs in facial attribute estimation. Moreover, it strongly surpassed them in COVID-19 detection, according to an external test dataset. The ISNet precisely ignored the image regions outside of the lungs or faces. Therefore, in COVID-19 detection it reduced the effects of background bias and shortcut learning, and it improved security in facial attribute estimation. ISNet presents an accurate, fast, and light methodology. The successful implicit segmentation, considering two largely diverse fields, highlights the architecture's general applicability.

## Content
LRPDenseNet.py: code to create a DenseNet, based on the original TorchVision model, but  without in-place ReLU, and with an extra ReLU in transition layers.

ISNetFunctions.py: functions to define the heatmap loss, and to propagate relevance through the LRP Block. 

ISNetLayers.py: LRP Block layers.

globals.py: global variables, for skip connections between classifier and LRP block.

ISNetLightning.py: PyTorch Lightning model of the ISNet, use for simple multi-GPU/multi-node training.

AlternativeModels: Pytorch implementations of benchmark DNNs used in the paper (AG-Sononet, U-Net, multi-task U-Net) and code to produce their heatmaps. Note that hese maps were not used for background relevance minimization, thus, the alternative DNNs were not tested as ISNets. For the AG-Sononet please follow the installation instructions in https://github.com/ozan-oktay/Attention-Gated-Networks, afterwards, to create LRP heatmaps substitute the file sononet_grid_attention.py (in Attention-Gated-Networks-master/models/networks) by the version we provide, and employ the code in AGSononetLRP.py.

## ISNet Creation Example
Defining a DenseNet121 based ISNet:

DenseNet=LRPDenseNet.densenet121(pretrained=False)
#change last layer if needed

net=ISNetLayers.IsDense(DenseNet,heat=True,e=1e-2, Zb=True)

## Citation
Bassi, Pedro RAS, and Andrea Cavalli. "ISNet: Costless and Implicit Image Segmentation for Deep Classifiers, with Application in COVID-19 Detection." arXiv preprint arXiv:2202.00232 (2022).
