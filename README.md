Official code for paper "Towards Ignoring Backgrounds and Improving Generalization: a Costless DNN Visual Attention Mechanism" (https://arxiv.org/abs/2202.00232).

Abstract:
This work introduces an attention mechanism for image classifiers and the corresponding deep neural network (DNN)
architecture, dubbed ISNet. During training, the ISNet uses segmentation targets to learn how to find the image’s region of
interest and concentrate its attention on it. The proposal is based on a novel concept, background relevance minimization
in LRP explanation heatmaps. It can be applied to virtually any classification neural network architecture, without any extra
computational cost at run-time. Capable of ignoring the background, the resulting single DNN can substitute the common
pipeline of a segmenter followed by a classifier, being faster and lighter. We tested the ISNet with three applications: COVID-19
and tuberculosis detection in chest X-rays, and facial attribute estimation. The first two tasks employed mixed training databases,
which fostered background bias and shortcut learning. By focusing on lungs, the ISNet reduced shortcut learning, improving
generalization to external (out-of-distribution) test datasets. When training data presented background bias, the ISNet’s
test performance significantly surpassed standard classifiers, multi-task DNNs (performing classification and segmentation),
attention-gated neural networks, Guided Attention Inference Networks, and the standard segmentation-classification pipeline.
Facial attribute estimation demonstrated that ISNet could precisely focus on faces, being also applicable to natural images.
ISNet presents an accurate, fast, and light methodology to ignore backgrounds and improve generalization, especially when
background bias is a concern.

## Content
LRPDenseNet.py: code to create a DenseNet, based on the original TorchVision model, but  without in-place ReLU, and with an extra ReLU in transition layers.

ISNetFunctions.py: functions to define the heatmap loss, and to propagate relevance through the LRP Block. 

ISNetLayers.py: LRP Block layers.

globals.py: global variables, for skip connections between classifier and LRP block.

ISNetLightning.py: PyTorch Lightning model of the ISNet, use for simple multi-GPU/multi-node training.

AlternativeModels: Pytorch implementations of benchmark DNNs used in the paper (AG-Sononet, U-Net, multi-task U-Net, HAM and GAIN). For the standard DenseNet121, use the code in LRPDenseNet.py. For the AG-Sononet, please follow the installation instructions in https://github.com/ozan-oktay/Attention-Gated-Networks, afterwards, to create LRP heatmaps substitute the file sononet_grid_attention.py (in Attention-Gated-Networks-master/models/networks) by the version we provide, and employ the code in AGSononetLRP.py. For HAM, please visit https://github.com/oyxhust/HAM. We provide code to implement the DNN in PyTorch Lightning.

## ISNet Creation Example
Defining a DenseNet121 based ISNet:

DenseNet=LRPDenseNet.densenet121(pretrained=False)
#change last layer if needed

net=ISNetLayers.IsDense(DenseNet,heat=True,e=1e-2, Zb=True)

## Citation
Bassi, Pedro RAS, Dertkigil, Sergio SJ, and Andrea Cavalli. "Towards Ignoring Backgrounds and Improving Generalization: a Costless DNN Visual Attention Mechanism." ArXiv preprint. ArXiv:2202.00232 (2022).

## Observations for the Training Procedure
For better stability and convergence in the training procedure, we suggest employing gradient clipping (we used norm of 1) and deterministic operations, which may be selected with the following code: torch.use_deterministic_algorithms(True).

A higher P hyper-parameter (heatmap loss weight) increases bias resistance, but reduces training speed. A lower d hyper-parameter (GWRP decay) also increases bias resistance, but reduces training stability. If training losses do not converge, please consider increasing the d hyper-parameter and/or reducing learning rate. 
