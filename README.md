
<div align="center">
  <img src="https://github.com/user-attachments/assets/e804dce9-9961-4a11-a046-3a7df4836552" alt="ISNetLogo" width="200"/>
</div>


# ISNet & Faster ISNet

Natural and medical images (e.g., X-rays) coommonly present background features that are correlated to the classes we want to classify. For example, in an early COVID-19 classification dataset, most COVID-19 X-rays came from Italy, while most healthy images came from the USA. Thus, classifiers trained in these datasets saw Italian words in the background of X-rays as signs of COVID-19, increasing their confidence for the COVID-19 class. These classifiers generalized poorly to new hospitals. To mitigate background bias, we introduced the **ISNet**, a training procedure that guides classifiers to focus on relevant image regions (e.g., lungs) and ignore biased backgrounds. To achieve this, the ISNet minimizes background attention in Layer-wise Relevance Propagation (LRP) heatmaps. The ISNet ignored even strong background bias in X-rays and natural images, improving OOD generalization (e.g., to new hospitals). The ISNet surpassed several alternative methods, like Right for the Right Reasons and Grad-CAM based methods. The **Faster ISNet** is an evolution of the ISNet, with faster training and **easy application to any neural network architecture**. During testing, the ISNet adds no extra computational cost to the classifier.


## Papers

<b>Improving deep neural network generalization and robustness to background bias via layer‐wise relevance propagation optimization</b> <br/>
Pedro R. A. S. Bassi, Sergio S. J. Dertkigil, Andrea Cavalli <br/>
**Nature Communications**, 15, 291 (2024) <br/>
[Read More](https://www.nature.com/articles/s41467-023-44371-z) <br/>

<br/>

<b>Faster ISNet for Background Bias Mitigation on Deep Neural Networks</b> <br/>
Pedro R. A. S. Bassi, Sergio Decherchi, Aandrea Cavalli <br/>
**IEEE Access** (2024) <br/>
[Read More](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10681068) <br/>


## Installation
```bash
cd code
conda create --name isnet python=3.8
conda activate isnet
conda install pip=23.3.2
conda install ipykernel
pip install -r requirements.txt
```

## Quick Start

<div align="center">
  <img src="https://github.com/user-attachments/assets/56fac6b6-c4bc-4b91-9a8c-2daa1fca4e54" alt="ISNetTeaser" width="500"/>
</div>

### Train and test

Use the following code to train and test an ISNet in the MNIST dataset with background bias.

For the original version of the ISNet (Nature Communications):
```bash
python RunISNet.py
```

For the latest version of the ISNet (faster training):
```bash
python RunISNetFlex.py
```

### Results

As shown in the figure above, the ISNet will not pay attention to the background of the images, where we inserted an artificial bias. This bias is highly correlated to the MNIST classes, so standard classifiers will learn to pay great attention to them, but they will suffer in testing if we remove the bias or change the correlation between biases and classes. Meanwhile, the ISNet will have the same accuracy in testing whether the bias is present or not, because it ignroes the image backgrounds.



## Use LRP-Flex to explain an arbitrary DNN decision
This repository also includes LRP-Flex, a easy to use methodology which creates LRP heatmaps for any classifier architecture in PyTorch. The Fatser ISNet is based on LRP-Flex.


```
import ISNetFlexTorch
import LRPDenseNetZe

#Examples of network and image
DenseNet=LRPDenseNetZe.densenet121(pretrained=False)
image=torch.randn([1,3,224,224])

#LRP-Flex PyTorch Wrapper
net=ISNetFlexTorch.ISNetFlex(model=DenseNet,
                             architecture='densenet121',#write architecture name only for densenet, resnet and VGG
                             selective=True,Zb=True,multiple=False,HiddenLayerPenalization=False,
                             randomLogit=False,explainLabels=True)#set explainLabels=False when defining ISNet

#Explain class 3
out=net(image,runLRPFlex=True,labels=torch.tensor([3]))
logits=out['output']
heatmap=out['LRPFlex']['input']

#Plot heatmap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
h=heatmap.squeeze().mean(0).detach().numpy()
norm=colors.TwoSlopeNorm(vmin=h.min(), vcenter=0, vmax=h.max())
plt.imshow(h,cmap='RdBu_r', norm=norm,interpolation='nearest')
plt.show()
```


## Faster ISNet Creation Examples

Dependencies: Python, PyTorch, PyTorch Lightning

#### LRP-Flex-based ISNets: An easy and fast way to make classifiers ignore backgrounds (Faster ISNet Paper)

<details>
  <summary><strong>Click to expand details</strong></summary>

  ```
  import LRPDenseNetZe
  import ISNetFlexLightning

  DenseNet=LRPDenseNetZe.densenet121(pretrained=False)#Example of DNN

  #Stochastic ISNet
  net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=False,multiple=False,
                                      HiddenLayerPenalization=False,
                                      randomLogit=True,heat=True)
                                  
  #Stochastic ISNet LRP Deep Supervision
  net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=False,multiple=False,
                                      HiddenLayerPenalization=True,
                                      randomLogit=True,heat=True)
  #Selective ISNet
  net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=True,multiple=False,
                                      HiddenLayerPenalization=False,
                                      randomLogit=False,heat=True)

  #Selective ISNet LRP Deep Supervision
  net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=True,multiple=False,
                                      HiddenLayerPenalization=True,
                                      randomLogit=False,heat=True)
                                  
  #Original ISNet
  net=ISNetFlexLightning.ISNetFlexLgt(model=DenseNet,selective=False,multiple=True,
                                      HiddenLayerPenalization=False,
                                      randomLogit=False,heat=True)
  ```

</details>

#### LRP Block-based ISNets (Original ISNet Paper - Nature Comms.):


<details>
  <summary><strong>Click to expand details</strong></summary>

  ```
  import ISNetLightningZe

  #Dual ISNet
  net=ISNetLightningZe.ISNetLgt(architecture='densenet121',classes=10,selective=False,multiple=False,
                                penalizeAll=False,highest=False,randomLogit=True,rule='z+e')

  #Dual ISNet LRP Deep Supervision
  net=ISNetLightningZe.ISNetLgt(architecture='densenet121',classes=10,selective=False,multiple=False,
                                penalizeAll=True,highest=False,randomLogit=True,rule='z+e')                           

  ```

</details>

## Files and Content

<details>
  <summary><strong>Click to expand details</strong></summary>

  ### LRP-Flex-based ISNets:

  ISNets based on the LRP-Flex model agnostic implementation from "Faster ISNet for Background Bias Mitigation on Deep Neural Networks".

  ISNetFlexLightning.py: PyTorch Lightning implementation of Selective, Stochastic and Original ISNets, based on LRP-Flex.

  ISNetFlexTorch.py: PyTorch implementation of Selective, Stochastic and Original ISNets, based on LRP-Flex.

  ### LRP Block-based ISNets:

  ISNets based on the LRP Block implementation, from (1), with the modifications explained in Appendix B of the paper "Faster ISNet for Background Bias Mitigation on Deep Neural Networks". Implemented for DenseNet, ResNet, VGG and simple nn.Sequential backbones.

  ISNetLightningZe.py: PyTorch Lightning implementation of all Faster and Original ISNets, based on LRP Block.

  ISNetLayersZe.py: PyTorch implementation of all Faster and Original ISNets, based on LRP Block.

  ISNetFunctionsZe.py: Functions for LRP Block, introduced in (1) and expanded in this work.

  ### ISNet Softmax Grad * Input Ablation:

  ISNetLightningZeGradient.py: Implementation of ISNet Softmax Grad * Input ablation study.

  ### Extras:

  globalsZe.py global variables shared across modules.

  LRPDenseNetZe.py: DenseNet code, based on TorchVision. Removes in-place ReLU, and adds an extra ReLU in transition layers. From (1).

  resnet.py: resnet code, based on TorchVision. Removes in-place ReLU, and adds an extra ReLU in transition layers.


  ### Training Script Examples:

  RunISNetGrad.py: Train and test ISNet Softmax Grad* Input on MNIST.

  RunISNet.py: Train and test LRP Block-based ISNets on MNIST.

  RunISNetFlex.py: Train and test LRP-Flex-based ISNets on MNIST.

  SingleLabelEval.py: Evaluation script.

  compare_auc_delong_xu.py: Dependency of SingleLabelEval.py.

  locations.py: Folder locations for training script.

</details>


## Pre-trained Models
For the applications in the Nature Communications paper (COVID-19 and Tuberculosis detection, CheXpert classification, dog breed classification and facial attribute estimation), please find the ISNet pre-trained weigths on: https://drive.google.com/drive/folders/1hIJp4c11R65db2K7EN4rYQ7dcr7ce72t?usp=sharing


## Observations for the Training Procedure
For better stability and convergence in the training procedure, we suggest employing gradient clipping (we used norm of 1) and deterministic operations, which may be selected with the following code: torch.use_deterministic_algorithms(True).

A higher P hyper-parameter (heatmap loss weight) increases bias resistance, but reduces training speed. A lower d hyper-parameter (GWRP decay in the heatmap loss) also increases bias resistance, but reduces training stability. If training losses do not converge, please consider increasing the d hyper-parameter and/or reducing learning rate.



## Datasets

Our study is based on public datasets.

COVID-19 X-ray database: see details in https://doi.org/10.1038/s41467-023-44371-z.

Tuberculosis X-ray database: see details in https://doi.org/10.1038/s41467-023-44371-z.

Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/

MNIST: http://yann.lecun.com/exdb/mnist/


## Citations
If you use this code, please cite the papers below:

```
@article{Bassi2024,
  title = {Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization},
  volume = {15},
  ISSN = {2041-1723},
  url = {http://dx.doi.org/10.1038/s41467-023-44371-z},
  DOI = {10.1038/s41467-023-44371-z},
  number = {1},
  journal = {Nature Communications},
  publisher = {Springer Science and Business Media LLC},
  author = {Bassi,  Pedro R. A. S. and Dertkigil,  Sergio S. J. and Cavalli,  Andrea},
  year = {2024},
  month = jan 
}
```

```
@ARTICLE{Bassi2024-qj,
  title     = "Faster {ISNet} for background bias mitigation on deep neural
               networks",
  author    = "Bassi, Pedro R A S and Decherchi, Sergio and Cavalli, Andrea",
  journal   = "IEEE Access",
  publisher = "Institute of Electrical and Electronics Engineers (IEEE)",
  volume    =  12,
  pages     = "155151--155167",
  year      =  2024,
  copyright = "https://creativecommons.org/licenses/by/4.0/legalcode"
}

```



