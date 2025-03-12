Official code for paper "Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization" (https://www.nature.com/articles/s41467-023-44371-z), a background bias mitigation optimizaiton strategy (Nature Communications, 2024).

### Code usability and readme improvements comming soon

---

Abstract:

Features in images’ backgrounds can spuriously correlate with the images’ classes, representing background bias. They can influence the classifier’s decisions, causing shortcut learning (Clever Hans effect). The phenomenon generates deep neural networks (DNNs) that perform well on standard evaluation datasets but generalize poorly to real-world data. Layer-wise Relevance Propagation (LRP) explains DNNs’ decisions. Here, we show that the optimization of LRP heatmaps can minimize the background bias influence on deep classifiers, hindering shortcut learning. By not increasing run-time computational cost, the approach is light and fast. Furthermore, it applies to virtually any classification architecture. After injecting synthetic bias in images’ backgrounds, we compared our approach (dubbed ISNet) to eight state-of-the-art DNNs, quantitatively demonstrating its superior robustness to background bias. Mixed datasets are common for COVID-19 and tuberculosis classification with chest X-rays, fostering background bias. By focusing on the lungs, the ISNet reduced shortcut learning. Thus, its generalization performance on external (out-of-distribution) test databases significantly surpassed all implemented benchmark models.

# Installation
```bash
conda create --name isnet python=3.8
conda activate isnet
conda install pip=23.3.2
conda install ipykernel
pip install -r requirements.txt
```

## Content
LRPDenseNet.py: code to create a DenseNet, based on the original TorchVision model, but  without in-place ReLU, and with an extra ReLU in transition layers.

ISNetFunctions.py: functions to define the heatmap loss, and to propagate relevance through the LRP Block. 

ISNetLayers.py: LRP Block layers and ISNet PyTorch model.

globals.py: global variables, for communication between classifier and LRP block.

ISNetLightning.py: PyTorch Lightning ISNet model, use for simple multi-GPU/multi-node training.

AlternativeModels: Pytorch implementations of benchmark DNNs used in the paper (AG-Sononet, U-Net, multi-task U-Net, HAM and GAIN). For the standard DenseNet121, use the code in LRPDenseNet.py. For the AG-Sononet, please follow the installation instructions in https://github.com/ozan-oktay/Attention-Gated-Networks, afterwards, to create LRP heatmaps substitute the file sononet_grid_attention.py (in Attention-Gated-Networks-master/models/networks) by the version we provide, and employ the code in AGSononetLRP.py. For HAM, please visit https://github.com/oyxhust/HAM. We provide code to implement the DNN in PyTorch Lightning.


## ISNet Creation Example
Defining a DenseNet121-based ISNet:
```
import LRPDenseNet
import ISNetLayers

DenseNet=LRPDenseNet.densenet121(pretrained=False)
#change last layer if needed

net=ISNetLayers.IsDense(DenseNet,heat=True,e=1e-2, Zb=True)
```

## Citation
Bassi, P.R.A.S., Dertkigil, S.S.J. & Cavalli, A. Improving deep neural network generalization and robustness to background bias via layer-wise relevance propagation optimization. Nat Commun 15, 291 (2024). https://doi.org/10.1038/s41467-023-44371-z

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

## Observations for the Training Procedure
For better stability and convergence in the training procedure, we suggest employing gradient clipping (we used norm of 1) and deterministic operations, which may be selected with the following code: torch.use_deterministic_algorithms(True).

A higher P hyper-parameter (heatmap loss weight) increases bias resistance, but reduces training speed. A lower d hyper-parameter (GWRP decay in the heatmap loss) also increases bias resistance, but reduces training stability. If training losses do not converge, please consider increasing the d hyper-parameter and/or reducing learning rate.

## Pre-trained Models
For the applications in the paper (COVID-19 and Tuberculosis detection, CheXpert classification, dog breed classification and facial attribute estimation), please find the ISNet pre-trained weigths on: https://drive.google.com/drive/folders/1hIJp4c11R65db2K7EN4rYQ7dcr7ce72t?usp=sharing
