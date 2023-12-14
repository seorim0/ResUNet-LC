# **MULTI-LABEL ABNORMALITY CLASSIFICATION FROM 12-LEAD ECG USING A 2D RESIDUAL U-NET**   
   
This is an official repo of the paper "**MULTI-LABEL ABNORMALITY CLASSIFICATION FROM 12-LEAD ECG USING A 2D RESIDUAL U-NET**," which is accepted to ICASSP 2024.   

**Abstract**：This paper proposes a two-dimensional (2D) deep neural network (DNN) model for the electrocardiogram (ECG) abnormality classification, which effectively utilizes the inter and intra-lead information comprised in the 12-lead ECG.
The proposed model is designed using a stack of residual U-shaped (ResU) blocks so that it can effectively capture ECG features in a multiscale.
The 2D features extracted by the ResU block are down-mixed to 1D features using a lead combiner block designed to merge features of the lead domain into both the time and channel domain.
Through experiments, we confirm that our model outperforms other state-of-the-art models in various metrics.

## Update:  
* **2023.12.14** Upload codes  

## Requirements 
This repo is tested with Ubuntu 22.04, PyTorch 2.0.1, Python3.10, and CUDA11.7. For package dependencies, you can install them by:

```
pip install -r requirements.txt    
```   


## Getting started    
1. Install the necessary libraries.   
2. Download the PhysioNet Challenge 2021 database and place it in '../Dataset/' folder.   
```
├── 📦 ResUNet_LC   
│   └── 📂 dataset   
│       └── 📜 train_dataset.csv   
│       └── 📜 test_dataset.csv   
│   └── ...   
└── 📦 Dataset   
    └── 📂 physionet_challenge_dataset
        └── 📂 physionet.org 
            └── ...
```
If you want to get csv file, please contact us.

3. Run [train_interface.py](https://github.com/seorim0/ResUNet-LC/blob/main/train_interface.py)
  * You can simply change any parameter settings if you need to adjust them.   ([options.py](https://github.com/seorim0/ResUNet-LC/blob/main/options.py)) 


## Results  

![001](https://github.com/seorim0/ResUNet-LC/assets/55497506/fe74d4be-3b02-495c-b3db-0d60ef31b81a)  

![002](https://github.com/seorim0/ResUNet-LC/assets/55497506/75d998ba-8f45-48d7-8ea1-63ba93d44d5b)

![f1_scores_graph_distribution](https://github.com/seorim0/ResUNet-LC/assets/55497506/d7f69fda-a6e4-42d3-8111-e1d76012066f)

![auprc_graph_distribution](https://github.com/seorim0/ResUNet-LC/assets/55497506/f219168f-11ff-478b-a423-284a046302eb)


## Reference   
**Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021**    
Matthew Reyna, Nadi Sadr, Annie Gu, Erick Andres Perez Alday, Chengyu Liu, Salman Seyedi, Amit Shah, and Gari Clifford  
[[paper]](https://physionet.org/content/challenge-2021/1.0.3/)   
**Automatic diagnosis of the 12-lead ECG usinga deep neural network**    
Antônio H. Ribeiro, et al.  
[[paper]](https://www.nature.com/articles/s41467-020-15432-4) [[code]](https://github.com/antonior92/automatic-ecg-diagnosis)  
**A multi-view multi-scale neural network for multi-label ECG classification**    
Shunxiang Yang, Cheng Lian, Zhigang Zeng, Bingrong Xu, Junbin Zang, and Zhidong Zhang  
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10021962) [[code]](https://github.com/ysxGitHub/MVMS-net)    
**Classification of ECG using ensemble of residual CNNs with attention mechanism**    
Petr Nejedly, Adam Ivora, Radovan Smisek, Ivo Viscor, Zuzana Koscova, Pavel Jurak, and Filip Plesinger  
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9662723) [[code]](https://moody-challenge.physionet.org/2021/)  


## Contact  
Please get in touch with us if you have any questions or suggestions.   
E-mail: allmindfine@yonsei.ac.kr (Seorim Hwang) / jbcha7@yonsei.ac.kr (Jaebine Cha)
