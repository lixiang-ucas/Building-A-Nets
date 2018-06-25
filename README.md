

## Description
This branch implements a building extraction on remote sensing images, combining the adversarial networks with a FC-DenseNet model.

## Network structure

![image](https://github.com/lixiang-ucas/Building-A-Nets/blob/master/Images/began-v4.jpg)
Overview of our segmentation architecture with the adversarial network. Left: The segmentation network takes an aerial image as input and produces a pixel-wise classification label map. Right: A label map, chosen from segmentation output or ground truth, is multiplied with their corresponding input aerial image to produce a masked image, and the adversarial network takes this masked image map as input and adopts an auto-encoder network to reconstruct it.




## Preparation


## Train
nohup /home/mmvc/anaconda2/envs/Xiang_Li3/bin/python inria3.py --exp_id 1 --model 'FC-DenseNet158' > log1.log
## Test
step 1: to get the prediction results
python run_prediction.py

step 1: evaluation
python eval_aerial.py prediction_#108

##  Results
Test accuracy of different models on the Massachuttes dataset.

|      Model                  | Breakeven ($\rho$ = 3)       | Breakeven ($\rho$ = 0)            |   Time (s) |
|------------------------|:-------------------:|:---------------------:|:------:|
| Mnih-CNN~\cite{mnih2013machine} | 92.71 |   76.61 | 8.7| 
| Mnih-CNN+CRF~\cite{mnih2013machine} |  92.82 | 76.38 | 26.6
|Saito-multi-MA~\cite{saito2016multiple} |   95.03 | 78.73| 67.7|
|Saito-multi-MACIS~\cite{saito2016multiple}  | 95.09 | 78.72 | 67.8|
|HF-FCN~\cite{zuo2016hf} | 96.43 | 84.24 | 1.07|
|Ours (56 layers) |  96.40 | 83.17 | \textbf{1.01}|
|Ours (158 layers) | \textbf{96.78} |    \textbf{84.79} |    4.38|


Validation accuracy of different network depths on Inria Aerial Image Labeling dataset.

|FC-DenseNet (56 layers) | 74.64 | 96.01|
|------------------------|:-------------------:|:---------------------:|
|Ours (56 layers) | 74.75 | 96.01|
| FC-DenseNet (103 layers) | 75.58 | 96.19 |
| Ours (103 layers) | 76.31 | 96.32 |
|FC-DenseNet (158 layers) | 77.11 | 96.45  |
|Ours (158 layers) | \textbf{78.73}  | \textbf{96.71} |

![image](https://github.com/lixiang-ucas/Building-A-Nets/blob/master/Images/examples.jpg)
