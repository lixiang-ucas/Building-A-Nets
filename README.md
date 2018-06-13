

## Description
This branch implements a building extraction on remote sensing images, combining the adversarial networks with a FC-DenseNet model.

## Preparation


## Train
nohup /home/mmvc/anaconda2/envs/Xiang_Li3/bin/python inria3.py --exp_id 1 --model 'FC-DenseNet158' > log1.log
## Test
step 1: to get the prediction results
python run_prediction.py

step 1: evaluation
python eval_aerial.py prediction_#108
