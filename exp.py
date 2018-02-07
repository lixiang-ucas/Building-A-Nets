############
# 训练
############
nohup python main.py --model 'Encoder-Decoder-Skip' --gpu 0> log_encoder.file


python main.py 
#7
python main7.py --dataset './data/mass_buildings/patches'

nohup python main.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet56' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 0 > log_fc56.file

#8
nohup python main.py --dataset './data/mass_buildings/patches' --model 'Encoder-Decoder-Skip' --crop_height 256 --crop_width 256 --num_val_images 1000 > log_encoder.file
#9
nohup python main.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet158' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 0 > log_fc158.file

#HF-FCN
nohup python main.py --dataset './data/mass_buildings/patches' --model 'HF-FCN' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 1 > log_fc158

#11 RefineNet
nohup python main.py --dataset './data/mass_buildings/patches' --model 'RefineNet-Res101' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 0 > log_RefineNet-Res101.txt

#12 1cls
nohup python main_1cls.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet103' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 1 > log_#12.txt

#13 batch_size=2
nohup python main.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet103' --crop_height 256 --crop_width 256 --num_val_images 100 --batch_size 2 --gpu 0 > log_#13.txt


#14
nohup python main_ISPRS2.py --dataset '/media/zhoun/Data/lx/caffe/DeepNetsForEO/ISPRS/Vaihingen/vaihingen_128_128_32_fold1' --model 'FC-DenseNet103' --crop_height 128 --crop_width 128 --batch_size 3 --gpu 0 > log_#14.txt

nohup python main_ISPRS2.py --dataset '/home/mmvc/Xiang_Li/Vaihingen128' --model 'FC-DenseNet103' --crop_height 128 --crop_width 128 --batch_size 3 --gpu 0 > log_#14.txt

##not implemented!
nohup python main_ISPRS2.py --dataset '/media/zhoun/Data/lx/caffe/DeepNetsForEO/ISPRS/Vaihingen/vaihingen_128_128_32_fold1' --model 'FC-DenseNet103' --crop_height 128 --crop_width 128 --batch_size 3 --balanced_weight 1 > log_#15.txt

#15 dropout=0.5
nohup python main.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet158' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 0 > log_#15-3.txt


#16
nohup python main16.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet56' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 1 > log_#16.txt


#17
nohup python main.py --dataset './data/mass_buildings/patches' --model 'FC-DenseNet158' --crop_height 256 --crop_width 256 --num_val_images 100 --gpu 0  --is_balanced_weight 1 > log_#17.txt


#18 给交点像素(边缘像素)更大的权重
nohup python main_soft_cost.py --dataset './data/mass_buildings/patches256' --model 'FC-DenseNet158' --is_BC 1 --is_edge_weight 1 --crop_height 256 --crop_width 256 --num_val_images 100 -gpu 0 > log_#18.txt


#19 Camvid
nohup python main_soft_cost.py --gpu 0 --is_edge_weight 1 --crop_height 256 --crop_width 256 --num_epochs 1 > log_#19.txt

#20 No sliding window
nohup python main_soft_cost.py --dataset './data/mass_buildings/patches256-2' --model 'FC-DenseNet158' --is_BC 1 --is_edge_weight 1 --crop_height 256 --crop_width 256 --h_flip 1 --v_flip 1 --num_epochs 200 --gpu 0

#21 ISPRS3
nohup python main_ISPRS3.py --dataset '../DL_DATA/Vaihingen/vaihingen_128_128_32_fold1' --model 'FC-DenseNet158' --is_BC 1 --crop_height 128 --crop_width 128 --gpu 1 > log_#21.txt

#22 change weight value
nohup python main_soft_cost-2.py --exp_id 22 --continue_training 1 --dataset './data/mass_buildings/patches256' --model 'FC-DenseNet158' --is_BC 1 --is_edge_weight 1 --crop_height 256 --crop_width 256 --h_flip 1 --v_flip 1 --num_val_images 100 --gpu 0 > log_#22.txt


#23 Camvid
nohup python main23.py --exp_id 23 --gpu 1> log_#23.txt

#41
nohup python main_soft_cost.py --exp_id 41 --dataset './data/AerialImageDataset/patches256' --model 'FC-DenseNet158' --is_BC 1 --crop_height 256 --crop_width 256 --h_flip 1 --v_flip 1 --num_epochs 200 --gpu 0


#24
nohup python main_multi_gpus.py --exp_id 24 --dataset './data/mass_buildings/patches256' --model 'FC-DenseNet158' --is_BC 1 --is_edge_weight 1 --crop_height 256 --crop_width 256 --h_flip 1 --v_flip 1 --num_val_images 100 --gpu_ids 1 --batch_size 2 > log_#24.txt


#
# 融合HF-FCN和DenseNet,将HF-FCN中的融合层加到DenseNet的Decoder中
# 使用ImageNet与训练DenseNet
# DenseNet中的BC层,transction down

