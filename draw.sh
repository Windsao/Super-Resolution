#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --model RFDN --scale 4 --patch_size 256 --save debug --reset --distil --data_aug 'cutmix'
HIP_VISIBLE_DEVICES=2 python main.py --model RFDN --scale 4 --patch_size 256 --save debug --reset --distil --data_aug 'cutmix'
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --model EDSR --scale 2 --patch_size 192 --save debug --reset 
