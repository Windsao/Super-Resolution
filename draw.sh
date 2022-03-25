CUDA_VISIBLE_DEVICES=6,7 python main.py --model RFDN --scale 4 --patch_size 256 --save debug --reset --epochs 800 --distil --teacher_model 'EDSR' --resume_dir '../experiment/EDSRmin_SI_8_rfdn32_x4_1/'
# CUDA_VISIBLE_DEVICES=0 python main.py --model RFDN --scale 4 --patch_size 256 --beta $1 --save distil_EDSR_rfdn32_x4_$1 --epochs 500 --reset --distil --teacher_model 'EDSR_paper'
# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --model EDSR --scale 2 --patch_size 192 --save debug --reset --data_aug 'pixel_mask'
# --resume_dir '../experiment/EDSRmin_SI_8_rfdn32_x4_1/'