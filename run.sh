export MIOPEN_USER_DB_PATH=/scratch/yf22/ 
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --save aug100_0.5_$1 --reset > test_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --beta $2 --start_aug $3 --save L1_beta_$2_$3_0.5  --reset > debug_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 4 --patch_size 256 --save baseline2_EDSR_x4 --reset > test_$1.txt 2>&1 &

# Baseline script
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --save baseline_rfdn16_x4_500 --epochs 500 --reset > r16_$1.txt 2>&1 &

# Distil script
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save distil_EDSR_rfdn_x4_500_$2 --epochs 500 --reset --distil --teacher_model 'EDSR' > r16_$1.txt 2>&1 &

# Aug and distil script
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save EDSRmin_c0.1_rfdn16_x4_500_$2 --reset --distil --data_aug 'cutmix' --epochs 500 --teacher_model 'EDSR' --aug_alpha 0.1 > r16_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save EDSRmin_m1.2_rfdn16_x4_500_$2 --reset --distil --data_aug 'mixup' --epochs 500 --teacher_model 'EDSR' --aug_beta 1.2 > r16_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save EDSRmin_advcm0.1_rfdn16_x4_500_$2 --reset --distil --data_aug 'advcutmix' --epochs 500 --teacher_model 'EDSR' --aug_alpha 0.1 > test_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save EDSRmin_cm_rfdn16_x4_500_$2 --reset --distil --data_aug 'cutmixup' --epochs 500 --teacher_model 'EDSR' --aug_beta 1.2 --aug_alpha 0.1 > r16_$1.txt 2>&1 &

# mix method script
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save EDSRmin_c0.1_rfdn_x4_300_$2 --reset --distil --cutmix --teacher_model 'EDSR' --aug_alpha 0.1 > test_$1.txt 2>&1 &

# Debug
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --save debug --reset

#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save debug --reset --distil --data_aug 'mixup' --epochs 500 --teacher_model 'EDSR' --aug_alpha 0.05 i
#HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 4 --patch_size 256 --beta $2 --save EDSRmin_cm_rfdn_x4_500_$2 --reset --distil --data_aug 'cutmixup' --epochs 500 --teacher_model 'EDSR' --aug_beta 1.2 --aug_alpha 0.1
