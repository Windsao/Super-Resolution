#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --save aug100_0.5_$1 --reset > test_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --beta $2 --start_aug $3 --save L1_beta_$2_$3_0.5  --reset > debug_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --save debug --reset --distil
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --save debug --reset
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --beta $2 --save distil_base_$2 --reset --distil > test_$1.txt 2>&1 &
HIP_VISIBLE_DEVICES=$1 python main.py --model RFDN --scale 2 --patch_size 64 --save debug --reset

#HIP_VISIBLE_DEVICES=$1,$2 python main.py --model EDSR --scale 2 --save adv_edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset > adv_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=0,1,2,3,4 python main.py --model EDSR --scale 2 --save test --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset
#python main.py --model EDSR --scale 2 --save adv_edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset > paper_$1.txt 2>&1 &
#HIP_VISIBLE_DEVICES=$1 python main.py --model EDSR --scale 2 --patch_size 96 --save test --reset 
