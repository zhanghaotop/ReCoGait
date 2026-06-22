# # **************** For SUSTech1K ****************
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/recogait/recogait_SUSTech1K.yaml --phase test
