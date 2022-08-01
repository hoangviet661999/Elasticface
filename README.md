Run scripts:
 1. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
 2. pip install -r requirement.txt
 3. python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1235 train.py --config/flw