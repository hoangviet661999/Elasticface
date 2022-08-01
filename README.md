1. Cấu hình dataset trong config/config.py:
      config.rec = path to train_folder
      config.val_rec = path to val_folder
      config.num_classes = num classes
      config.num_image = num images
      config.num_epoch = num epoches

2. Run script: 
      1. pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
      2. pip install -r requirement.txt
      3. python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1235 train.py --dataset LFW(config.dataset)