from easydict import EasyDict as edict

config = edict()

config.rec = "Dataset/lfw-deepfunneled/lfw-deepfunneled"
config.num_classes = 5749
config.num_image = 13233
config.num_epoch = 35   #  [22, 30, 35]
config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
config.eval_step= 958 #33350