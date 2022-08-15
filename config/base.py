from easydict import EasyDict as edict

config = edict()
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128 # batch size per GPU
config.lr = 0.1
config.output = "output/R100_ElasticArcFace" # train model output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.50
config.std=0.05
config.warmup_epoch = -1

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

config.loss="ElasticArcFace"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus

if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05


# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet100"
config.SE=False # SEModule

config.rec = "../datasets/ms1m-retinaface-t1/"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch =  26
config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
config.eval_step=5686
config.lr_func = lr_step_func
