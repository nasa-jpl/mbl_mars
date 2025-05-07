from models.LoFTR.src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

# Overwrite the following params to match the values in cvpr_ds_config.py
# These produce much better results in evaluation
cfg.LOFTR.COARSE.TEMP_BUG_FIX = False
cfg.LOFTR.MATCH_COARSE.SKH_PREFILTER = True
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4
#####################

cfg.TRAINER.CANONICAL_LR = 2e-4 #2e-3 #8e-3
# true_lr is canonical_lr scaled by the true batch size (that depends on number of gpus)
#cfg.TRAINER.TRUE_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

cfg.TRAINER.PRETRAINED_MODEL_WEIGHTS = "models/LoFTR/weights/"