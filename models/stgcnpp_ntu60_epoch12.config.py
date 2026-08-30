# ST-GCN++ on NTU60 (2D COCO-17 skeletons), adapted for a single 6 GB GPU.
#
# The upstream config assumes 8 GPUs x 16 samples = an effective batch of 128,
# and its lr=0.1 is set for that. Running it unchanged on one GPU would train at
# 8x the intended learning rate per sample and diverge, so auto_scale_lr is
# switched on to rescale lr by the real batch size.
#
# Why NTU60 first: it is the only skeleton dataset with pretrained ST-GCN++
# weights, and several of its classes line up with the target ones —
# "playing with phone/tablet", "writing", "reading", "punching", "walking",
# "sitting down", "standing up". Training here validates the whole path and
# produces a backbone to fine-tune on collected data later.
#
# NOTE on licensing: NTU RGB+D is licensed for academic research only. This run
# is research/validation; a deployed model must not depend on these weights.
#
#   python mmaction2/tools/train.py configs/stgcnpp_ntu60_1gpu.py
_base_ = ('../mmaction2/configs/skeleton/stgcnpp/'
          'stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py')

# ST-GCN++ is small (~1.4M params); the memory goes on the 100-frame clips.
train_dataloader = dict(batch_size=16, num_workers=4,
                        persistent_workers=True)
val_dataloader = dict(batch_size=16, num_workers=4,
                      persistent_workers=True)
test_dataloader = dict(batch_size=1, num_workers=4)

# Rescale lr from the base 8x16 batch to whatever we actually run.
auto_scale_lr = dict(enable=True, base_batch_size=128)

# Mixed precision: roughly halves activation memory and speeds up training,
# which matters on a laptop GPU.
optim_wrapper = dict(type='AmpOptimWrapper')

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='acc/top1'),
    logger=dict(interval=100))

work_dir = 'work_dirs/stgcnpp_ntu60_1gpu'
