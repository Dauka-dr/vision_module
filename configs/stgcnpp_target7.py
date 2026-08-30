# Fine-tune the NTU60 backbone onto the 7 target classes that NTU actually covers.
#
# The remaining four target classes — "sitting", "standing", "sitting attentive",
# "sleeping" — are deliberately absent. NTU is built from transitions ("sit down",
# "stand up") and contains no sustained states at all, so they cannot be learned
# from it at any sample count; they are handled geometrically instead.
#
#   python mmaction2/tools/train.py configs/stgcnpp_target7.py
_base_ = './stgcnpp_ntu60_1gpu.py'

CLASSES = ['phone', 'writing', 'raising_hand', 'talking', 'walking',
           'running', 'fighting']

# Start from the 60-class run: the graph convolutions have already learned what
# human motion looks like, which is what transfers. Only the classification head
# is rebuilt, since 60 outputs become 7.
load_from = 'models/stgcnpp_ntu60_epoch12.pth'
model = dict(cls_head=dict(num_classes=len(CLASSES)))

ann_file = 'dataset/annotations/target_7cls.pkl'
train_dataloader = dict(
    dataset=dict(dataset=dict(ann_file=ann_file, split='xsub_train')))
val_dataloader = dict(dataset=dict(ann_file=ann_file, split='xsub_val'))
test_dataloader = dict(dataset=dict(ann_file=ann_file, split='xsub_val'))

# 9.4k clips against NTU60's 40k, and starting from trained weights rather than
# noise — far fewer epochs are needed, and a high lr would destroy what the
# backbone already knows.
train_cfg = dict(max_epochs=10)
param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=0, T_max=10, by_epoch=True,
         convert_to_iter_based=True)
]
optim_wrapper = dict(optimizer=dict(lr=0.01))

# Per-class numbers matter more than the average here: the whole point is to see
# whether "phone" and "writing" get confused, which an aggregate score hides.
val_evaluator = [dict(type='AccMetric',
                      metric_list=('top_k_accuracy', 'mean_class_accuracy'))]
test_evaluator = val_evaluator

work_dir = 'work_dirs/stgcnpp_target7'
