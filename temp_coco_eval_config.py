# Inherit from the original ProbPose config file
_base_ = './configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py'

# Forcefully remove the original dataloader and evaluator settings
# to prevent incorrect merging of dictionaries.
del _base_.val_dataloader
del _base_.test_dataloader
del _base_.val_evaluator
del _base_.test_evaluator

# ============== DATASET SETTINGS =================
data_root = 'C:/Users/mulso/Desktop/aislab/seongmo_anapa'
ann_file = 'annotations/person_keypoints_val2017.json'

# Get the validation pipeline from the base config
val_pipeline = _base_.val_pipeline

# Define a clean dataset configuration for COCO validation
coco_val_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode='topdown',
    ann_file=ann_file,
    data_prefix=dict(img='val2017/'),
    test_mode=True,
    pipeline=val_pipeline
)

# Define the dataloader for validation/testing
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=coco_val_dataset
)
val_dataloader = test_dataloader

# Define the evaluator for validation/testing
test_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/{ann_file}'
)
val_evaluator = test_evaluator
