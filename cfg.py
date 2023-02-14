from mmcv import Config
from mmdet.apis import set_random_seed
from dataset import XMLCustomDataset

cfg = Config.fromfile('mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco_softNMS(creator).py')
print(f"Default Config:\n{cfg.pretty_text}")

cfg.dataset_type = 'XMLCustomDataset'
cfg.data_root = 'input/data_root/'

cfg.data.test.type = 'XMLCustomDataset'
cfg.data.test.data_root = 'input/data_root/'
cfg.data.test.ann_file = 'dataset/ImageSets/Main/val.txt'
cfg.data.test.img_prefix = 'dataset/'

cfg.data.train.type = 'XMLCustomDataset'
cfg.data.train.data_root = 'input/data_root/'
cfg.data.train.ann_file = 'dataset/ImageSets/Main/train.txt'
cfg.data.train.img_prefix = 'dataset/'

cfg.data.val.type = 'XMLCustomDataset'
cfg.data.val.data_root = 'input/data_root/'
cfg.data.val.ann_file = 'dataset/ImageSets/Main/val.txt'
cfg.data.val.img_prefix = 'dataset/'

cfg.data.samples_per_gpu = 10
cfg.model.bbox_head.num_classes = 6
cfg.load_from = 'checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'

cfg.optimizer.lr = 0.008 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 5

cfg.work_dir = 'outputs'
cfg.evaluation.metric = 'mAP'
cfg.evaluation.save_best = 'mAP'
cfg.evaluation.interval = 1
cfg.checkpoint_config.interval = 15

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_epochs = 300

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')
    ]

print('#'*50)
print(f'Config:\n{cfg.pretty_text}')

