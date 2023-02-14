from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from cfg import cfg
from utils import get_annots, draw_bboxes
from tqdm import tqdm

import argparse
import mmcv
import glob as glob
import os
import numpy as np

# Contruct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', default='input/inference_data_1',
    help='path to the input data'
)
parser.add_argument(
    '-w', '--weights', default='outputs/MDPI-5_yolof_r50_c5_8x8_1x_coco/best_mAP_epoch_15.pth', help='weight file path'
)
parser.add_argument(
    '-t', '--threshold', default=0, type=float,
    help='detection threshold for bounding box visualization'
)
args = vars(parser.parse_args())    

# Create an output directory.
output_dir = os.path.join('outputs', 'inference_outputs') 
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)
colors = np.random.uniform(0, 255, size=(7, 3))
CLASSES = (
        'crazing', 'inclusion',
        'patches', 'pitted_surface',
        'rolled-in_scale', 'scratches'
        )

# Build the model.
model = init_detector(cfg, args['weights'])

image_paths = sorted(glob.glob(f"{args['input']}/*.jpg"))
xml_paths = sorted(glob.glob(f"{args['input']}/*.xml"))

for i, (image_path, xml_path) in tqdm(
    enumerate(zip(image_paths, xml_paths)), 
    total=len(image_paths)
):
    image = mmcv.imread(image_path)
    names, bboxes = get_annots(xml_path)
    annotated_image = draw_bboxes(image, names, bboxes, colors, CLASSES)
    # Carry out the inference.
    result = inference_detector(model, image)
    # Draw the annotations on the ground truth annotated images.
    frame = model.show_result(annotated_image, result, score_thr=args['threshold'])
    # mmcv.imshow(frame)
    # Initialize a file name to save the reuslt.
    save_name = f"{image_path.split(os.path.sep)[-1].split('.')[0]}"
    mmcv.imwrite(frame, os.path.join(output_dir, save_name+'.jpg'))