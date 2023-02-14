from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from cfg import cfg

import argparse
import mmcv
import time
import cv2
import os

# Contruct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', default='input/inference_data/video_1.mp4',
    help='path to the input file'
)
parser.add_argument(
    '-w', '--weights', required=True, help='weight file path'
)
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold for bounding box visualization'
)
args = vars(parser.parse_args())

# Create an output directory.
output_dir = os.path.join('outputs', 'inference_outputs') 
os.makedirs(output_dir, exist_ok=True)

# Build the model.
model = init_detector(cfg, args['weights'])

cap = mmcv.VideoReader(args['input'])
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    os.path.join(output_dir, save_name+'.mp4'), 
    fourcc, 
    cap.fps,
    (cap.width, cap.height)
)

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
for frame in mmcv.track_iter_progress(cap):
    # Increment frame count.
    frame_count += 1
    start_time = time.time()# Forward pass start time.
    result = inference_detector(model, frame)
    end_time = time.time() # Forward pass end time.
    # Get the fps.
    fps = 1 / (end_time - start_time)
    # Add fps to total fps.
    total_fps += fps
    show_result = model.show_result(frame, result, score_thr=args['threshold'])
    # Write the FPS on the current frame.
    cv2.putText(
        show_result, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2, cv2.LINE_AA
    )
    mmcv.imshow(show_result, 'Result', wait_time=1)
    out.write(show_result)

# Release VideoCapture()
out.release()
# Close all frames and video windows
cv2.destroyAllWindows()
# Calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")