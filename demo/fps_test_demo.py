import argparse
import time
import cv2
import torch
from tqdm import tqdm

from mmpose.apis import MMPoseInferencer

def main():
    parser = argparse.ArgumentParser(description='MMPose FPS Test Demo')
    parser.add_argument('video_input', help='Path to the input video file')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference')
    args = parser.parse_args()

    # Initialize the inferencer
    inferencer = MMPoseInferencer(
        pose2d=args.config,
        pose2d_weights=args.checkpoint,
        device=args.device
    )

    # Open video file
    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_input}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video_input}")
    print(f"Total frames: {total_frames}")
    print("Starting FPS measurement...")

    # --- Warm-up run ---
    # To get a more accurate measurement, we run inference once before timing.
    print("Warming up the GPU...")
    ret, frame = cap.read()
    if ret:
        # The inferencer might be a generator, so we iterate through it.
        for _ in inferencer(frame, return_vis=False):
            pass
    torch.cuda.synchronize() # Wait for the GPU to finish
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video
    print("Warm-up complete.")

    # --- Timed run ---
    start_time = time.time()
    frames_processed = 0

    # Create a progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference. We don't need to store the results.
        for _ in inferencer(frame, return_vis=False):
            pass
        
        frames_processed += 1
        pbar.update(1)

    # Wait for all GPU tasks to complete before stopping the timer
    torch.cuda.synchronize()
    end_time = time.time()
    
    pbar.close()
    cap.release()

    total_time = end_time - start_time
    fps = frames_processed / total_time if total_time > 0 else 0

    print("\n--- FPS Measurement Results ---")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")

if __name__ == '__main__':
    main()
