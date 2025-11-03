import argparse
import cv2
import os
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

def main():
    parser = argparse.ArgumentParser(description='MMPose Video Inference for a folder of videos.')
    parser.add_argument('input_folder', help='Path to the folder containing input videos.')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('output_folder', help='Path to the folder to save output videos.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bounding boxes on the output video.')

    args = parser.parse_args()

    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Initialize the MMPoseInferencer
    inferencer = MMPoseInferencer(
        pose2d=args.config,
        pose2d_weights=args.checkpoint,
        device=args.device
    )

    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Find all video files in the input folder
    video_files = [f for f in os.listdir(args.input_folder) if os.path.splitext(f)[1].lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {args.input_folder}")
        return

    for video_file in video_files:
        input_video_path = os.path.join(args.input_folder, video_file)
        output_video_path = os.path.join(args.output_folder, video_file)

        # Initialize video reader
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {input_video_path}")
            continue

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process video frame by frame
        results_generator = inferencer(input_video_path, return_vis=True, draw_bbox=args.draw_bbox)

        print(f"Processing video: {input_video_path}")
        for result in tqdm(results_generator, total=total_frames):
            visualized_frame_rgb = result['visualization'][0]
            visualized_frame_bgr = cv2.cvtColor(visualized_frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(visualized_frame_bgr)

        # Release resources
        cap.release()
        video_writer.release()

        print(f"\nOutput video successfully saved to: {output_video_path}")

if __name__ == '__main__':
    main()
