import argparse
import cv2
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

def main():
    parser = argparse.ArgumentParser(description='MMPose Video Inference and Visualization Demo')
    parser.add_argument('video_input', help='Path to the input video file')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('video_output', help='Path to save the output video file')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bounding boxes on the output video.')

    args = parser.parse_args()

    # Initialize the MMPoseInferencer
    inferencer = MMPoseInferencer(
        pose2d=args.config,
        pose2d_weights=args.checkpoint,
        device=args.device
    )

    # Initialize video reader
    cap = cv2.VideoCapture(args.video_input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {args.video_input}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.video_output, fourcc, fps, (width, height))

    # Process video frame by frame
    # The inferencer returns a generator, which we can iterate through.
    results_generator = inferencer(args.video_input, return_vis=True, draw_bbox=args.draw_bbox)

    print(f"Processing video: {args.video_input}")
    for result in tqdm(results_generator, total=total_frames):
        # 'result' is a dictionary containing the visualized frame in RGB format
        visualized_frame_rgb = result['visualization'][0]
        # Convert RGB to BGR for OpenCV to save correctly
        visualized_frame_bgr = cv2.cvtColor(visualized_frame_rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(visualized_frame_bgr)

    # Release resources
    cap.release()
    video_writer.release()

    print(f"\nOutput video successfully saved to: {args.video_output}")

if __name__ == '__main__':
    main()
