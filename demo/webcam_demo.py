import argparse
import cv2
import time
from mmpose.apis import MMPoseInferencer

def main():
    parser = argparse.ArgumentParser(description='MMPose Real-time Webcam Demo')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device to use for inference')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bounding boxes.')
    parser.add_argument('--draw-heatmap', action='store_true', help='Draw heatmaps.(수정으로 추가한 부분)')
    args = parser.parse_args()

    # Initialize the MMPoseInferencer
    inferencer = MMPoseInferencer(
        pose2d=args.config,
        pose2d_weights=args.checkpoint,
        device=args.device
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam ID
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    prev_time = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Run inference and get visualization
        # The inferencer returns a generator; we get the first result.
        result_generator = inferencer(frame, return_vis=True, draw_bbox=args.draw_bbox, draw_heatmap=args.draw_heatmap)
        result = next(result_generator)
        visualized_frame = result['visualization'][0]

        # Convert RGB to BGR for correct color display
        visualized_frame = cv2.cvtColor(visualized_frame, cv2.COLOR_RGB2BGR)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(visualized_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('MMPose Real-time Demo', visualized_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
