import cv2
from ultralytics import YOLO
import torch

def run_yolo_mac():
    # Check if MPS is available to ensure we aren't falling back to CPU silently
    if torch.backends.mps.is_available():
        device = 'mps'
        print("Success: MPS (Metal Performance Shaders) acceleration is available.")
    else:
        device = 'cpu'
        print("Warning: MPS not found. Falling back to CPU.")

    # 1. Load the model
    model = YOLO(model='yolo_models/yolov8n.pt')

    # 2. Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Note: On some Macs, you might need to set the video capture backend explicitly if 0 fails:
    # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting YOLOv8 on {device.upper()}...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 3. Run Inference
        # device='mps' targets the Apple Silicon GPU neural engine
        results = model(frame, device=device, verbose=False)
        print("Shape: ", frame.shape)

        # 4. Visualize Results
        annotated_frame = results[0].plot()

        # 5. Display the frame
        cv2.imshow("YOLOv8 - macOS (MPS)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_mac()