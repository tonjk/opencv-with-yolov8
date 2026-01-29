# generate python script using yolo to detect objects in an image and live feed from camera
import cv2
from ultralytics import YOLO
import torch
from pathlib import Path
import os

class PersonDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = self.model_path.split(".")[0].split("/")[-1]
        self.model = self._load_model()
        self.device = self._setup_device()

    def _load_model(self):
        return YOLO(self.model_path)

    def _setup_device(self):
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Success: MPS (Metal Performance Shaders) acceleration is available.")
        else:
            device = 'cpu'
            print("Warning: MPS not found. Falling back to CPU.")

        return device

    def detect_from_image(self, image_path: str, save_folder_path: str) -> None:
        """
        Detects objects from an image and saves the annotated image into a folder.

        Args:
            image_path (str): The path to the image file.
            save_folder_path (str): The path to the folder where the annotated image will be saved.

        Returns:
            None
        """
        # Load and reshape image to 640x640
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (640, 640))
        
        # detect and draw bounding box and save into folder
        results = self.model(image, conf=0.1, classes=[0], device=self.device, save=False, save_txt=False, save_conf=False, project="my-projects", name="run-test", exist_ok=True)
        annotated_image = results[0].plot()
        file_path = save_folder_path + "/" + image_path.split("/")[-1].split(".")[0] + "_" + self.model_name + ".jpg"
        save_path = Path(file_path)
        cv2.imwrite(save_path, annotated_image)
        print(f"Saved {save_path}")

    def open_cam(self):
        cap = cv2.VideoCapture(0)
        
        # Note: On some Macs, you might need to set the video capture backend explicitly if 0 fails:
        # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print(f"Starting {self.model_name} on {self.device.upper()}...")

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 3. Run Inference
            # device='mps' targets the Apple Silicon GPU neural engine
            results = self.model(frame, device=self.device, verbose=False, save=False, conf=0.65, project="my-projects", name="run-test", exist_ok=True, classes=[0]) # set project and name to specific folder

            # 4. Visualize Results
            annotated_frame = results[0].plot()

            # 5. Display the frame
            cv2.imshow(f"{self.model_name} - macOS (MPS)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    test_type = input("Enter 1 for image or 2 for live feed: ")

    if test_type == "1":
        folder_model_path = "yolo_models" # yolo_models , other_models
        model_names = [f for f in os.listdir(folder_model_path) if f.endswith(".pt")]
        print("Available models:", model_names)
        for model_name in model_names:
            yolo_model = PersonDetector(model_path=f"{folder_model_path}/{model_name}")
            yolo_model.detect_from_image(image_path="images/Thailand-Trains-4.jpg", save_folder_path="my-projects/img_res")
    
    elif test_type == "2":
        yolo_model = PersonDetector(model_path="other_models/crowd_best.pt")
        yolo_model.open_cam()

    else:
        print("Invalid input. Please enter 1 or 2.")