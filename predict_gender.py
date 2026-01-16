import cv2
import sqlite3
import datetime
from ultralytics import YOLO
from deepface import DeepFace
import logging

# --- Configuration ---
DB_NAME = "demographics.db"
MODEL_NAME = "yolov8n.pt"  # Use 'yolov8n.pt' for speed
CONFIDENCE_THRESHOLD = 0.5

# Setup Logging to suppress DeepFace warnings
logging.getLogger('deepface').setLevel(logging.ERROR)

def init_db():
    """Initialize SQLite database to store detection data."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            track_id INTEGER,
            gender TEXT,
            age INTEGER,
            confidence REAL
        )
    ''')
    conn.commit()
    return conn

def log_to_db(conn, track_id, gender, age, conf):
    """Insert a new detection record into the database."""
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO detections (timestamp, track_id, gender, age, confidence) VALUES (?, ?, ?, ?, ?)",
        (timestamp, track_id, gender, age, conf)
    )
    conn.commit()
    print(f"Logged: ID {track_id} | {gender} | {age}")

def analyze_person(img_crop):
    """
    Run DeepFace analysis on a cropped image of a person.
    enforce_detection=False allows it to guess even if a clear face isn't perfectly found.
    """
    try:
        # DeepFace.analyze returns a list of dicts. We take the first one.
        # We use a faster detector backend like 'opencv' or 'ssd' for speed, 
        # or 'retinaface' for accuracy (slower).
        # weight path = /Users/usr/.deepface/weights/gender_model_weights.h5 and age_model_weights.h5 538 MB each
        results = DeepFace.analyze(
            img_path=img_crop, 
            actions=['age', 'gender'], 
            detector_backend='opencv', 
            enforce_detection=False,
            silent=True
        )
        if results:
            result = results[0]
            return result['dominant_gender'], result['age']
    except Exception as e:
        # Sometimes analysis fails if the image is too blurry or small
        pass
    return "Unknown", 0

def main():
    # 1. Initialize Database
    conn = init_db()
    
    # 2. Load YOLOv8 Model
    model = YOLO(MODEL_NAME)
    
    # 3. Open Webcam
    cap = cv2.VideoCapture(0) # Change to 1 or video file path if needed
    
    # Cache to store data for tracked IDs so we don't re-analyze them every frame
    # Format: { track_id: {'gender': 'Man', 'age': 25} }
    track_history = {}

    print("Starting Detection & Collection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run YOLOv8 Tracking
        # persist=True is crucial for keeping the same ID for the same person across frames
        results = model.track(frame, device='mps', persist=True, verbose=False, classes=[0]) # class 0 is person

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box

                # If this is a new person we haven't analyzed yet...
                if track_id not in track_history:
                    # 5. Crop the person
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Ensure crop is valid size
                    if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                        # 6. Analyze Age/Gender (Heavy Operation)
                        gender, age = analyze_person(person_crop)
                        
                        # Store in history
                        track_history[track_id] = {'gender': gender, 'age': age}
                        
                        # 7. Log to Database
                        log_to_db(conn, track_id, gender, age, float(conf))
                
                # Retrieve info from history for display
                info = track_history.get(track_id, {'gender': 'Detecting...', 'age': ''})
                
                # Draw Bounding Box & Label
                label = f"ID: {track_id} {info['gender']} {info['age']}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Demographics Collector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("Data collection stopped. Saved to demographics.db")

if __name__ == "__main__":
    main()