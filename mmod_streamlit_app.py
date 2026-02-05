'''
Mock up app for testing multi-model object detection
'''
# TODO: train a new model for 3 classes: harness, hardhat, smoking

import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time

# Set page configuration
st.set_page_config(page_title="Multi-Model Object Detection", layout="wide", page_icon="📸")

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Dark theme tweaks */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Upload area styling */
    [data-testid="stFileUploader"] {
        border: 1px dashed #4B5563;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Right panel scrolling */
    .detection-list {
        height: 80vh;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    /* Detection card styling */
    .detection-card {
        background-color: #262730;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 10px;
        border: 1px solid #374151;
    }
    
    .detection-card img {
        border-radius: 4px;
        width: 80px;
        height: 50px;
        object-fit: cover;
    }
    
    .detection-info {
        flex: 1;
    }
    
    .detection-class {
        font-weight: bold;
        font-size: 0.9rem;
        color: #E5E7EB;
        margin-bottom: 2px;
    }
    
    .detection-meta {
        font-size: 0.75rem;
        color: #9CA3AF;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.title("UPLOAD MEDIA")
    st.markdown("Drag & drop or click to upload Image/Video File")
    st.caption("Supported formats: JPG, PNG, MP4, AVI")

with col_header_2:
    st.header("DETECTED OBJECTS")
    # st.toggle("Detecting Results", value=True)

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        # Check if files exist to avoid silent failures
        models = {}
        paths = {
            "Harness": "mock_models/harness-best-train.pt",
            "PPE": "mock_models/ppe-best.pt",
            "Smoking": "mock_models/smoke-best-train.pt"
        }
        for name, path in paths.items():
            if os.path.exists(path):
                models[name] = YOLO(path)
            else:
                st.error(f"Model file not found: {path}")
                return None, None, None
        return models["Harness"], models["PPE"], models["Smoking"]
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

harness_model, hardhat_model, smoking_model = load_models()

if not all([harness_model, hardhat_model, smoking_model]):
    st.warning("One or more models failed to load. Please check the model paths.")
    st.stop()

# --- Sidebar Controls (Hidden/Minimal per design, but keeping for functionality) ---
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Model Config")
    conf_harness = st.slider("Harness Confidence", 0.0, 1.0, 0.25, 0.05)
    conf_ppe = st.slider("PPE Confidence", 0.0, 1.0, 0.25, 0.05)
    conf_smoking = st.slider("Smoking Confidence", 0.0, 1.0, 0.25, 0.05)
    
    conf_thresholds = {
        "Harness": conf_harness,
        "PPE": conf_ppe,
        "Smoking": conf_smoking
    }
    
    st.subheader("Visual Settings")
    color_harness = st.color_picker("Harness Color", "#00FF00")
    color_ppe = st.color_picker("PPE Color", "#FFFF00")
    color_smoking = st.color_picker("Smoking Color", "#FF0000")
    
    # Helper to convert hex to BGR
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    color_map = {
        "Harness": hex_to_bgr(color_harness),
        "PPE": hex_to_bgr(color_ppe),
        "Smoking": hex_to_bgr(color_smoking)
    }

    st.divider()
    enable_harness = st.checkbox("Detect Harness", value=True)
    enable_ppe = st.checkbox("Detect PPE (Hardhat/Vest)", value=True)
    enable_smoking = st.checkbox("Detect Smoking", value=True)

# --- Helper Functions ---
def process_frame(frame, conf_dict):
    annotated_frame = frame.copy()
    all_detections = []

    def apply_model(model, classes_filter, model_name):
        c_conf = conf_dict.get(model_name, 0.25)
        # Use color from global map (sidebar)
        color = color_map.get(model_name, (0, 255, 0))
        
        results = model(frame,
                        verbose=False,
                        # device="mps", 
                        conf=c_conf,
                        classes=classes_filter,
                        iou=0.3, # keep IOU setting
                        )
        for r in results:
            # We do NOT use r.plot() here to keep control of colors
            # nonlocal annotated_frame
            # annotated_frame = r.plot(img=annotated_frame)
            
            for box in r.boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                # Draw Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label Background
                label = f"{class_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                c2 = (x1 + t_size[0], y1 - t_size[1] - 3)
                cv2.rectangle(annotated_frame, (x1, y1), c2, color, -1) # Filled
                
                # Draw Text (White)
                cv2.putText(annotated_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                all_detections.append({
                    "model": model_name,
                    "class": class_name,
                    "conf": conf,
                    "box": box.xyxy[0].cpu().numpy()
                })
        return

    if enable_harness: apply_model(harness_model, [1], "Harness")
    if enable_ppe: apply_model(hardhat_model, [0], "PPE")
    if enable_smoking: apply_model(smoking_model, [0], "Smoking")

    return annotated_frame, all_detections

def render_detection_card(image_bgr, detection):
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Create valid filename-safe string (not used for file but for ID if needed)
    # We will use st.image for now, but to put it in a custom HTML card we'd need base64
    # For simplicity, we use st.container with columns
    
    with st.container():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(img_rgb, width="stretch") # or width="stretch" (deprecated warning fix)
        with c2:
            st.markdown(f"**{detection['class']}**")
            st.markdown(f"<span style='color:grey; font-size:12px;'>{detection['model']} • {detection['conf']:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:grey; font-size:12px;'>Detected at {time.strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)
        st.divider()

# --- Main Layout ---
col_main, col_sidebar = st.columns([3, 1])

with col_main:
    uploaded_file = st.file_uploader("Upload Media", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], label_visibility="collapsed")
    main_display = st.empty()

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type in ['jpg', 'jpeg', 'png']:
            # Image Processing
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            with st.spinner('Scanning image...'):
                annotated_frame, detections = process_frame(image, conf_thresholds)
                # Display Result
                main_display.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), width="stretch")
                
            # Update sidebar
            with col_sidebar:
                st.markdown("### Detected Objects")
                if detections:
                    for det in detections:
                        render_detection_card(annotated_frame, det)
                else:
                    st.info("No objects detected")

        elif file_type in ['mp4', 'mov', 'avi']:
            # Video Processing
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            cap = cv2.VideoCapture(video_path)
            
            # Button to start (optional, maybe auto-start?)
            if st.button("Start Analysis", key="start_btn"):
                
                # Container for list in sidebar
                list_container = col_sidebar.container()
                history_data = [] # Store recent detections to render
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    annotated_frame, detections = process_frame(frame, conf_thresholds)

                    
                    # Update Main Display
                    main_display.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
                    
                    # Update Sidebar if detection found
                    if detections:
                        with list_container:
                            # We just render the new ones on top?
                            # Streamlit appends by default in a container loop.
                            # To simulate a scrollable list that updates, we can just append.
                            # The user wants a 1 sec delay per frame detected.
                            
                            for det in detections:
                                # Use full frame as requested
                                render_detection_card(annotated_frame, det)

                            
                        # Delay as requested
                        # time.sleep(1)

                cap.release()
                st.success("Analysis Complete")
