import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import tempfile
import os
import time
import torch 
# NOTE: Removed the 'import dill' line to prevent the ModuleNotFoundError crash.

# --- Configuration ---
# ⚠️ ACTION REQUIRED: Ensure 'best.pt' is in the root of your repository
MODEL_PATH = 'best.pt'
TITLE = "AI License Plate Detector 🚗"
CONFIDENCE_DEFAULT = 50 
DEFAULT_INFERENCE_SIZE = 480 

# --- Helper Functions ---

@st.cache_resource 
def load_yolo_model(path):
    """
    Loads the YOLOv8 model using PyTorch's torch.load() to safely handle 
    cross-OS dependency issues (like map_location='cpu') before initializing YOLO.
    """
    st.info("Loading model... Please wait.")
    try:
        # 1. Use torch.load() to load the weights file directly with the cross-OS fix.
        # This tells PyTorch to load the model's weights onto the CPU and ignore 
        # any platform-specific dependencies (like DLLs) saved in the checkpoint.
        weights = torch.load(path, map_location='cpu')

        # 2. Save the loaded weights dictionary to a new temporary file.
        # This strips the checkpoint of potentially corrupting original metadata/paths.
        temp_path = os.path.join(tempfile.gettempdir(), 'fixed_best.pt')
        torch.save(weights, temp_path)

        # 3. Initialize the YOLO model using the CLEANED weights file path.
        model = YOLO(temp_path, task='detect')
        return model
        
    except Exception as e:
        st.error(f"Failed to load model from '{path}'.")
        st.error("There is a deeper incompatibility between the model's saved state and the current environment. This error is usually final.")
        st.error(f"Final Error: {e}")
        return None


def draw_boxes_and_confidence(image_np, results, scale_factor=1.0):
    """
    Draws bounding boxes and confidence scores (only) on the image/frame, 
    and applies a scale factor to resize bounding boxes back to the original frame size.
    """
    # Convert NumPy array to PIL Image for drawing (handling BGR/RGB)
    is_bgr = len(image_np.shape) == 3 and image_np.shape[2] == 3 and image_np.dtype == np.uint8 and image_np[0, 0, 0] != image_np[0, 0, 2]
    
    if is_bgr:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    else:
        image_pil = Image.fromarray(image_np)
        
    draw = ImageDraw.Draw(image_pil)
    
    # Font scaling for visual consistency
    try:
        font = ImageFont.truetype("arial.ttf", int(25 * scale_factor)) 
    except IOError:
        font = ImageFont.load_default()

    for r in results:
        if r.boxes is None:
            continue
            
        for box in r.boxes:
            # Scale coordinates back up to original frame size
            x1, y1, x2, y2 = [int(val * scale_factor) for val in box.xyxy[0]]
            confidence = float(box.conf[0]) * 100 
            
            # Define Colors
            box_color = (0, 255, 0) # Green
            text_color = (255, 255, 255) # White text
            fill_color = (0, 128, 0) # Darker green for text background

            # 1. Draw rectangle for bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            # 2. Draw confidence text background
            text = f"{confidence:.1f}%"
            
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(text, font) 
            
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 5, y1], fill=fill_color)
            draw.text((x1 + 2, y1 - text_h - 3), text, font=font, fill=text_color)
            
    # Return the annotated PIL Image (RGB) back to an RGB NumPy array
    return np.array(image_pil) 


# --- Video Processing Function ---

def process_video(uploaded_file, model, confidence, inference_size):
    """Handles video upload, frame-by-frame processing, and smooth live display."""
    col_original, col_results = st.columns(2)
    
    if uploaded_file is not None:
        st.toast('Video uploaded! Starting processing...', icon='📤')
        
        # 1. Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        # 2. Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file.")
            os.unlink(video_path)
            return

        # Get video information
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the scaling factor for Bounding Box coordinates
        scale_factor = original_width / inference_size
        
        # Display placeholders
        with col_original:
            st.subheader("Original Video Stream")
            st.video(uploaded_file, format="video/mp4", start_time=0) 
            
        with col_results:
            st.subheader(f"Detection Results (Inference: {inference_size}px)")
            detection_frame_placeholder = st.empty()
            fps_text = st.empty() 

        
        progress_bar = st.progress(0, text="Processing frames...")
        frame_count = 0
        
        # 3. Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            start_time = time.time() 

            # OPTIMIZATION: Resize the frame before inference
            resized_frame = cv2.resize(frame, (inference_size, int(original_height * (inference_size / original_width))))
            
            # Run YOLOv8 detection on the SMALLER frame
            results = model.predict(
                source=resized_frame, 
                conf=confidence, 
                imgsz=inference_size, 
                save=False, show=False, verbose=False
            )
            
            # Draw custom boxes and confidence, scaling the boxes back to the original size
            annotated_frame_rgb_np = draw_boxes_and_confidence(frame.copy(), results, scale_factor=scale_factor)
            
            # Update the placeholder for smooth display
            detection_frame_placeholder.image(
                annotated_frame_rgb_np, 
                caption=f"Frame {frame_count} / {total_frames}", 
                channels="RGB", 
                use_container_width=True
            )
            
            # Update FPS and progress
            end_time = time.time()
            current_fps = 1 / (end_time - start_time)
            fps_text.info(f"⚡️ Processing Speed: **{current_fps:.2f} FPS** | Detections: {len(results[0].boxes)}")
            
            if total_frames > 0:
                progress_bar.progress(frame_count / total_frames, text=f"Processing frames... {frame_count}/{total_frames}")

        # 4. Cleanup
        cap.release()
        os.unlink(video_path) 
        progress_bar.empty()
        fps_text.empty()
        st.success(f"✅ Video processing complete! Total frames processed: {frame_count}")
    
# --- Image Processing Function ---

def process_image(uploaded_file, model, confidence, inference_size):
    """Handles image upload, processing, and display."""
    col_original, col_results = st.columns(2)

    if uploaded_file is not None:
        st.toast('Image uploaded! Processing...', icon='📤')
        image = Image.open(uploaded_file).convert("RGB") 

        with col_original:
            st.subheader("Original Image")
            st.image(image, use_container_width=True, caption="Uploaded Image")
        
        # Automatic Prediction
        image_np_rgb = np.array(image)
        original_width, original_height = image.size
        scale_factor = original_width / inference_size
        
        # OPTIMIZATION: Resize the image before inference
        resized_image = cv2.resize(image_np_rgb, (inference_size, int(original_height * (inference_size / original_width))))

        # Run YOLOv8 detection on the SMALLER image
        results = model.predict(
            source=resized_image, 
            conf=confidence, 
            imgsz=inference_size,
            save=False, show=False, verbose=False
        )

        # Process and Display Results
        with col_results:
            st.subheader("Detection Results")
            if results and results[0].boxes and len(results[0].boxes) > 0:
                st.toast(f'✅ Found {len(results[0].boxes)} license plate(s)!', icon='🎯')
                
                # Draw custom boxes and confidence, scaling back to original size
                annotated_image_rgb_np = draw_boxes_and_confidence(image_np_rgb.copy(), results, scale_factor=scale_factor)
                
                st.image(annotated_image_rgb_np, use_container_width=True, caption="Detected License Plate(s)")
                    
                st.success(f"Detected {len(results[0].boxes)} license plate(s) above {int(confidence*100)}% confidence.")

            else:
                st.warning("No license plates detected at the current confidence level.")
                st.image(image, use_container_width=True, caption="No detections found")

# --- Main App UI Logic ---

def main():
    """The main function to run the Streamlit application."""
    
    st.set_page_config(
        page_title=TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional look
    st.markdown("""
        <style>
        .stApp {
            background-color: #0d1117; 
            color: #c9d1d9;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #58a6ff;
        }
        /* Style adjustments for cleaner UI */
        </style>
    """, unsafe_allow_html=True)

    st.title(TITLE)
    st.markdown("---") 
    st.subheader("Upload a media file for instant license plate detection.")
    st.markdown("*(Powered by YOLOv8)*")

    # --- Sidebar Configuration ---
    st.sidebar.header("⚙️ Settings")
    
    source_type = st.sidebar.radio("Select Source Type:", ('Image', 'Video'))
    
    confidence = st.sidebar.slider(
        "Detection Confidence Threshold (%)", min_value=10, max_value=100, value=CONFIDENCE_DEFAULT, step=5
    ) / 100.0
    st.sidebar.info(f"Detections below {int(confidence*100)}% will be filtered out.")
    
    inference_size = st.sidebar.slider(
        "Inference Resolution (Faster: Low | Accurate: High)", min_value=320, max_value=1280, value=DEFAULT_INFERENCE_SIZE, step=32
    )

    # Model Loading (Call with the fix)
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        st.error("Application halted.")
        return 
        
    st.sidebar.success(f"Model: `{MODEL_PATH}` loaded successfully.")
    
    # --- Main Content: File Uploader ---
    
    if source_type == 'Image':
        uploaded_file = st.file_uploader("🖼️ Choose an image file:", type=['jpg', 'jpeg', 'png'])
        process_image(uploaded_file, model, confidence, inference_size)
        
    elif source_type == 'Video':
        uploaded_file = st.file_uploader("📹 Choose a video file (MP4, MOV, AVI):", type=['mp4', 'avi', 'mov'])
        process_video(uploaded_file, model, confidence, inference_size)


if __name__ == "__main__":
    main()
