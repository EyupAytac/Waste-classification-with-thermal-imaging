import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd  # Added for Excel reading
import sys

# Import your local project files
import config
from model import ThermalMaterialClassifier


# ---------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------
def load_model(model_path, device):
    print(f"[System] Loading model from {model_path}...")
    model = ThermalMaterialClassifier(
        in_channels=1,
        num_classes=3,
        spatial_dim=512,
        temporal_hidden=256,
        temporal_layers=2,
        num_heads=8,
        dropout=0.2
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------------------------------------------------
# MAIN PROCESSING LOOP
# ---------------------------------------------------------
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import pandas as pd
import sys
import config
from model import ThermalMaterialClassifier

# ---------------------------------------------------------
# CONSTANTS 
# ---------------------------------------------------------
SKIP_SEC = 5.0
WINDOW_SEC = 20.0 
MAX_FRAMES = 100
OUT_SIZE = 224

def load_model(model_path, device):
    print(f"[System] Loading model from {model_path}...")
    model = ThermalMaterialClassifier(
        in_channels=1,
        num_classes=3,
        spatial_dim=512,
        temporal_hidden=256,
        temporal_layers=2,
        num_heads=8,
        dropout=0.2
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_live_inference_strict(video_path, model, device, threshold=0.75):
    # 1. Setup
    roi = (440,120,500,200)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    # Calculate Frame Indices strictly based on your function
    start_frame = int(round(SKIP_SEC * fps))
    window_frames = int(round(WINDOW_SEC * fps))
    end_frame = start_frame + window_frames
    
    print(f"[System] FPS: {fps:.2f}")
    print(f"[System] Collecting Data from Frame {start_frame} to {end_frame}...")

    frames_buffer = []  
    prediction_result = None
    inference_done = False
    
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- VISUALIZATION SETUP ---
        frame = cv2.resize(frame, (640, 480))
        display_frame = frame.copy()
        
        # Draw ROI box if it exists
        if roi:
            cv2.rectangle(display_frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

        
        if start_frame <= frame_idx < end_frame:
           
            proc_frame = cv2.resize(frame, (640, 480))
            
            # 2. Gaussian Blur
            proc_frame = cv2.GaussianBlur(proc_frame, (5, 5), 10)

            if roi is not None:
                x1, y1, x2, y2 = roi
                H, W = proc_frame.shape[:2]
                x1c = max(0, min(W - 1, x1))
                x2c = max(0, min(W, x2))
                y1c = max(0, min(H - 1, y1))
                y2c = max(0, min(H, y2))
                
                if x2c > x1c and y2c > y1c:
                    proc_frame = proc_frame[y1c:y2c, x1c:x2c]
            
            # 4. Grayscale + Resize to OUT_SIZE + Normalize
            if proc_frame.size != 0:
                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (OUT_SIZE, OUT_SIZE))
                gray = gray.astype(np.float32) / 255.0
                frames_buffer.append(gray) # Store [H, W]

            # Visual Feedback
            cv2.putText(display_frame, "Status: COLLECTING DATA", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Frames: {len(frames_buffer)}", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        elif frame_idx >= end_frame and not inference_done:
            print(f"[System] Window finished. Processing {len(frames_buffer)} frames...")
            
            if len(frames_buffer) == 0:
                prediction_result = "Error: No frames captured"
            else:
                
                if len(frames_buffer) > MAX_FRAMES:
                    pick_idx = np.linspace(0, len(frames_buffer) - 1, MAX_FRAMES).round().astype(int)
                    final_frames = [frames_buffer[i] for i in pick_idx]
                else:
                    final_frames = frames_buffer
                
                # 6. Tensor Stacking [1, T, 1, H, W]
                frames_arr = np.stack(final_frames, axis=0) # [T, H, W]
                frames_arr = frames_arr[:, np.newaxis, :, :] # [T, 1, H, W]
                tensor = torch.from_numpy(frames_arr).float()
                tensor = tensor.unsqueeze(0).to(device) # [1, T, 1, H, W]
                
                # Run Model
                with torch.no_grad():
                    logits = model(tensor) 
                    
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)
                
                # Result
                conf_val = conf.item()
                label = config.IDX_TO_LABEL[pred_idx.item()]
                
                status_icon = "✅" if conf_val >= threshold else "⚠️"
                prediction_result = f"{label} ({conf_val:.1%}) {status_icon}"
                
                print(f"[Result] {label} | Confidence: {conf_val:.4f}")
                print("Probabilities:", probs.cpu().numpy())

            inference_done = True
            # Clean up memory
            del frames_buffer

        # --- DISPLAY PHASE (Post-Processing) ---
        elif inference_done:
             # Draw Background
            cv2.rectangle(display_frame, (20, 20), (550, 100), (0, 0, 0), -1)
            cv2.putText(display_frame, "RESULT:", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            
            color = (0, 255, 0) if "Error" not in prediction_result else (0, 0, 255)
            cv2.putText(display_frame, prediction_result, (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        else:
            # Waiting for start_frame (skip period)
            cv2.putText(display_frame, f"Waiting... ({frame_idx}/{start_frame})", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        # Show Output
        cv2.imshow('Strict Preprocess Stream', display_frame)
        frame_idx += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def main():
   
    MODEL_PATH = "best_model_advanced_thermal_2.pth"
    VIDEO_PATH = "FLIR0588_metal.mp4"  # <--- UPDATE THIS
    THRESHOLD = 0.75
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = load_model(MODEL_PATH, device)
        run_live_inference_strict(VIDEO_PATH, model, device, THRESHOLD)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()