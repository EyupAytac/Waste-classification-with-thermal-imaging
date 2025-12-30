import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse

# Import your local project files
import config
from model import ThermalMaterialClassifier

def load_model(model_path, device):
    print(f"[System] Loading model from {model_path}...")
    
    # Initialize the architecture
    model = ThermalMaterialClassifier(
        num_classes=len(config.LABEL_MAP),
        spatial_dim=512,
        temporal_hidden=256
    ).to(device)

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def process_video(video_path):
    """
    Reads the entire video, resizes/normalizes every frame, 
    and prepares the final tensor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Preprocessing (Resize -> Blur -> Gray -> Normalize)
        # We process every frame first, then sample.
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.GaussianBlur(frame, (5, 5), 10)
        
        # Convert to Grayscale & Resize to Model Input Size (224x224)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (config.OUT_SIZE, config.OUT_SIZE))
        
        # Normalize (0-1)
        gray = gray.astype(np.float32) / 255.0
        frames.append(gray)

    cap.release()
    
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("Video contains no frames.")

    # 2. Downsampling / Sampling
    # The model expects exactly MAX_FRAMES (e.g., 100).
    # If video is longer, we pick 100 evenly spaced frames.
    # If video is shorter, we take them all.
    if total_frames > config.MAX_FRAMES:
        indices = np.linspace(0, total_frames - 1, config.MAX_FRAMES).round().astype(int)
        frames = [frames[i] for i in indices]
    
    # 3. Convert to Tensor
    # Shape: [Time, H, W] -> [1, Time, 1, H, W]
    tensor = np.stack(frames, axis=0)
    tensor = torch.from_numpy(tensor).float()
    tensor = tensor.unsqueeze(0).unsqueeze(2) # Add Batch and Channel dimensions
    
    return tensor, total_frames

def main():
    # --- CONFIGURATION ---
    MODEL_PATH = "best_model_advanced_thermal_claudeVersion.pth"
    VIDEO_PATH = "FLIR0654_plastic.mp4"  # <--- REPLACE WITH YOUR VIDEO PATH
    THRESHOLD = 0.75
    # ---------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 1. Setup
        model = load_model(MODEL_PATH, device)
        print(f"[System] Processing video: {VIDEO_PATH}")

        # 2. Process Video
        input_tensor, frame_count = process_video(VIDEO_PATH)
        input_tensor = input_tensor.to(device)
        lengths = torch.tensor([input_tensor.shape[1]]).to(device)

        print(f"[System] Video read successfully. Frames: {frame_count}. Running inference...")

        # 3. Run Inference
        with torch.no_grad():
            logits = model(input_tensor, lengths)
            probs = F.softmax(logits, dim=1)
            max_prob, idx = torch.max(probs, dim=1)

        # 4. Results
        confidence = max_prob.item()
        label_idx = idx.item()
        label_name = config.IDX_TO_LABEL[label_idx]
        
        print("\n" + "="*40)
        print(" FINAL RESULT")
        print("="*40)
        
        if confidence >= THRESHOLD:
            print(f"Detected Object:  {label_name.upper()}")
            print(f"Confidence:       {confidence:.4f}")
            print(f"Status:           CONFIDENT ✅")
        else:
            print(f"Top Prediction:   {label_name} (Low Confidence)")
            print(f"Confidence:       {confidence:.4f}")
            print(f"Status:           UNCERTAIN ⚠️")
            
        print("-" * 40)
        print("Raw Probabilities:")
        for i, prob in enumerate(probs.cpu().numpy()[0]):
            name = config.IDX_TO_LABEL[i]
            print(f"  - {name}: {prob:.4f}")
        print("="*40 + "\n")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()