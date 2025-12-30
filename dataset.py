import os
import cv2
import torch
import hashlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import config

class ThermalWasteVideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        excel_name="datas.xlsx",
        max_frames=100,
        out_size=224,
        skip_sec=5.0,
        heating_sec=10.0,
        cooling_sec=10.0,
        debug_print=False,
        use_cache=True,
        cache_dir="/content/thermal_cache",
        cache_format="pt",
        force_rebuild_cache=False,
    ):
        self.root_dir = root_dir
        self.max_frames = int(max_frames)
        self.out_size = int(out_size)
        self.skip_sec = float(skip_sec)
        self.window_sec = float(heating_sec + cooling_sec)
        self.debug_print = bool(debug_print)

        self.use_cache = bool(use_cache)
        self.cache_dir = cache_dir
        self.cache_format = cache_format
        self.force_rebuild_cache = bool(force_rebuild_cache)

        self.index_path = os.path.join(self.cache_dir, "cache_index.csv")
        
        self._init_cache()
        self._load_excel(excel_name)

    def _init_cache(self):
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            if not os.path.exists(self.index_path):
                pd.DataFrame(columns=[
                    "cache_file", "video", "label",
                    "x1", "y1", "x2", "y2",
                    "skip_sec", "window_sec", "max_frames", "out_size"
                ]).to_csv(self.index_path, index=False)

    def _load_excel(self, excel_name):
        excel_path = os.path.join(self.root_dir, excel_name)
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel not found: {excel_path}")

        self.df = pd.read_excel(excel_path)
        required_cols = ["FileName", "Label", "topleftx", "toplefty", "botrightx", "botrighty"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns in Excel: {missing}")

        self.df["Label"] = self.df["Label"].astype(str).str.strip()

    def __len__(self):
        return len(self.df)

    def _cache_key(self, video_name, roi):
        x1, y1, x2, y2 = roi
        base = (
            f"{video_name}|{x1},{y1},{x2},{y2}|"
            f"skip={self.skip_sec}|win={self.window_sec}|"
            f"maxf={self.max_frames}|sz={self.out_size}"
        )
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def _cache_path(self, key):
        ext = ".pt" if self.cache_format == "pt" else ".npy"
        return os.path.join(self.cache_dir, f"{key}{ext}")

    def _save_cache(self, path, frames_tensor):
        if self.cache_format == "pt":
            torch.save(frames_tensor, path)
        else:
            np.save(path, frames_tensor.numpy())

    def _load_cache(self, path):
        if self.cache_format == "pt":
            return torch.load(path, map_location="cpu")
        else:
            arr = np.load(path)
            return torch.from_numpy(arr).float()

    def _append_index_row(self, cache_file, video_name, label_str, roi):
        # Optimizing: Read header only first to avoid heavy reads if not needed, 
        # but for safety keeping original logic
        df_idx = pd.read_csv(self.index_path)
        if (df_idx["cache_file"] == cache_file).any():
            return
        x1, y1, x2, y2 = roi
        df_idx.loc[len(df_idx)] = [
            cache_file, video_name, label_str,
            x1, y1, x2, y2,
            self.skip_sec, self.window_sec, self.max_frames, self.out_size
        ]
        df_idx.to_csv(self.index_path, index=False)

    def _read_video_roi_frames(self, video_path, video_name, label_str, roi):
        if self.use_cache:
            key = self._cache_key(video_name, roi)
            cpath = self._cache_path(key)
            if (not self.force_rebuild_cache) and os.path.exists(cpath):
                if self.debug_print:
                    print(f"[CACHE HIT] {video_name} -> {os.path.basename(cpath)}")
                self._append_index_row(os.path.basename(cpath), video_name, label_str, roi)
                return self._load_cache(cpath)

        x1, y1, x2, y2 = roi
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        fps = float(fps)

        start_frame = int(round(self.skip_sec * fps))
        window_frames = int(round(self.window_sec * fps))
        end_frame = start_frame + window_frames

        frames = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx < start_frame:
                idx += 1
                continue
            if idx >= end_frame:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.GaussianBlur(frame, (5, 5), 10)

            H, W = frame.shape[:2]
            x1c = max(0, min(W - 1, x1))
            x2c = max(0, min(W, x2))
            y1c = max(0, min(H - 1, y1))
            y2c = max(0, min(H, y2))

            if x2c > x1c and y2c > y1c:
                roi_img = frame[y1c:y2c, x1c:x2c]
                if roi_img.size != 0:
                    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (self.out_size, self.out_size))
                    gray = gray.astype(np.float32) / 255.0
                    frames.append(gray[None, ...])

            idx += 1

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((1, self.out_size, self.out_size), dtype=np.float32)]

        if len(frames) > self.max_frames:
            pick_idx = np.linspace(0, len(frames) - 1, self.max_frames).round().astype(int)
            frames = [frames[i] for i in pick_idx]

        frames = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames).float()

        if self.use_cache:
            key = self._cache_key(video_name, roi)
            cpath = self._cache_path(key)
            try:
                self._save_cache(cpath, frames_tensor)
                self._append_index_row(os.path.basename(cpath), video_name, label_str, roi)
                if self.debug_print:
                    print(f"[CACHE SAVE] {video_name} -> {os.path.basename(cpath)}")
            except Exception as e:
                print(f"[CACHE WARN] Could not save cache for {video_name}: {e}")

        return frames_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = str(row["FileName"])
        label_str = str(row["Label"]).strip()

        if label_str not in config.LABEL_MAP:
            # Fallback or error - strictly enforcing config map
            raise ValueError(f"Unknown label '{label_str}' at row {idx}. Update config.LABEL_MAP.")

        x1 = int(round(float(row["topleftx"])))
        y1 = int(round(float(row["toplefty"])))
        x2 = int(round(float(row["botrightx"])))
        y2 = int(round(float(row["botrighty"])))

        video_path = os.path.join(self.root_dir, video_name)
        
        if self.debug_print:
            print(f"[DATASET] {idx+1}/{len(self.df)} : {video_name}")

        frames = self._read_video_roi_frames(video_path, video_name, label_str, (x1, y1, x2, y2))
        y = torch.tensor(config.LABEL_MAP[label_str], dtype=torch.long)
        return frames, y

def pad_collate(batch):
    frames_list, y_list = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in frames_list], dtype=torch.long)

    T_max = int(lengths.max().item())
    B = len(frames_list)
    C, H, W = frames_list[0].shape[1], frames_list[0].shape[2], frames_list[0].shape[3]

    out = torch.zeros((B, T_max, C, H, W), dtype=torch.float32)
    for i, f in enumerate(frames_list):
        out[i, : f.shape[0]] = f

    y = torch.stack(y_list, dim=0)
    return out, y, lengths