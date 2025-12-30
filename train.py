import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import local modules
import config
from dataset import ThermalWasteVideoDataset, pad_collate
from model import ThermalMaterialClassifier

class Trainer:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler() if device == "cuda" else None
        
        self.best_acc = -1.0
        self.best_state = None

    def train_epoch(self):
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for x, y, lengths in self.train_loader:
            x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)

            if self.scaler:
                with autocast():
                    logits = self.model(x, lengths)
                    loss = self.loss_fn(logits, y)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x, lengths)
                loss = self.loss_fn(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.detach().item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        return total_loss / max(total, 1), total_correct / max(total, 1)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        all_preds, all_true = [], []

        for x, y, lengths in self.test_loader:
            x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
            logits = self.model(x, lengths)
            loss = self.loss_fn(logits, y)

            total_loss += loss.detach().item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.cpu().numpy())
            all_true.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
        all_true = np.concatenate(all_true) if len(all_true) > 0 else np.array([])
        
        return total_loss / max(total, 1), total_correct / max(total, 1), all_true, all_preds

    def run(self, epochs):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            tr_loss, tr_acc = self.train_epoch()
            te_loss, te_acc, _, _ = self.evaluate()

            self.scheduler.step(te_loss)

            print(f"Epoch {epoch+1:02d}/{epochs} | "
                  f"LR {current_lr:.2e} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc*100:.2f}% | "
                  f"Test Loss {te_loss:.4f} Acc {te_acc*100:.2f}%")

            if te_acc > self.best_acc:
                self.best_acc = te_acc
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"  → New best accuracy: {self.best_acc*100:.2f}%")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        if self.best_state:
            self.model.load_state_dict(self.best_state)
            
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'scheduler_state': self.scheduler.state_dict()
        }
        path = os.path.join(config.ROOT_DIR, f'checkpoint_epoch{epoch+1}.pth')
        torch.save(checkpoint, path)
        print(f"  → Checkpoint saved: {path}")

def main():
    # 1. Setup Drive (if applicable)
    if os.path.exists('/content/drive'):
        print("Drive already mounted or running in local environment similar to Colab.")
    else:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            print("Running locally (Google Colab modules not found).")

    # 2. Prepare Dataset
    print("\n" + "="*60 + "\nLOADING DATASET\n" + "="*60)
    full_ds = ThermalWasteVideoDataset(
        root_dir=config.ROOT_DIR,
        excel_name=config.EXCEL_NAME,
        max_frames=config.MAX_FRAMES,
        out_size=config.OUT_SIZE,
        skip_sec=config.SKIP_SEC,
        heating_sec=config.HEATING_SEC,
        cooling_sec=config.COOLING_SEC,
        debug_print=config.DATASET_PROGRESS_PRINT,
        use_cache=config.USE_CACHE,
        cache_dir=config.CACHE_DIR,
        cache_format=config.CACHE_FORMAT,
        force_rebuild_cache=config.FORCE_REBUILD_CACHE
    )

    # Cache prebuild
    if config.USE_CACHE:
        print("\n=== CACHE CHECK / PREBUILD ===")
        for i in range(len(full_ds)):
            _ = full_ds[i]
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(full_ds)} videos")
    
    # Split
    labels = full_ds.df["Label"].map(config.LABEL_MAP).values
    idx = np.arange(len(full_ds))
    tr_idx, te_idx = train_test_split(
        idx, train_size=config.TRAIN_SPLIT, random_state=config.SEED, shuffle=True, stratify=labels
    )
    
    train_ds = torch.utils.data.Subset(full_ds, tr_idx)
    test_ds = torch.utils.data.Subset(full_ds, te_idx)

    # Weights
    train_labels = full_ds.df.iloc[tr_idx]["Label"].map(config.LABEL_MAP).values
    counts = np.bincount(train_labels, minlength=len(config.LABEL_MAP))
    weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
    weights = torch.tensor(weights, device=config.DEVICE)
    
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    print(f"\nTrain: {len(train_ds)} | Test: {len(test_ds)}")
    print(f"Class Weights: {weights.cpu().numpy()}")

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=2, collate_fn=pad_collate, pin_memory=(config.DEVICE=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=2, collate_fn=pad_collate, pin_memory=(config.DEVICE=="cuda"))

    # 3. Model Setup
    print("\n" + "="*60 + "\nBUILDING MODEL\n" + "="*60)
    model = ThermalMaterialClassifier(
        num_classes=len(config.LABEL_MAP),
        spatial_dim=512,
        temporal_hidden=256
    ).to(config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # 4. Run Training
    trainer = Trainer(model, train_loader, test_loader, loss_fn, optimizer, scheduler, config.DEVICE)
    trainer.run(config.EPOCHS)

    # 5. Final Report
    print("\n" + "="*60 + "\nFINAL EVALUATION\n" + "="*60)
    _, _, y_true, y_pred = trainer.evaluate()
    
    print(f"\nBest Test Accuracy: {trainer.best_acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    target_names = [config.IDX_TO_LABEL[i] for i in range(len(config.LABEL_MAP))]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Save
    out_path = os.path.join(config.ROOT_DIR, config.MODEL_OUT_NAME)
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to: {out_path}")

if __name__ == "__main__":
    main()