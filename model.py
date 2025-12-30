import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=1, feat_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, feat_dim, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert feat_dim % num_heads == 0
        
        self.qkv = nn.Linear(feat_dim, feat_dim * 3)
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out


class TemporalEncoder(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, num_layers=2, num_heads=8, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, feat_dim)
        self.attention = TemporalAttention(feat_dim, num_heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(feat_dim)
        self.ln2 = nn.LayerNorm(feat_dim)
        self.ff = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        
        x = self.proj(lstm_out)
        x = self.ln1(x)
        
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        x = self.ln2(x)
        x = x + self.ff(x)
        return x, mask


class ThermalDeltaModule(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim // 2)
        )
    
    def forward(self, x, lengths=None):
        B, T, C = x.shape
        if lengths is None:
            baseline = x[:, :T//4].mean(dim=1)
            peak = x[:, T//4:3*T//4].max(dim=1)[0]
            final = x[:, 3*T//4:].mean(dim=1)
        else:
            baseline_list, peak_list, final_list = [], [], []
            for i in range(B):
                L = lengths[i].item()
                baseline_list.append(x[i, :L//4].mean(dim=0))
                peak_list.append(x[i, L//4:3*L//4].max(dim=0)[0])
                final_list.append(x[i, 3*L//4:L].mean(dim=0))
            baseline = torch.stack(baseline_list)
            peak = torch.stack(peak_list)
            final = torch.stack(final_list)
        
        delta_feat = torch.cat([baseline, peak, final], dim=1)
        return self.fc(delta_feat)


class ThermalMaterialClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, spatial_dim=512, 
                 temporal_hidden=256, temporal_layers=2, num_heads=8, dropout=0.2):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(in_channels, spatial_dim)
        self.temporal_encoder = TemporalEncoder(
            spatial_dim, temporal_hidden, temporal_layers, num_heads, dropout
        )
        self.delta_module = ThermalDeltaModule(spatial_dim)
        
        fusion_dim = spatial_dim + spatial_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, spatial_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, spatial_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(spatial_dim // 2, num_classes)
        
    def forward(self, x, lengths=None):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        spatial_feats = self.spatial_encoder(x_flat)
        spatial_feats = spatial_feats.view(B, T, -1)
        
        temporal_feats, mask = self.temporal_encoder(spatial_feats, lengths)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            temporal_agg = (temporal_feats * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            temporal_agg = temporal_feats.mean(dim=1)
        
        delta_feats = self.delta_module(spatial_feats, lengths)
        
        fused = torch.cat([temporal_agg, delta_feats], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        
        return logits