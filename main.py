"""
CvT-based Image Quality Assessment for Derained Images.

Architecture follows the paper's RCTB design:
    RCTB_i = Conv Token Embedding + Transformer Block (CvT stage i)
             + Feature Refinement Module (FRM)

Distortion priors:
    RDE -> 384-ch rain distortion representation
    SDE -> 512-ch structural distortion representation

Provides:
    - FeatureRefinementModule : FRM inside each RCTB
    - CvTFeatureExtractor     : CvT features + RDE / SDE priors
    - CvTFeatureRegression    : quality regression head
    - DerainIQADataset        : precomputed-feature dataset
    - CvTIQATrainer           : train / validate / test loop
"""
from __future__ import annotations
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import cvt
from models import RDE, SDE


def get_device() -> torch.device:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")



# RCTB components

class FeatureRefinementModule(nn.Module):
    """Feature Refinement Module (FRM) inside each RCTB.

    Depth-wise separable residual block:
        1x1 expand -> 3x3 depthwise -> GELU -> 1x1 project -> + identity.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1,
                                groups=hidden, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.act(out)
        out = self.conv2(out)
        return x + out



# Feature extraction

class CvTFeatureExtractor:
    """Extract multi-stage CvT features, RDE and SDE priors."""

    def __init__(
        self,
        model_path: str = "checkpoint/CvT-21-384x384-IN-22k.pth",
        rde_ckpt: str = "checkpoint/rde_best.pth",
        sde_ckpt: str = "checkpoint/sde_best.pth",
    ) -> None:
        for p in (model_path, rde_ckpt, sde_ckpt):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Checkpoint not found: {p}")

        self.model = cvt.get_cls_model().cuda()
        self.model.load_state_dict(
            torch.load(model_path, weights_only=False), strict=False
        )
        self.model.eval().cuda()

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.rde = self._load_module(RDE(), rde_ckpt)
        self.sde = self._load_module(SDE(), sde_ckpt)

    @staticmethod
    def _load_module(net: nn.Module, ckpt: str) -> nn.Module:
        net.load_state_dict(torch.load(ckpt, weights_only=False))
        return net.eval().cuda()

    def extract_features(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            x = self.transform(image).unsqueeze(0).cuda()
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")

        feats = {}
        with torch.no_grad():
            x0, cls0 = self.model.stage0(x)
            feats["stage0"] = {"feature_map": x0.cuda(), "cls_token": cls0}

            x1, cls1 = self.model.stage1(x0)
            feats["stage1"] = {"feature_map": x1.cuda(), "cls_token": cls1}

            x2, cls2 = self.model.stage2(x1)
            feats["stage2"] = {"feature_map": x2.cuda(), "cls_token": cls2.cuda()}
        return feats

    def load_image(self, image_path: str) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img = Image.open(image_path).convert("RGB")
        return transform(img).unsqueeze(0).cuda()

    def extract_rde(self, image_path: str,
                    target_size=(6, 6)) -> torch.Tensor:
        """RDE output: [1, 384, H/8, W/8] -> adaptive pool -> [1, 384, t, t]."""
        img = self.load_image(image_path)
        with torch.no_grad():
            rep = self.rde(img)
        return F.adaptive_avg_pool2d(rep, target_size).cuda()

    def extract_sde(self, image_path: str,
                    target_size=(6, 6)) -> torch.Tensor:
        """SDE output: [1, 512, H/4, W/4] -> adaptive pool -> [1, 512, t, t]."""
        img = self.load_image(image_path)
        with torch.no_grad():
            rep = self.sde(img)
        return F.adaptive_avg_pool2d(rep, target_size).cuda()

class DistortionGuidedEnhancement(nn.Module):
    """Unified RAAM / SAAM.

    Q = W_q * F_dist,  K, V = W_kv * F_stage
    A = Softmax(QK^T / sqrt(d)) V,  4 heads
    """

    def __init__(self, dist_dim: int, stage_dim: int,
                 embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dist_dim, embed_dim)
        self.kv_proj = nn.Linear(stage_dim, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)

    def forward(self, dist_feat: torch.Tensor,
                stage_feat: torch.Tensor) -> torch.Tensor:
        if dist_feat.dim() == 4:
            q = self.q_proj(dist_feat.flatten(2).transpose(1, 2))
        elif dist_feat.dim() == 2:
            q = self.q_proj(dist_feat).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected dist_feat shape: {dist_feat.shape}")

        kv = self.kv_proj(stage_feat.flatten(2).transpose(1, 2))
        fused, _ = self.attn(q, kv, kv)
        return fused.mean(dim=1)

# Regression model

# ---------------------------------------------------------------------------
# Distortion-guided Feature Enhancement Module (RAAM / SAAM)
# ---------------------------------------------------------------------------
class DistortionGuidedEnhancement(nn.Module):
    """Unified architecture for RAAM and SAAM.

    Paper (Section 3.3):
        Q_r = W_q^r F_r,   K_r, V_r = W_kv^r (F_1)      -> RAAM
        Q_s = W_q^s F_s,   K_s, V_s = W_kv^s (F_3)      -> SAAM
        A   = Softmax(Q K^T / sqrt(d)) V               (4 heads)

    Instantiated twice in CvTFeatureRegression:
        - RAAM : dist_dim=384 (RDE), stage_dim=64  (CvT stage0 / F_1)
        - SAAM : dist_dim=512 (SDE), stage_dim=384 (CvT stage2 / F_3)
    """

    def __init__(self, dist_dim: int, stage_dim: int,
                 embed_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dist_dim, embed_dim)
        self.kv_proj = nn.Linear(stage_dim, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

    def forward(self, dist_feat: torch.Tensor,
                stage_feat: torch.Tensor) -> torch.Tensor:
        # Q from distortion prior
        if dist_feat.dim() == 4:
            q = self.q_proj(dist_feat.flatten(2).transpose(1, 2))  # [B, hw, D]
        elif dist_feat.dim() == 2:
            q = self.q_proj(dist_feat).unsqueeze(1)                # [B, 1, D]
        else:
            raise ValueError(f"Unexpected dist_feat shape: {dist_feat.shape}")

        # K, V from stage feature
        kv = self.kv_proj(stage_feat.flatten(2).transpose(1, 2))

        fused, _ = self.attn(q, kv, kv)
        return fused.mean(dim=1)                                   # [B, D]


# ---------------------------------------------------------------------------
# Regression model
# ---------------------------------------------------------------------------
class CvTFeatureRegression(nn.Module):
    """Fuse CvT features + RAAM / SAAM enhanced priors for MOS regression.

    RCTB outputs (paper Eq. 1):
        F_i = RCTB_i(F_{i-1}) = FRM_i(CvT_Stage_i(F_{i-1})), i = 1, 2, 3

    Token sequence fed to the fusion Transformer:
        T = [F_1, F_2, F_3, F_cls, F_r, F_s]           (6 tokens, 256-d each)
    (the paper writes [F_1, F_2, F_3, F_r, F_s] in the formula, but the
     regression head 1536 = 256 x 6 confirms CLS is included.)

    Regression head: 1280 - 512 - 256 - 128 - 1  (GELU + dropout)
    """

    def __init__(self, num_heads: int = 4, hidden_dim: int = 512,
                 dropout: float = 0.3, use_frm: bool = True) -> None:
        super().__init__()
        self.use_frm = use_frm

        # ---- FRM inside each RCTB (paper Eq. 1) ----
        if use_frm:
            self.frm0 = FeatureRefinementModule(64)
            self.frm1 = FeatureRefinementModule(192)
            self.frm2 = FeatureRefinementModule(384)

        # ---- Stage-specific projections to common 256-d ----
        self.map_f0 = nn.Linear(64, 256)
        self.map_f1 = nn.Linear(192, 256)
        self.map_f2 = nn.Linear(384, 256)
        self.map_cls = nn.Linear(384, 256)

        # ---- Attention pooling for stage-level tokens ----
        self.attn_pool = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1),
        )

        # ---- RAAM: RDE prior -> CvT stage0 (F_1) ----
        self.raam = DistortionGuidedEnhancement(
            dist_dim=384, stage_dim=64, embed_dim=256, num_heads=num_heads,
        )
        # ---- SAAM: SDE prior -> CvT stage2 (F_3) ----
        self.saam = DistortionGuidedEnhancement(
            dist_dim=512, stage_dim=384, embed_dim=256, num_heads=num_heads,
        )

        # ---- Token-level fusion ----
        # 2 layers, d=256, 4 heads, FFN=2048 (default), matching the paper
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=2048,
                batch_first=True,
            ),
            num_layers=2,
        )

        # ---- Regression head: 1536 - 512 - 256 - 128 - 1 ----
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 * 5, hidden_dim),   # 1536 -> 512
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 -> 256
            nn.GELU(), nn.Dropout(dropout),
        )
        self.quality_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),  # 256 -> 128
            nn.GELU(),
            nn.Linear(128, 1),                # 128 -> 1
        )

    def attention_pooling(self, x: torch.Tensor,
                          proj: nn.Module) -> torch.Tensor:
        tokens = proj(x.flatten(2).transpose(1, 2))
        attn = torch.softmax(self.attn_pool(tokens), dim=1)
        return (tokens * attn).sum(dim=1)

    def forward(self, f0, f1, f2, rain, blur):
        # ---- RCTB outputs F_i = FRM_i(CvT_Stage_i(F_{i-1})) ----
        if self.use_frm:
            f0 = self.frm0(f0)     # F_1, 64-ch
            f1 = self.frm1(f1)     # F_2, 192-ch
            f2 = self.frm2(f2)     # F_3, 384-ch

        pf0 = self.attention_pooling(f0, self.map_f0)
        pf1 = self.attention_pooling(f1, self.map_f1)
        pf2 = self.attention_pooling(f2, self.map_f2)

        # ---- RAAM: rain prior -> F_1 ; SAAM: structural prior -> F_3 ----
        rain_f = self.raam(rain, f0)     # F_r
        blur_f = self.saam(blur, f2)     # F_s

        # ---- Unified token sequence (6 tokens) ----
        tokens = torch.stack([pf0, pf1, pf2, rain_f, blur_f], dim=1)
        tokens = self.fusion_transformer(tokens)

        fused = tokens.reshape(tokens.size(0), -1)   # [B, 1536]
        fused = self.feature_fusion(fused)
        return self.quality_regressor(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DerainIQADataset(Dataset):
    def __init__(self, preload_dir: str) -> None:
        self.files = sorted(glob.glob(os.path.join(preload_dir, "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"No .pt files under: {preload_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], weights_only=False)
        return (
            data["f0"].float(), data["f1"].float(), data["f2"].float(),
            data["cls"].float(), data["rain"].float(), data["blur"].float(),
            torch.tensor(data["mos"], dtype=torch.float32), data["image_name"],
        )



# Trainer

class CvTIQATrainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 lr: float = 1e-4) -> None:
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50,
        )

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_plcc = 0.0

    @staticmethod
    def ranking_loss(preds: torch.Tensor, targets: torch.Tensor,
                     min_diff: float = 0.05,
                     margin: float = 0.02) -> torch.Tensor:
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        pred_diff = preds[:, None] - preds[None, :]
        target_diff = targets[:, None] - targets[None, :]
        mask = target_diff > min_diff

        loss_matrix = F.relu(margin - pred_diff)
        if not mask.any():
            return preds.sum() * 0.0
        return loss_matrix[mask].mean()

    @staticmethod
    def _to_device(batch, device):
        f0, f1, f2, cls, rain, blur, targets, _ = batch
        return (
            f0.squeeze(1).to(device), f1.squeeze(1).to(device),
            f2.squeeze(1).to(device), cls.squeeze(1).to(device),
            rain.squeeze(1).to(device), blur.squeeze(1).to(device),
            targets.to(device),
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            f0, f1, f2, cls, rain, blur, targets = self._to_device(
                batch, self.device)

            self.optimizer.zero_grad()
            preds = self.model(f0, f1, f2, rain, blur).reshape(-1)

            loss = (0.7 * self.criterion(preds, targets)
                    + 0.3 * self.ranking_loss(preds, targets))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        preds_all, targets_all = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                f0, f1, f2, cls, rain, blur, targets = self._to_device(
                    batch, self.device)
                preds = self.model(f0, f1, f2, rain, blur)
                total_loss += self.criterion(preds, targets).item()
                preds_all.extend(preds.cpu().numpy())
                targets_all.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        mse = mean_squared_error(targets_all, preds_all)
        mae = mean_absolute_error(targets_all, preds_all)
        plcc, _ = pearsonr(targets_all, preds_all)
        srocc, _ = spearmanr(targets_all, preds_all)

        self.scheduler.step()

        print(f"Validation - Epoch {epoch} | Loss: {avg_loss:.6f} | "
              f"MSE: {mse:.6f} | MAE: {mae:.6f} | "
              f"PLCC: {plcc:.6f} | SROCC: {srocc:.6f}")

        if plcc > self.best_plcc:
            self.best_plcc = plcc
            torch.save(self.model.state_dict(), "bestplcc.pth")
            print(f"Best model saved (PLCC: {plcc:.6f})")

        return avg_loss, mse, mae, plcc, srocc

    def test(self, model_path: str | None = None,
             save_fig: str | None = "training_results.png"):
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, weights_only=False))

        self.model.eval()
        preds_all, targets_all = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                f0, f1, f2, cls, rain, blur, targets = self._to_device(
                    batch, self.device)
                preds = self.model(f0, f1, f2, rain, blur)
                preds_all.extend(preds.cpu().numpy())
                targets_all.extend(targets.cpu().numpy())

        mse = mean_squared_error(targets_all, preds_all)
        mae = mean_absolute_error(targets_all, preds_all)
        rmse = np.sqrt(mse)
        plcc, _ = pearsonr(targets_all, preds_all)
        srocc, _ = spearmanr(targets_all, preds_all)
        krocc, _ = kendalltau(targets_all, preds_all)

        print("Test Results:")
        print(f"  MSE  : {mse:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
        print(f"  PLCC : {plcc:.6f} | SROCC: {srocc:.6f} | KROCC: {krocc:.6f}")

        if save_fig:
            self._plot_results(targets_all, preds_all, save_fig)

        return {"mse": mse, "mae": mae, "rmse": rmse,
                "plcc": plcc, "srocc": srocc, "krocc": krocc}

    @staticmethod
    def _plot_results(targets, predictions, save_path: str):
        targets, predictions = np.array(targets), np.array(predictions)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(targets, predictions, alpha=0.6)
        axes[0].plot([targets.min(), targets.max()],
                     [targets.min(), targets.max()], "r--", linewidth=2)
        axes[0].set(xlabel="True MOS", ylabel="Predicted MOS",
                    title="Predicted vs True MOS")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(predictions - targets, bins=30,
                     alpha=0.7, edgecolor="black")
        axes[1].set(xlabel="Prediction Error", ylabel="Frequency",
                    title="Error Distribution")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def train(self, epochs: int):
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, _, _, plcc, srocc = self.validate(epoch)
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"PLCC: {plcc:.6f} | SROCC: {srocc:.6f}")



def main():
    train_set = DerainIQADataset("preload_derain1_scene/train")
    val_set = DerainIQADataset("preload_derain1_scene/val")
    test_set = DerainIQADataset("preload_derain1_scene/test")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False,
                             num_workers=0)

    print("Data loaders ready:")
    print(f"  Train : {len(train_set)} samples")
    print(f"  Val   : {len(val_set)} samples")
    print(f"  Test  : {len(test_set)} samples")

    model = CvTFeatureRegression()
    trainer = CvTIQATrainer(model, train_loader, val_loader, test_loader,
                            lr=1e-4)

    # trainer.train(epochs=150)
    trainer.test("bestplcc.pth")


if __name__ == "__main__":
    main()
