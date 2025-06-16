import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob, os
import random
import torchvision.utils as vutils

# --- Teacher Encoder (frozen ResNet) ---
class TeacherEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, layers=[1,2,3,4]):
        super().__init__()
        resnet = getattr(models, backbone)(pretrained=pretrained)
        self.stages = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ])
        for p in self.parameters(): p.requires_grad = False
        self.layers = layers

    def forward(self, x):
        feats = []
        out = x
        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx in self.layers:
                feats.append(out)
        return feats  # list of [B,C,H,W]

# --- Multi-Scale Feature Fusion ---
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, hidden_dim=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])

    def forward(self, feats):
        target_size = feats[-1].shape[2:]
        outs = []
        for f, proj in zip(feats, self.projs):
            z = proj(f)
            if z.shape[2:] != target_size:
                z = F.interpolate(z, size=target_size, mode='bilinear', align_corners=False)
            outs.append(z)
        return torch.cat(outs, dim=1)

# --- Bottleneck Embedding (One-Class) ---
class OneClassEmbedding(nn.Module):
    def __init__(self, in_channels, bottleneck_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(bottleneck_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# --- Student Decoder (symmetric to teacher) ---
class StudentDecoder(nn.Module):
    def __init__(self, bottleneck_dim=64, out_channels_list=[256,512,1024,2048]):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = bottleneck_dim
        for idx, out_ch in enumerate(out_channels_list):
            if idx == 0:
                # project bottleneck to highest-level channels
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            else:
                # upsample by factor 2
                self.blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            in_ch = out_ch

    def forward(self, x):
        feats = []
        out = x
        for blk in self.blocks:
            out = blk(out)
            feats.append(out)
        return list(reversed(feats))  # reverse to match teacher order

# --- Reverse Distillation Model ---
class ReverseDistillationModel(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, layers=[1,2,3,4], hidden_dim=256, bottleneck_dim=64):
        super().__init__()
        self.teacher = TeacherEncoder(backbone, pretrained, layers)
        in_chs = [self.teacher.stages[i][-1].bn3.num_features for i in layers]
        self.fusion = MultiScaleFeatureFusion(in_chs, hidden_dim)
        self.embed = OneClassEmbedding(hidden_dim * len(in_chs), bottleneck_dim)
        # student out_channels list reversed (from deep to shallow)
        self.student = StudentDecoder(bottleneck_dim, out_channels_list=list(reversed(in_chs)))

    def forward(self, x):
        t_feats = self.teacher(x)
        fused = self.fusion(t_feats)
        bneck = self.embed(fused)
        s_feats = self.student(bneck)
        return t_feats, s_feats

# --- Cosine Distillation Loss ---
def distillation_loss(t_feats, s_feats):
    loss = 0.0
    # student feats are reversed to match teacher
    for t, s in zip(t_feats, s_feats):
        B,C,H,W = t.shape
        t_flat = t.view(B, C, -1)
        s_flat = s.view(B, C, -1)
        cos = F.cosine_similarity(t_flat, s_flat, dim=1)  # [B, H*W]
        loss += (1 - cos).mean()
    return loss / len(t_feats)

# --- Dataset for flat folder of images ---
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None, exts=('png','jpg','bmp')):
        self.paths = []
        for ext in exts:
            self.paths += glob.glob(os.path.join(root, f'*.{ext}'))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

# --- Training on normal images ---
def train(model, loader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(
        list(model.fusion.parameters()) +
        list(model.embed.parameters()) +
        list(model.student.parameters()), lr=lr)
    for e in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            t_feats, s_feats = model(imgs)
            loss = distillation_loss(t_feats, s_feats)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        print(f"Epoch {e+1}/{epochs}, Loss: {total_loss/len(loader.dataset):.4f}")

# --- Inference: compute anomaly map for a single image ---
def compute_anomaly_map(model, img, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)
        t_feats, s_feats = model(x)
        # per-scale maps upsampled to input size
        maps = []
        for t,s in zip(t_feats, s_feats):
            # cosine distance per spatial location
            cos = F.cosine_similarity(t, s, dim=1, eps=1e-6)  # [1,H,W]
            dist = 1 - cos
            # upsample to original resolution
            m = F.interpolate(dist.unsqueeze(0), size=img.shape[1:], mode='bilinear', align_corners=False)
            maps.append(m.squeeze(0))
        # average across scales
        anomaly_map = torch.stack(maps, dim=0).mean(dim=0)
    return anomaly_map.cpu()

# --- Main example ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # transforms for training and inference
    train_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    infer_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    # Load normal training images
    train_ds = FlatFolderDataset('data/train', transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)

    # Build and train model
    model = ReverseDistillationModel(backbone='resnet50', pretrained=True)
    train(model, train_loader, epochs=20, lr=5e-4, device=device)

 # Synthetic anomaly generator (Gaussian noise)
    def synth_anomaly(x, num_patches=5, patch_size=(32, 32), sigma=0.5):
        """
        Добавляет шумовые артефакты в случайные патчи изображения.
        x: [1, C, H, W] или [C, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1, C, H, W]
        x_noised = x.clone()
        _, C, H, W = x.shape

        for _ in range(num_patches):
            h = random.randint(0, H - patch_size[0])
            w = random.randint(0, W - patch_size[1])
            patch = x[:, :, h:h+patch_size[0], w:w+patch_size[1]]

            # Сильный шум или случайный "артефакт"
            noise = torch.randn_like(patch) * sigma
            patch_anom = torch.clamp(patch + noise, 0.0, 1.0)

            x_noised[:, :, h:h+patch_size[0], w:w+patch_size[1]] = patch_anom

        return x_noised.squeeze(0)
    
    # Папка для сохранения аномальных изображений
    save_dir = 'results/synthetic_anomalies'
    os.makedirs(save_dir, exist_ok=True)


    # Prepare evaluation on synthetic anomalies
    normal_paths = glob.glob('data/test/*.png') + glob.glob('data/test/*.bmp')
    y_true, y_scores = [], []
    model.eval()
    with torch.no_grad():
        for i, path in enumerate(normal_paths):
            img = infer_tf(Image.open(path).convert('RGB')).to(device)
            # Compute clean score
            amap_clean = compute_anomaly_map(model, img, device=device)
            score_clean = amap_clean.mean().item()
            y_true.append(0)
            y_scores.append(score_clean)

            # Generate synthetic anomaly and compute score
            img_anom = synth_anomaly(img.unsqueeze(0)).squeeze(0)
            amap_anom = compute_anomaly_map(model, img_anom, device=device)
            score_anom = amap_anom.mean().item()
            y_true.append(1)
            y_scores.append(score_anom)
                
            # Сохраняем аномальное изображение
            save_path = os.path.join(save_dir, f"anomaly_{i:04d}.png")
            vutils.save_image(img_anom.cpu(), save_path)

    # Compute ROC AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_scores)
    print(f"Synthetic Test ROC-AUC: {auc:.4f}")