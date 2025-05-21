import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os, glob

# --- Teacher Encoder (замороженный) ---
class TeacherEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, layers=[1,2,3,4]):
        super(TeacherEncoder, self).__init__()
        resnet = getattr(models, backbone)(pretrained=pretrained)
        self.stages = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        ])
        for param in self.parameters(): param.requires_grad = False
        self.layers = layers

    def forward(self, x):
        feats, out = [], x
        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx in self.layers: feats.append(out)
        return feats

# --- MultiScale Fusion & Bottleneck Embedding ---
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, hidden_dim=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
    def forward(self, feats):
        target = feats[-1].shape[2:]
        processed = []
        for f, p in zip(feats, self.projs):
            z = p(f)
            if z.shape[2:] != target:
                z = F.interpolate(z, size=target, mode='bilinear', align_corners=False)
            processed.append(z)
        return torch.cat(processed, dim=1)

class OneClassEmbedding(nn.Module):
    def __init__(self, in_channels, bottleneck_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, 1),
            nn.BatchNorm2d(bottleneck_dim), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

# --- Student Decoder (с корректировкой порядка и размеров) ---
class StudentDecoder(nn.Module):
    def __init__(self, bottleneck_dim=64, out_channels_list=[2048,1024,512,256]):
        super().__init__()
        in_ch = bottleneck_dim
        self.blocks = nn.ModuleList()
        for idx, out_ch in enumerate(out_channels_list):
            if idx == 0:
                # 1x1 conv: сохраняем пространственный размер 8x8
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
                ))
            else:
                # транспонированный свёрточный для удвоения размера
                self.blocks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
                ))
            in_ch = out_ch

    def forward(self, x):
        feats, out = [], x
        for blk in self.blocks:
            out = blk(out)
            feats.append(out)
        return feats

# --- Полная модель Reverse Distillation ---
class ReverseDistillationModel(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, layers=[1,2,3,4], hidden_dim=256, bottleneck_dim=64):
        super().__init__()
        self.teacher = TeacherEncoder(backbone, pretrained, layers)
        # Получаем каналы выходов Bottleneck каждого этапа учителя
        in_chs = [self.teacher.stages[i][-1].bn3.num_features for i in layers]
        self.msf = MultiScaleFeatureFusion(in_chs, hidden_dim)
        self.oce = OneClassEmbedding(hidden_dim * len(in_chs), bottleneck_dim)
        # Декодер студента: обратный порядок каналов учителя
        self.student = StudentDecoder(bottleneck_dim, out_channels_list=list(reversed(in_chs)))

    def forward(self, x):
        with torch.no_grad(): t_feats = self.teacher(x)
        fused = self.msf(t_feats)
        bneck = self.oce(fused)
        s_feats = self.student(bneck)
        return t_feats, s_feats

def distillation_loss(t_feats, s_feats, reduction='mean'):
    """
    Косинусная «потеря» между признаками учителя и студента.
    - t_feats: список тензоров [B, C, H, W] из модели-учителя
    - s_feats: список тензоров [B, C, H, W] из модели-студента
    - reduction: 'none' | 'mean' | 'sum'
    Возвращает:
      - если reduction='none': тензор [B]
      - иначе: скаляр
    """
    # Переворачиваем признаки студента, чтобы разрешения совпадали
    s_feats_rev = list(reversed(s_feats))
    B = t_feats[0].shape[0]
    device = t_feats[0].device

    # Собираем пер-сэмпл потери
    loss_per_sample = torch.zeros(B, device=device)
    for t, s in zip(t_feats, s_feats_rev):
        # приводим к [B, C, H*W]
        t_flat = t.view(B, t.shape[1], -1)
        s_flat = s.view(B, s.shape[1], -1)
        # получаем [B, H*W] косинусных схожестей
        cos = F.cosine_similarity(t_flat, s_flat, dim=1)
        # и усредняем по пространству → [B]
        loss_per_sample += (1 - cos).mean(dim=1)

    # нормируем по числу уровней (опционально)
    loss_per_sample /= len(t_feats)

    if reduction == 'none':
        return loss_per_sample
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:  # 'mean'
        return loss_per_sample.mean()
    
# --- Синтетический генератор аномалий ---
class NoiseGenerator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, padding=1), nn.Tanh()
        )
    def forward(self, x): return torch.clamp(x + 0.2 * self.net(x), 0, 1)

# --- FlatFolderDataset ---
class FlatFolderDataset(Dataset):
    def __init__(self, root, exts=('bmp','png','jpg'), transform=None):
        self.files = []
        for ext in exts: self.files += glob.glob(os.path.join(root, f'*.{ext}'))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, 0

# --- Обучение студента ---
def train_student(model, loader, epochs=20, lr=5e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(
        list(model.msf.parameters()) + list(model.oce.parameters()) + list(model.student.parameters()), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            t_feats, s_feats = model(imgs)
            loss = distillation_loss(t_feats, s_feats)
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total/len(loader.dataset):.4f}")

def generate_and_evaluate(model, noise_gen, loader, device='cuda'):
    """
    Генерация синтетических аномалий, сбор скорингов для нормальных и аномальных
    примеров и оценка ROC-AUC.
    """
    model.eval()
    noise_gen.eval()
    scores, labels = [], []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)

            # 1) Оцениваем «нормальные» изображения
            t_feats_clean = model.teacher(imgs)
            fused_clean   = model.msf(t_feats_clean)
            bneck_clean   = model.oce(fused_clean)
            s_feats_clean = model.student(bneck_clean)

            # получаем вектор скорингов (малая потеря на норме)
            clean_scores = distillation_loss(t_feats_clean, s_feats_clean, reduction='none')
            scores += clean_scores.cpu().tolist()
            labels += [0] * clean_scores.size(0)

            # 2) Оцениваем «аномальные» (синтетические) изображения
            anom = noise_gen(imgs)
            t_feats_anom = model.teacher(anom)
            fused_anom   = model.msf(t_feats_anom)
            bneck_anom   = model.oce(fused_anom)
            s_feats_anom = model.student(bneck_anom)

            # большая потеря для аномалий → метка 1
            anom_scores = distillation_loss(t_feats_anom, s_feats_anom, reduction='none')
            scores += anom_scores.cpu().tolist()
            labels += [1] * anom_scores.size(0)

    from sklearn.metrics import roc_auc_score
    # Теперь и чистые, и аномальные примеры присутствуют
    auc = roc_auc_score(labels, scores)
    print(f"Synthetic Anomaly ROC-AUC: {auc:.3f}")

# --- Основной запуск ---
if __name__ == '__main__':
    import torch.multiprocessing as mp
    # mp.set_start_method('fork', force=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    train_ds = FlatFolderDataset('data/human', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    model = ReverseDistillationModel('resnet50', pretrained=True)
    train_student(model, train_loader, epochs=20, lr=5e-4, device=device)
    noise_gen = NoiseGenerator(channels=3).to(device)
    generate_and_evaluate(model, noise_gen, train_loader, device=device)