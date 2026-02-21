import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks      = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_dropout(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.norm(x)
        return self.head(x[:, 0])


def vit_tiny(num_classes=10, img_size=32, patch_size=4):
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1,
    )

def vit_small(num_classes=1000):
    return VisionTransformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, num_classes=num_classes)

def vit_base(num_classes=1000):
    return VisionTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, num_classes=num_classes)

def vit_large(num_classes=1000):
    return VisionTransformer(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_classes=num_classes)


def get_cifar10_loaders(batch_size=128, img_size=32):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def run_training(num_epochs=30, batch_size=128, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = vit_tiny(num_classes=10, img_size=32, patch_size=4).to(device)
    train_loader, val_loader = get_cifar10_loaders(batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc  = 0.0
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "best_vit.pth")
        print(f"Epoch [{epoch:3d}/{num_epochs}]  Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.2f}%")
    return model


CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

@torch.no_grad()
def predict(model, image_tensor, class_names=None, device="cpu"):
    model.eval().to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    probs = F.softmax(model(image_tensor.to(device)), dim=-1)[0]
    top5  = probs.topk(5)
    for prob, idx in zip(top5.values, top5.indices):
        label = class_names[idx] if class_names else str(idx.item())
        print(f"  {label:<15} {prob.item()*100:.2f}%")
    pred_idx = probs.argmax().item()
    return pred_idx, (class_names[pred_idx] if class_names else pred_idx)


if __name__ == "__main__":
    model = vit_tiny(num_classes=10, img_size=32, patch_size=4)
    dummy = torch.randn(4, 3, 32, 32)
    out   = model(dummy)
    print(f"Input  shape : {list(dummy.shape)}")
    print(f"Output shape : {list(out.shape)}")
    print(f"Params       : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")