"""2단계 분류 모델 (학습 스크립트와 같은 구조)

ViTCBAM   : ViT-B/16 패치 토큰(196 x 768) -> 768 x 14 x 14 특징 맵 -> CBAM -> 전역 평균 풀링 -> 분류 헤드
ResNetCls : ResNet50 (2048차원 특징) -> 분류 헤드
분류 헤드 : BatchNorm -> Dropout(0.4) -> Linear(dim->128) -> ReLU -> Linear(128->5)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, vit_b_16

IMG_SIZE = 224


def make_head(dim, num_classes):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        nn.Dropout(0.4),
        nn.Linear(dim, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes),
    )


class ResNetCls(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        weights = None
        if pretrained:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2
        m = resnet50(weights=weights)
        m.fc = nn.Identity()
        self.backbone = m
        self.head = make_head(2048, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


class ChannelAttention(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // ratio, ch, 1, bias=False),
        )

    def forward(self, x):
        return x * torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.mx(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ViTCBAM(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        weights = None
        if pretrained:
            from torchvision.models import ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        vit = vit_b_16(weights=weights)
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        self.hidden = 768
        self.grid = IMG_SIZE // 16
        self.cbam = CBAM(self.hidden)
        self.head = make_head(self.hidden, num_classes)

    def forward(self, x):
        n = x.shape[0]
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)
        x = self.encoder(x)
        fmap = x[:, 1:, :].transpose(1, 2).reshape(n, self.hidden, self.grid, self.grid)
        fmap = self.cbam(fmap)
        return self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(fmap, 1), 1))


STAGE2_MODELS = {"resnet50": ResNetCls, "vit_cbam": ViTCBAM}


def build_stage2(name, num_classes=5, pretrained=False):
    return STAGE2_MODELS[name](num_classes=num_classes, pretrained=pretrained)
