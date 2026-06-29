import torch
import torch.nn as nn
from pathlib import Path

class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 获取中间层特征，n=1表示最后一层
        feats = self.model.get_intermediate_layers(x, n=1, reshape=True)[0]
        # [B, C, H_p, W_p] -> [B, H_p, W_p, C]
        feats = feats.permute(0, 2, 3, 1)
        # L2 归一化，余弦相似度计算
        feats = torch.nn.functional.normalize(feats, dim=-1, p=2)
        return feats

if __name__ == "__main__":
    print("[INFO] 正在加载 DINOv3 ViT-S/16 预训练模型...")
    # 加载预训练模型 (LVD-1689M 权重)
    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir / "dinov3"
    weights_path = base_dir / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    if not repo_dir.exists():
        raise FileNotFoundError(f"DINOv3 repo directory not found: {repo_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"DINOv3 weights file not found: {weights_path}")

    # 加载预训练模型
    base_model = torch.hub.load(
        repo_or_dir=str(repo_dir),
        model="dinov3_vits16",
        source="local",
        pretrained=False,
    )
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    extractor = DINOv3FeatureExtractor(base_model)

    # 创建 dummy input (例如 512x512 分辨率)
    # 注意：长宽必须是 16 的倍数
    H, W = 512, 512
    print(f"[INFO] 创建尺寸为 {H}x{W} 的虚拟输入...")
    dummy_input = torch.randn(1, 3, H, W)

    output_path = "dinov3_vits16_features_512.onnx"
    print(f"[INFO] 开始导出 ONNX 模型至 {output_path} ...")
    
    # 导出 ONNX
    torch.onnx.export(
        extractor,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["patch_features"],
        opset_version=17,
        # dynamic_axes={"input": {0: "batch_size"}, "patch_features": {0: "batch_size"}}
    )
    print(f"[SUCCESS] ONNX 导出完成: {output_path}")
