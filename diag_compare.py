"""诊断脚本：比较 ONNX 和 TRT 对同一张图的特征输出"""
import ctypes

import numpy as np
import onnxruntime as ort
import cv2
import tensorrt as trt
from dinov3_trt_infer import DINOv3TensorRT, preprocess

# Load the IdentityBarrier plugin so its creator is registered before we
# deserialize an engine that uses it.
ctypes.CDLL("./libidentity_barrier.so")
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")

IMG_PATH   = "reference_image.jpg"
ONNX_PATH  = "dinov3_vits16_features_512.onnx"
TRT_PATH   = "dinov3_vits16_features_512_plugin.engine"

def stats(name, feats):
    flat  = feats[0].reshape(-1, 384)           # [1024, 384]
    norms = np.linalg.norm(flat, axis=1)        # L2 norm per patch
    # 所有 patch 与 patch(0,0) 的余弦相似度（两者均为单位向量时 = cosine sim）
    sims  = flat @ flat[0:1].T                  # [1024, 1]
    print(f"\n=== {name} ===")
    print(f"  feature shape : {feats.shape}")
    print(f"  patch norms   : min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}  std={norms.std():.6f}")
    print(f"  inter-patch sim (vs patch[0,0]): min={sims.min():.4f}  mean={sims.mean():.4f}  max={sims.max():.4f}")
    print(f"  patch[0,0] first 8 vals: {flat[0, :8]}")
    print(f"  patch[31,31] first 8 vals: {flat[-1, :8]}")
    return flat

# --- ONNX ---
sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
img  = cv2.imread(IMG_PATH)
inp  = preprocess(img).astype(np.float32)
onnx_feats = sess.run(None, {"input": inp})[0]
flat_onnx  = stats("ONNX", onnx_feats)

# --- TRT ---
model     = DINOv3TensorRT(TRT_PATH)
trt_feats = model.infer(preprocess(img))
flat_trt  = stats("TRT", trt_feats)

# --- 直接对比 ---
diff = flat_onnx - flat_trt
print(f"\n=== ONNX vs TRT diff ===")
print(f"  Max abs diff  : {np.abs(diff).max():.6f}")
print(f"  Mean abs diff : {np.abs(diff).mean():.6f}")
print(f"  Corr (patch[0]): {np.corrcoef(flat_onnx[0], flat_trt[0])[0,1]:.6f}")
