# DINOv3 ViT-S/16 转onnx和tensorrt推理

---

## 1. 环境

- JetPack 6.2 / L4T R36.4.3，TensorRT 10.3.0.30。

---

## 2. Step 1 — 导出 ONNX（在 x86）

```bash
python3 export_onnx.py
```
- 输入固定 `1x3x512x512`（长宽须为 16 的倍数）


---

## 3. Step 2 — 编译 IdentityBarrier plugin（在 Jetson）

```bash
nvcc -shared -std=c++17 -o libidentity_barrier.so identity_barrier_plugin.cu \
     -Xcompiler -fPIC \
     -I/usr/include/aarch64-linux-gnu \
     -L/usr/lib/aarch64-linux-gnu -lnvinfer -lcudart
```
- 成功后生成 `libidentity_barrier.so`，推理时候需要用到
- 或者直接使用我已经编译好的so

---

## 4. Step 3 — 插入 plugin barrier（在 Jetson）

```bash
python3 insert_plugin_barrier.py \
  --in  dinov3_vits16_features_512.onnx \
  --out dinov3_vits16_features_512_plugin.onnx
```
期望输出：
```
Found 12 QK^T and 12 attn@V matmul nodes.
Inserted 60 IdentityBarrier plugin nodes.
Saved patched model -> dinov3_vits16_features_512_plugin.onnx
```

---

## 5. Step 4 — 构建 TRT 引擎（在 Jetson）

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=dinov3_vits16_features_512_plugin.onnx \
  --saveEngine=dinov3_vits16_features_512_plugin.engine \
  --noTF32 \
  --staticPlugins=./libidentity_barrier.so
```
- **`--staticPlugins=./libidentity_barrier.so` 必加**，否则 ONNX 解析器找不到 `IdentityBarrier` 算子

---

## 6. Step 5 — 数值验证（在 Jetson）

```bash
python3 diag_compare.py
```
**通过标准**（实测值）：
```
=== ONNX vs TRT diff ===
  Max abs diff  : 0.000002
  Mean abs diff : 0.000000
  Corr (patch[0]): 1.000000
```
- `Corr ≈ 1.0`（与 ONNX 一致）即为成功。

---
