"""
实验4：模型导出（Export）
学习目标：
  1. 导出 ONNX（通用跨平台格式）
  2. 导出 TensorRT engine（GPU 最高性能）
  3. 用导出的模型推理（接口完全相同）
  4. 对比 PT / ONNX / TensorRT 推理速度
"""

import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

IMG  = "/vepfs-mlp2/queue010/20252203765/ultralytics/ultralytics/assets/bus.jpg"
OUT  = Path("/vepfs-mlp2/queue010/20252203765/yolo_learn/output")
OUT.mkdir(exist_ok=True)

# ── 0. 准备：加载原始 PT 模型 ────────────────────────────────
model_pt = YOLO("yolo11n.pt")

# ── 1. 导出 ONNX ─────────────────────────────────────────────
print("=" * 60)
print("Step 1: 导出 ONNX")
print("=" * 60)

onnx_path = model_pt.export(
    format="onnx",
    imgsz=640,        # 导出时固定输入尺寸
    half=False,       # FP32；True 导出 FP16（需要支持的硬件）
    dynamic=False,    # False=静态 batch=1；True=动态 batch（灵活但稍慢）
    simplify=True,    # 用 onnxsim 简化计算图，推荐开启
    opset=11,         # ONNX opset 版本，11 兼容性最好
)
print(f"\nONNX 模型已保存: {onnx_path}")

# ── 2. 导出 TensorRT engine ──────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: 导出 TensorRT Engine（FP16）")
print("=" * 60)
print("注意：首次导出需要编译，耗时约 1~5 分钟，之后可直接加载")

engine_path = model_pt.export(
    format="engine",
    imgsz=640,
    half=True,        # FP16，速度约为 FP32 的 2x，精度损失极小
    device="0",       # 字符串形式，避免后台进程 CUDA 初始化问题
    simplify=True,
    # workspace=4,    # GPU workspace 内存限制（GB），默认 4
    # int8=True,      # INT8 量化，速度更快但需要校准数据集
)
print(f"\nTensorRT engine 已保存: {engine_path}")

# ── 3. 加载导出的模型做推理（接口完全一致）────────────────────
print("\n" + "=" * 60)
print("Step 3: 加载导出模型推理（接口与 PT 完全相同）")
print("=" * 60)

model_onnx   = YOLO(str(onnx_path))
model_engine = YOLO(str(engine_path))

for name, model in [("ONNX", model_onnx), ("TensorRT", model_engine)]:
    results = model.predict(IMG, conf=0.25, device="0", verbose=False)
    r = results[0]
    print(f"\n[{name}] 检测到 {len(r.boxes)} 个目标:")
    for i in range(len(r.boxes)):
        cls_name = r.names[int(r.boxes.cls[i].item())]
        conf = r.boxes.conf[i].item()
        print(f"  {cls_name}: conf={conf:.2f}")

# ── 4. 速度基准测试 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: 推理速度对比（warmup 5次，测试 50次）")
print("=" * 60)

frame = cv2.imread(IMG)
WARMUP = 5
RUNS   = 50

results_table = {}

for name, model in [
    ("PyTorch PT ", model_pt),
    ("ONNX      ", model_onnx),
    ("TensorRT  ", model_engine),
]:
    # warmup
    for _ in range(WARMUP):
        model.predict(frame, conf=0.25, device="0", verbose=False)

    # benchmark
    t0 = time.perf_counter()
    for _ in range(RUNS):
        model.predict(frame, conf=0.25, device="0", verbose=False)
    elapsed = (time.perf_counter() - t0) / RUNS * 1000  # ms per frame

    fps = 1000 / elapsed
    results_table[name] = (elapsed, fps)
    print(f"  {name}: {elapsed:6.2f} ms/frame  ({fps:.1f} FPS)")

# 计算加速比
pt_ms = results_table["PyTorch PT "][0]
print("\n加速比（相对 PyTorch PT）:")
for name, (ms, fps) in results_table.items():
    speedup = pt_ms / ms
    print(f"  {name}: {speedup:.2f}x")

print(f"""
【导出格式选择指南】
  PT（PyTorch）  : 开发调试首选，灵活，无需编译
  ONNX           : 跨平台部署（CPU/GPU均可），推理速度约 2~3x PT
  TensorRT FP16  : NVIDIA GPU 最高性能，速度约 3~5x PT，生产部署首选
  TensorRT INT8  : 更快（需校准），适合实时性要求极高的场景

【Franka + Realsense 高速抓取建议】
  - 部署阶段用 TensorRT FP16 engine
  - engine 文件与 GPU 型号绑定，换机器需重新导出
  - 导出一次，反复使用，加载速度比 PT 快很多
""")
