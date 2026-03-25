"""
实验1：YOLO 推理基础用法
学习目标：
  1. 加载预训练模型
  2. 对图片运行推理
  3. 解析 Results 对象（boxes / conf / cls / xyxy）
  4. 保存可视化结果
"""

from ultralytics import YOLO
from pathlib import Path

# ── 1. 加载模型 ──────────────────────────────────────────────
# 首次运行会自动从网络下载权重到 ~/.cache/ultralytics/
# 模型规格：n(nano) < s < m < l < x(extra-large)，越大越准但越慢
# model = YOLO("yolo11n.pt")   # 用 YOLO11 nano，最轻量
model = YOLO("/vepfs-mlp2/queue010/20252203765/ultralytics/yolo26x.pt")
# model = YOLO("yolo26n.pt")

# ── 2. 运行推理 ──────────────────────────────────────────────
IMG = "/vepfs-mlp2/queue010/20252203765/yolo_learn/assets/bus.jpg"

results = model.predict(
    source=IMG,
    conf=0.25,      # 置信度阈值，低于此值的检测框被过滤
    iou=0.7,        # NMS IoU 阈值，越小框越少（去重更激进）
    imgsz=640,      # 推理时的图片尺寸（会自动 letterbox 缩放）
    device=0,       # 使用 GPU 0；改成 "cpu" 则用 CPU
    verbose=True,   # 打印推理速度信息
)

# ── 3. 解析 Results 对象 ─────────────────────────────────────
# results 是一个列表，每张图片对应一个 Results 对象
r = results[0]

print("\n" + "="*50)
print(f"图片尺寸 (H, W): {r.orig_shape}")
print(f"检测到的目标数量: {len(r.boxes)}")
print("="*50)

# r.boxes 包含所有检测框
boxes = r.boxes

# 坐标格式
xyxy  = boxes.xyxy    # [x1, y1, x2, y2]  左上角+右下角，像素坐标
xywh  = boxes.xywh    # [cx, cy, w, h]     中心点+宽高，像素坐标
xyxyn = boxes.xyxyn   # 归一化到 [0,1] 的 xyxy

conf  = boxes.conf    # 置信度 shape: (N,)
cls   = boxes.cls     # 类别 ID  shape: (N,)

print("\n各检测框详情：")
for i in range(len(boxes)):
    class_id   = int(cls[i].item())
    class_name = r.names[class_id]
    confidence = conf[i].item()
    x1, y1, x2, y2 = xyxy[i].tolist()

    print(f"  [{i}] {class_name:12s}  conf={confidence:.2f}  "
          f"box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# ── 4. 保存可视化结果 ────────────────────────────────────────
OUT_DIR = Path("/vepfs-mlp2/queue010/20252203765/yolo_learn/output")
OUT_DIR.mkdir(exist_ok=True)

# plot() 返回 BGR numpy array，已画好框和标签
annotated = r.plot(
    conf=True,       # 显示置信度
    line_width=2,    # 框线宽度
    font_size=12,    # 标签字体大小
)

import cv2
out_path = OUT_DIR / "bus_result.jpg"
cv2.imwrite(str(out_path), annotated)
print(f"\n结果已保存到: {out_path}")

# ── 5. 演示：提取用于机械臂的关键信息 ───────────────────────
# 实际抓取场景中，你需要的是目标的像素坐标和类别
print("\n[机械臂场景] 目标中心点坐标（像素）：")
for i in range(len(boxes)):
    class_name = r.names[int(cls[i].item())]
    cx, cy, w, h = xywh[i].tolist()
    print(f"  {class_name}: center=({cx:.0f}, {cy:.0f})  size=({w:.0f}x{h:.0f})")
