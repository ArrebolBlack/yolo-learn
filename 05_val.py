"""
实验5：验证（Val）与指标解读
学习目标：
  1. model.val() 的用法
  2. 理解 mAP / Precision / Recall / F1 含义
  3. 读懂 per-class 指标，判断哪些类别检测效果差
  4. 如何用指标决策：模型是否够用、需要更多数据还是更大模型
"""

from ultralytics import YOLO
from pathlib import Path

# ── 1. 基础验证用法 ──────────────────────────────────────────
print("=" * 60)
print("Step 1: 在 coco8 上验证预训练模型")
print("=" * 60)

model = YOLO("yolo11n.pt")

# val() 会自动下载 coco8 数据集（8张图，很快）
metrics = model.val(
    data="coco8.yaml",   # 数据集配置
    imgsz=640,
    conf=0.001,          # val 时用极低 conf，让所有检测框参与 PR 曲线计算
    iou=0.6,             # NMS IoU 阈值
    device="0",
    verbose=True,
)

# ── 2. 提取关键指标 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: 关键指标提取")
print("=" * 60)

# box 指标（检测任务）
box = metrics.box

print(f"\n【整体指标】")
print(f"  mAP50      = {box.map50:.4f}   (IoU=0.5 时的 mAP，宽松标准)")
print(f"  mAP50-95   = {box.map:.4f}   (IoU=0.5~0.95 均值，严格标准，主要指标)")
print(f"  Precision  = {box.mp:.4f}   (精确率：检测到的框中有多少是对的)")
print(f"  Recall     = {box.mr:.4f}   (召回率：所有真实目标中有多少被检测到)")

print(f"\n【每类别 mAP50-95】")
for i, (name, ap) in enumerate(zip(metrics.names.values(), box.maps)):
    if ap > 0:
        print(f"  {name:20s}: {ap:.4f}")

# ── 3. 验证训练好的自定义模型 ────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: 验证实验3训练的自定义模型")
print("=" * 60)

best_pt = "/vepfs-mlp2/queue010/20252203765/yolo_learn/runs/finetune_demo/weights/best.pt"
custom_model = YOLO(best_pt)

custom_metrics = custom_model.val(
    data="/vepfs-mlp2/queue010/20252203765/yolo_learn/my_dataset/dataset.yaml",
    imgsz=640,
    conf=0.001,
    iou=0.6,
    device="0",
    verbose=False,
)

cb = custom_metrics.box
print(f"\n自定义模型（3 epoch 演示）指标：")
print(f"  mAP50    = {cb.map50:.4f}")
print(f"  mAP50-95 = {cb.map:.4f}")
print(f"  Precision= {cb.mp:.4f}")
print(f"  Recall   = {cb.mr:.4f}")

print(f"\n每类别：")
for name, ap in zip(custom_metrics.names.values(), cb.maps):
    print(f"  {name:10s}: mAP50-95 = {ap:.4f}")

# ── 4. 指标解读指南 ──────────────────────────────────────────
print("""
============================================================
指标解读指南
============================================================

【mAP50-95 参考标准】
  < 0.30   : 模型效果差，需要更多数据或更大模型
  0.30~0.50: 基本可用，适合精度要求不高的场景
  0.50~0.70: 良好，大多数工业场景够用
  > 0.70   : 优秀，接近 SOTA

【Precision vs Recall 权衡】
  高 Precision 低 Recall → 漏检多，但误检少（适合：不能误报的场景）
  低 Precision 高 Recall → 误检多，但漏检少（适合：不能漏检的场景）
  机械臂抓取建议：提高 Recall（宁可多检测，不能漏掉目标）
    → 降低 conf 阈值（如 conf=0.15）

【per-class 指标用途】
  某类 mAP 很低 → 该类数据不足或标注质量差，针对性补充数据
  某类 Recall 低 → 该类目标经常漏检，考虑降低该类的 conf 阈值

【判断是否需要更多数据】
  训练 loss 低但 val mAP 低 → 过拟合，需要更多数据或数据增强
  训练 loss 和 val loss 都高 → 欠拟合，需要更大模型或更多 epoch
  val mAP 稳定不再提升      → 数据质量瓶颈，检查标注准确性
""")
