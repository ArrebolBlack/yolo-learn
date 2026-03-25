"""
实验3：自定义数据集训练/微调
学习目标：
  1. 理解 YOLO 数据集目录结构和标注格式
  2. 编写 dataset.yaml
  3. 用预训练权重微调（fine-tune）
  4. 理解训练输出：指标、权重保存位置
  5. 加载训练好的模型做推理

【数据集格式说明】
  YOLO 格式标注（.txt）：每行一个目标
    <class_id> <cx> <cy> <w> <h>
  所有值归一化到 [0, 1]，相对于图片宽高
  例：0 0.5 0.45 0.8 0.6  → class=0, 中心(50%,45%), 宽80%, 高60%
"""

import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── 1. 构建模拟自定义数据集 ──────────────────────────────────
# 目录结构（YOLO 标准格式）：
#   my_dataset/
#     images/
#       train/  *.jpg
#       val/    *.jpg
#     labels/
#       train/  *.txt   （与 images/train/ 同名，扩展名换成 .txt）
#       val/    *.txt

DATASET_ROOT = Path("/vepfs-mlp2/queue010/20252203765/yolo_learn/my_dataset")
SRC_IMG = Path("/vepfs-mlp2/queue010/20252203765/ultralytics/ultralytics/assets/bus.jpg")

for split in ["train", "val"]:
    (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# 复制图片，train 4张，val 2张（实际项目中换成你自己的图片）
img = cv2.imread(str(SRC_IMG))
H, W = img.shape[:2]

for split, count in [("train", 4), ("val", 2)]:
    for i in range(count):
        # 轻微扰动模拟不同图片
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        aug = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(DATASET_ROOT / "images" / split / f"img_{i:03d}.jpg"), aug)

        # 对应的标注文件（YOLO 格式）
        # 这里手动写 bus.jpg 里已知的目标框（从实验1的结果得到）
        # 格式：class_id cx cy w h（归一化）
        labels = [
            # bus: box=[4,229,796,728] in 810x1080
            f"5 {(4+796)/2/W:.6f} {(229+728)/2/H:.6f} {(796-4)/W:.6f} {(728-229)/H:.6f}",
            # person: box=[671,395,810,879]
            f"0 {(671+810)/2/W:.6f} {(395+879)/2/H:.6f} {(810-671)/W:.6f} {(879-395)/H:.6f}",
            # person: box=[47,400,239,904]
            f"0 {(47+239)/2/W:.6f} {(400+904)/2/H:.6f} {(239-47)/W:.6f} {(904-400)/H:.6f}",
        ]
        label_path = DATASET_ROOT / "labels" / split / f"img_{i:03d}.txt"
        label_path.write_text("\n".join(labels))

print(f"数据集已创建: {DATASET_ROOT}")
print(f"  train: {len(list((DATASET_ROOT/'images'/'train').glob('*.jpg')))} 张图片")
print(f"  val:   {len(list((DATASET_ROOT/'images'/'val').glob('*.jpg')))} 张图片")

# ── 2. 编写 dataset.yaml ─────────────────────────────────────
yaml_content = f"""# 自定义数据集配置
path: {DATASET_ROOT}   # 数据集根目录（绝对路径）
train: images/train    # 相对于 path 的训练集路径
val: images/val        # 相对于 path 的验证集路径

nc: 2                  # 类别数量（number of classes）
names:                 # 类别名称，索引对应标注文件中的 class_id
  0: person
  5: bus
"""
# 注意：names 的 key 必须与标注文件中的 class_id 一致
# 更常见的写法是从 0 开始连续编号：
yaml_content = f"""# 自定义数据集配置
path: {DATASET_ROOT}
train: images/train
val: images/val

nc: 2
names:
  0: person
  1: bus
"""
# 对应地，标注文件里 bus 用 class_id=1，person 用 class_id=0

# 重新写标注（用连续 ID）
for split in ["train", "val"]:
    count = 4 if split == "train" else 2
    for i in range(count):
        labels = [
            f"1 {(4+796)/2/W:.6f} {(229+728)/2/H:.6f} {(796-4)/W:.6f} {(728-229)/H:.6f}",  # bus=1
            f"0 {(671+810)/2/W:.6f} {(395+879)/2/H:.6f} {(810-671)/W:.6f} {(879-395)/H:.6f}",  # person=0
            f"0 {(47+239)/2/W:.6f} {(400+904)/2/H:.6f} {(239-47)/W:.6f} {(904-400)/H:.6f}",   # person=0
        ]
        (DATASET_ROOT / "labels" / split / f"img_{i:03d}.txt").write_text("\n".join(labels))

yaml_path = DATASET_ROOT / "dataset.yaml"
yaml_path.write_text(yaml_content)
print(f"\ndataset.yaml 已写入: {yaml_path}")
print(yaml_content)

# ── 3. 微调训练 ──────────────────────────────────────────────
print("=" * 60)
print("开始微调训练（3 epochs，仅用于演示流程）")
print("=" * 60)

model = YOLO("yolo11n.pt")   # 加载预训练权重

results = model.train(
    data=str(yaml_path),     # 数据集配置文件
    epochs=3,                # 实际项目建议 50~200
    imgsz=640,               # 训练图片尺寸
    batch=4,                 # batch size（小数据集用小 batch）
    device=0,                # GPU
    project="/vepfs-mlp2/queue010/20252203765/yolo_learn/runs",  # 保存目录
    name="finetune_demo",    # 实验名称
    exist_ok=True,           # 允许覆盖已有实验目录
    verbose=True,
    # 微调时常用的额外参数：
    # freeze=10,             # 冻结前 10 层（只训练后面的层，适合小数据集）
    # lr0=0.001,             # 微调时学习率通常比从头训练小 10x
    # patience=20,           # early stopping
)

# ── 4. 查看训练结果 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("训练完成，查看结果")
print("=" * 60)

save_dir = Path(results.save_dir)
print(f"\n所有输出保存在: {save_dir}")
print("\n目录内容:")
for f in sorted(save_dir.iterdir()):
    print(f"  {f.name}")

# 最佳权重和最后权重
best_pt  = save_dir / "weights" / "best.pt"
last_pt  = save_dir / "weights" / "last.pt"
print(f"\n最佳权重: {best_pt}  (exists={best_pt.exists()})")
print(f"最后权重: {last_pt}  (exists={last_pt.exists()})")

# 训练指标
print(f"\n最终训练指标:")
print(f"  box_loss:  {results.results_dict.get('train/box_loss', 'N/A')}")
print(f"  cls_loss:  {results.results_dict.get('train/cls_loss', 'N/A')}")
print(f"  mAP50:     {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"  mAP50-95:  {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

# ── 5. 用训练好的模型推理 ────────────────────────────────────
print("\n" + "=" * 60)
print("用训练好的模型推理")
print("=" * 60)

trained_model = YOLO(str(best_pt))
test_img = str(DATASET_ROOT / "images" / "val" / "img_000.jpg")
results_infer = trained_model.predict(test_img, conf=0.1, device=0, verbose=False)

r = results_infer[0]
print(f"\n检测到 {len(r.boxes)} 个目标:")
for i in range(len(r.boxes)):
    name = r.names[int(r.boxes.cls[i].item())]
    conf = r.boxes.conf[i].item()
    print(f"  {name}: conf={conf:.2f}")
