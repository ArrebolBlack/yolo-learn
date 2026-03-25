# Ultralytics YOLO 使用指南

> 面向具身智能项目（Franka 机械臂 + Realsense 相机）的实战记录
> 覆盖：推理 → 跟踪 → 训练 → 导出 → 验证 → 模型结构修改

---

## 目录

1. [环境配置](#1-环境配置)
2. [核心概念速览](#2-核心概念速览)
3. [实验1：推理（Inference）](#3-实验1推理inference)
4. [实验2：目标跟踪（Tracking）](#4-实验2目标跟踪tracking)
5. [实验3：自定义数据集训练/微调](#5-实验3自定义数据集训练微调)
6. [实验4：模型导出（ONNX / TensorRT）](#6-实验4模型导出onnx--tensorrt)
7. [实验5：验证与指标解读](#7-实验5验证与指标解读)
8. [实验6：修改模型结构](#8-实验6修改模型结构)
9. [机械臂抓取场景速查](#9-机械臂抓取场景速查)

---

## 1. 环境配置

### 硬件

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA A100-SXM4-80GB × 2 |
| CUDA | 12.2 / Driver 535.129.03 |

### 安装

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# 安装 PyTorch（CUDA 12.1 兼容 12.2）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 ultralytics（开发模式，方便修改源码）
pip install -e /path/to/ultralytics

# 安装 OpenCV（headless，无 GUI 依赖）
pip install opencv-python-headless

# 跟踪器依赖（首次用 track() 时会自动安装，也可提前装）
pip install lap>=0.5.12
```

### 验证

```python
import torch, cv2, ultralytics
print(torch.__version__)            # 2.5.1+cu121
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name(0))
print(cv2.__version__)              # 4.13.0
print(ultralytics.__version__)      # 8.4.24
```

### 软件版本

| 库 | 版本 |
|----|------|
| Python | 3.11.15 |
| torch | 2.5.1+cu121 |
| ultralytics | 8.4.24 |
| opencv-python-headless | 4.13.0 |
| numpy | 2.4.3 |

> **`pip install -e`（开发模式）**：修改源码后无需重新安装即可生效，适合后续学习模型结构时使用。

---

## 2. 核心概念速览

### 模型命名规则

```
yolo11n.pt
│    │ └── 后缀：.pt = PyTorch权重，.yaml = 结构定义
│    └──── 规格：n < s < m < l < x（nano 到 extra-large）
└───────── 版本：11 = YOLO11，26 = YOLO26，...
```

### 支持的任务类型

| 任务 | YAML | 说明 |
|------|------|------|
| detect | `yolo11n.yaml` | 目标检测（默认） |
| segment | `yolo11n-seg.yaml` | 实例分割 |
| pose | `yolo11n-pose.yaml` | 姿态估计 |
| classify | `yolo11n-cls.yaml` | 图像分类 |
| obb | `yolo11n-obb.yaml` | 旋转框检测 |

### 统一 API 结构

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # 加载

model.predict(...)   # 推理
model.track(...)     # 跟踪
model.train(...)     # 训练
model.val(...)       # 验证
model.export(...)    # 导出
model.info(...)      # 查看结构
```

---

## 3. 实验1：推理（Inference）

### 核心代码

```python
from ultralytics import YOLO

# 加载模型（首次自动下载权重到 ~/.cache/ultralytics/）
model = YOLO("yolo11n.pt")

# 推理
results = model.predict(
    source="image.jpg",
    conf=0.25,     # 置信度阈值，低于此值的框被过滤
    iou=0.7,       # NMS IoU 阈值，越小框越少
    imgsz=640,     # 推理图片尺寸（自动 letterbox 缩放）
    device="0",    # GPU；"cpu" 用 CPU
    verbose=True,
)

# 解析结果
r = results[0]
boxes = r.boxes

xyxy  = boxes.xyxy    # [x1, y1, x2, y2] 像素坐标，shape (N,4)
xywh  = boxes.xywh    # [cx, cy, w, h]   中心点+宽高，shape (N,4)
xyxyn = boxes.xyxyn   # 归一化到 [0,1] 的 xyxy
conf  = boxes.conf    # 置信度，shape (N,)
cls   = boxes.cls     # 类别 ID，shape (N,)

for i in range(len(boxes)):
    name = r.names[int(cls[i].item())]
    print(f"{name}: conf={conf[i]:.2f}, box={xyxy[i].tolist()}")

# 保存可视化（返回 BGR numpy array）
import cv2
cv2.imwrite("result.jpg", r.plot())
```

### 支持的输入源

```python
model.predict("image.jpg")                    # 单张图片
model.predict(["a.jpg", "b.jpg"])             # 批量图片
model.predict("video.mp4", stream=True)       # 视频（★ stream=True 节省内存）
model.predict(0, stream=True)                 # 摄像头（设备 ID）
model.predict(frame)                          # numpy array（OpenCV 帧）
model.predict("path/to/dir/*.jpg")            # glob 匹配
```

> **`stream=True` 很重要**：处理视频/摄像头时必须加，否则所有帧结果堆积在内存。

### 实测结果（bus.jpg，yolo11n，A100）

| 目标 | 置信度 | 中心点 (cx, cy) | 尺寸 (w×h) |
|------|--------|-----------------|------------|
| bus | 0.94 | (400, 479) | 792×499 |
| person | 0.89 | (740, 637) | 139×484 |
| person | 0.88 | (143, 652) | 192×505 |
| person | 0.86 | (284, 635) | 121×452 |
| person | 0.62 | (34, 714) | 69×316 |

**推理速度**：preprocess 13ms + inference 39ms + postprocess 36ms

### 机械臂场景关键字段

```python
# 目标中心点（像素），后续配合相机内参做 2D→3D 投影
cx, cy, w, h = r.boxes.xywh[i].tolist()

# 归一化坐标（与分辨率无关，更通用）
x1n, y1n, x2n, y2n = r.boxes.xyxyn[i].tolist()
```

---

## 4. 实验2：目标跟踪（Tracking）

### predict() vs track()

| | `predict()` | `track()` |
|---|---|---|
| 每帧独立 | ✅ | ✅ |
| 跨帧持久 ID | ❌ | ✅ |
| `boxes.id` | `None` | 整数 track_id |

### 核心代码

```python
model = YOLO("yolo11n.pt")

# 视频帧循环（Realsense 场景）
for frame in camera_stream:
    results = model.track(
        source=frame,
        tracker="bytetrack.yaml",  # 或 "botsort.yaml"
        persist=True,   # ★ 必须 True，否则每帧重置 ID
        conf=0.25,
        device="0",
        verbose=False,
    )

    r = results[0]
    if r.boxes.id is not None:
        for i in range(len(r.boxes)):
            track_id   = int(r.boxes.id[i].item())   # 持久 ID
            class_name = r.names[int(r.boxes.cls[i].item())]
            cx, cy, w, h = r.boxes.xywh[i].tolist()

# 新场景开始时重置跟踪状态
model.predictor = None
```

### 跟踪器对比

| | ByteTrack | BoT-SORT |
|---|---|---|
| 关联方式 | IoU（运动） | IoU + 全局运动补偿（GMC） |
| 速度 | 更快 | 稍慢 |
| 相机移动 | 一般 | 更好 |
| ReID 支持 | ❌ | ✅ |
| **推荐** | 固定相机 | 相机移动/高速/遮挡多 |

### 关键参数（高速移动物体调优）

```yaml
# bytetrack.yaml 或 botsort.yaml
track_buffer: 30       # 目标消失后保留 ID 的帧数，高速物体可调到 60
track_high_thresh: 0.25
match_thresh: 0.8      # IoU 匹配阈值，高速位移大时可降到 0.6
```

> **Franka 建议**：相机固定 → ByteTrack；相机随臂移动 → BoT-SORT + `gmc_method: sparseOptFlow`

---

## 5. 实验3：自定义数据集训练/微调

### 数据集目录结构

```
my_dataset/
  images/
    train/   img_001.jpg  img_002.jpg ...
    val/     img_001.jpg  img_002.jpg ...
  labels/
    train/   img_001.txt  img_002.txt ...  ← 同名，扩展名换 .txt
    val/     img_001.txt  img_002.txt ...
  dataset.yaml
```

### YOLO 标注格式

每个 `.txt` 文件，每行一个目标：

```
<class_id> <cx> <cy> <w> <h>
```

- 所有值归一化到 `[0, 1]`
- 从像素坐标 `[x1,y1,x2,y2]` 转换：

```python
cx = (x1 + x2) / 2 / img_width
cy = (y1 + y2) / 2 / img_height
w  = (x2 - x1) / img_width
h  = (y2 - y1) / img_height
```

> 标注工具推荐：**Roboflow**（在线，可直接导出 YOLO 格式）或 **LabelImg**（本地）

### dataset.yaml

```yaml
path: /absolute/path/to/my_dataset   # 推荐绝对路径
train: images/train
val: images/val

nc: 2                  # 类别数量
names:
  0: person
  1: bus               # 索引必须从 0 开始连续编号
```

### 训练代码

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # 预训练权重（迁移学习起点）

results = model.train(
    data="my_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="0",
    project="runs",       # 输出根目录
    name="my_exp",        # 实验名 → 输出到 runs/my_exp/
    # ── 微调专用 ──
    # freeze=10,          # 冻结前 N 层（小数据集防过拟合）
    # lr0=0.001,          # 微调时学习率约为从头训练的 1/10
    # patience=20,        # early stopping
)

# 续训
model = YOLO("runs/my_exp/weights/last.pt")
model.train(data="dataset.yaml", epochs=200, resume=True)
```

### 训练输出目录

```
runs/my_exp/
  weights/
    best.pt       ← 最佳权重（部署用这个）
    last.pt       ← 最后 epoch 权重（续训用这个）
  results.csv     ← 每 epoch 完整指标
  results.png     ← loss 和指标曲线图
  confusion_matrix.png
  val_batch0_pred.jpg   ← 验证集预测可视化
  args.yaml       ← 本次训练完整参数记录
```

### 训练指标含义

```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
  1/100  0.64G    1.121     2.786     1.268        29     640
```

| 指标 | 含义 |
|------|------|
| `box_loss` | 边框回归损失 |
| `cls_loss` | 分类损失 |
| `dfl_loss` | Distribution Focal Loss |
| `mAP50-95` | 主要评估指标（越高越好） |

### 针对机械臂场景的建议

| 场景 | 参数建议 |
|------|---------|
| 数据量 < 500 张 | `freeze=10`，只微调 head |
| 目标种类少（1~3 类） | `yolo11n` 或 `yolo11s` 足够 |
| 高速移动 | 数据增强加运动模糊 |
| 类别不平衡 | `cls=2.0` 加大分类损失权重 |

---

## 6. 实验4：模型导出（ONNX / TensorRT）

### 导出命令

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 导出 ONNX
model.export(
    format="onnx",
    imgsz=640,
    half=False,      # FP32；True 导出 FP16
    dynamic=False,   # 静态 batch=1
    simplify=True,   # 简化计算图，推荐开启
    opset=11,
)
# → 输出 yolo11n.onnx

# 导出 TensorRT FP16（耗时约 5~10 分钟，只需一次）
model.export(
    format="engine",
    imgsz=640,
    half=True,       # FP16
    device="0",      # 必须字符串形式
    simplify=True,
)
# → 输出 yolo11n.engine
```

### 加载导出模型（接口完全相同）

```python
model = YOLO("yolo11n.onnx")    # ONNX
model = YOLO("yolo11n.engine")  # TensorRT

results = model.predict("image.jpg", conf=0.25, device="0")
```

### 实测速度（A100，yolo11n，imgsz=640，batch=1）

| 格式 | ms/帧 | FPS | 加速比 |
|------|-------|-----|--------|
| PyTorch PT | 8.70ms | 115 | 1.0x |
| ONNX FP32 | 6.74ms | 148 | 1.3x |
| **TensorRT FP16** | **4.23ms** | **236** | **2.1x** |

### 注意事项

| 注意 | 说明 |
|------|------|
| engine 与 GPU 绑定 | 换 GPU 需重新导出 |
| engine 与 imgsz 绑定 | 导出和推理必须用同一尺寸 |
| FP16 精度损失 | 检测任务 mAP 通常只降 0.1~0.3% |
| 首次 TensorRT 导出 | A100 约 9 分钟，之后加载很快 |

### 格式选择指南

```
开发调试          → PyTorch PT (.pt)
跨平台 / CPU 部署 → ONNX (.onnx)          约 1.3x 加速
GPU 生产部署      → TensorRT (.engine)    约 2~5x 加速（推荐）
嵌入式 / Jetson   → TensorRT INT8         更快，需校准数据集
```

---

## 7. 实验5：验证与指标解读

### 核心代码

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

metrics = model.val(
    data="coco8.yaml",
    imgsz=640,
    conf=0.001,    # ★ val 时用极低 conf，让所有框参与 PR 曲线计算
    iou=0.6,
    device="0",
)

box = metrics.box
print(f"mAP50-95 = {box.map:.4f}")    # 主要指标
print(f"mAP50    = {box.map50:.4f}")
print(f"Precision= {box.mp:.4f}")
print(f"Recall   = {box.mr:.4f}")

# per-class 指标
for name, ap in zip(metrics.names.values(), box.maps):
    if ap > 0:
        print(f"  {name}: {ap:.4f}")
```

### 实测（yolo11n 预训练，coco8 验证集）

| 指标 | 值 |
|------|-----|
| mAP50-95 | 0.632 |
| mAP50 | 0.846 |
| Precision | 0.574 |
| Recall | 0.850 |

### mAP50-95 参考标准

| 范围 | 评价 |
|------|------|
| < 0.30 | 效果差，需更多数据或更大模型 |
| 0.30~0.50 | 基本可用 |
| 0.50~0.70 | 良好，多数工业场景够用 |
| > 0.70 | 优秀 |

### Precision / Recall 权衡

```
高P低R → 漏检多，误检少 → 提高 conf 阈值
低P高R → 误检多，漏检少 → 降低 conf 阈值

机械臂抓取：优先高 Recall（不能漏目标）→ 推荐 conf=0.15~0.20
```

### 问题诊断

| 现象 | 原因 | 解决 |
|------|------|------|
| 训练 loss 低，val mAP 低 | 过拟合 | 更多数据 / 数据增强 |
| 训练+val loss 都高 | 欠拟合 | 更大模型 / 更多 epoch |
| 某类 mAP 特别低 | 该类数据少或标注差 | 补充数据 |
| val mAP 不再提升 | 数据质量瓶颈 | 检查标注准确性 |

---

## 8. 实验6：修改模型结构

### YAML 语法

```yaml
# [from, repeats, module, args]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 上一层 → Conv(out=64, k=3, s=2)
  - [-1, 2, C3k2, [256, False, 0.25]]   # 重复 2 次
  - [-1, 1, SPPF, [1024, 5]]            # 空间金字塔池化

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 上采样 2x
  - [[-1, 6], 1, Concat, [1]]                   # 拼接当前层和第 6 层
  - [[16, 19, 22], 1, Detect, [nc]]             # 3 尺度检测头
```

### scales 缩放控制

```yaml
scales:
  #      [depth, width, max_channels]
  tiny:  [0.33, 0.25, 512]    # 1.5M 参数（自定义轻量）
  n:     [0.50, 0.25, 1024]   # 2.6M 参数（官方 nano）
  s:     [0.50, 0.50, 1024]   # 9.5M 参数
  m:     [0.50, 1.00, 512]    # 20M  参数
```

### 增加 P2 小目标检测头

```yaml
head:
  # ... 上采样到 P3 后继续 ...
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]           # 拼接 backbone P2（stride=4）
  - [-1, 2, C3k2, [128, False]]         # P2 检测头（新增）
  # ...
  - [[P2_idx, P3_idx, P4_idx, P5_idx], 1, Detect, [nc]]   # 4 个尺度
```

> 多 4.4% 参数，可检测更小目标，适合流水线小零件场景。

### 注册自定义模块

```python
# 1. 定义模块
class DSConv(nn.Module):
    """深度可分离卷积，参数比普通卷积少 88%"""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw  = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
        self.pw  = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

# 2. 持久化注册（开发模式安装后修改源码）
# 在 ultralytics/nn/modules/block.py 末尾添加上面的类
# 在 ultralytics/nn/modules/__init__.py 的 __all__ 中加入 "DSConv"

# 3. YAML 中使用
# backbone:
#   - [-1, 1, DSConv, [128, 3, 2]]
```

### 查看模型结构

```python
model = YOLO("yolo11n.pt")
model.info(detailed=False)   # 简要：层数、参数量、GFLOPs
model.info(detailed=True)    # 详细：每层 shape
print(model.model)           # PyTorch 原生打印
```

---

## 9. 机械臂抓取场景速查

### 典型流水线集成代码

```python
import cv2
import numpy as np
from ultralytics import YOLO

# ── 初始化（程序启动时执行一次）────────────────────────────
model = YOLO("yolo11n.engine")   # 生产环境用 TensorRT
# model = YOLO("yolo11n.pt")     # 开发调试用 PT

TARGET_CLASS = "bottle"          # 要抓取的目标类别
tracked = {}                     # {track_id: info}

# ── 主循环（每帧执行）───────────────────────────────────────
# pipeline = rs.pipeline()       # Realsense 初始化（需要 pyrealsense2）
# pipeline.start(config)

while True:
    # color_frame = pipeline.wait_for_frames().get_color_frame()
    # frame = np.asanyarray(color_frame.get_data())
    frame = cv2.imread("test.jpg")   # 替换为 Realsense 帧

    results = model.track(
        source=frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.20,        # 稍低，提高 Recall
        device="0",
        verbose=False,
    )

    r = results[0]
    targets = []

    if r.boxes.id is not None:
        for i in range(len(r.boxes)):
            name = r.names[int(r.boxes.cls[i].item())]
            if name != TARGET_CLASS:
                continue

            tid          = int(r.boxes.id[i].item())
            cx, cy, w, h = r.boxes.xywh[i].tolist()
            conf         = r.boxes.conf[i].item()

            targets.append({
                "id": tid,
                "center_px": (cx, cy),   # 像素坐标 → 配合深度图转 3D
                "size_px": (w, h),
                "conf": conf,
            })

    # → 将 targets 传给机械臂控制模块

# ── 推理速度参考（A100，yolo11n）──────────────────────────
# PT:        8.7ms  / 115 FPS
# ONNX:      6.7ms  / 148 FPS
# TensorRT:  4.2ms  / 236 FPS   ← 生产部署推荐
```

### 模型选型速查

| 需求 | 推荐模型 | 说明 |
|------|---------|------|
| 快速验证 | `yolo11n.pt` | 最轻量，2.6M 参数 |
| 精度优先 | `yolo11m.pt` / `yolo11l.pt` | 20~25M 参数 |
| 小目标（零件） | `yolo11n_p2.yaml` | 加 P2 检测头 |
| 实时部署 | `yolo11n.engine`（TensorRT FP16） | 4ms/帧 |
| 跨平台/CPU | `yolo11n.onnx` | 通用 |

### 常用 CLI 命令

```bash
# 推理
yolo predict model=yolo11n.pt source=image.jpg conf=0.25 device=0

# 训练
yolo train data=dataset.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0

# 验证
yolo val model=best.pt data=dataset.yaml device=0

# 导出
yolo export model=yolo11n.pt format=engine half=True device=0
yolo export model=yolo11n.pt format=onnx simplify=True
```

### 文件路径速查

```
/vepfs-mlp2/queue010/20252203765/yolo_learn/
  01_inference.py          # 实验1：推理
  02_tracking.py           # 实验2：跟踪
  03_train_finetune.py     # 实验3：训练/微调
  04_export.py             # 实验4：导出
  05_val.py                # 实验5：验证
  06_model_structure.py    # 实验6：模型结构
  yolo11n_p2.yaml          # P2 小目标检测头配置
  yolo11_custom_scale.yaml # 自定义 scale 配置
  my_dataset/              # 示例自定义数据集
  runs/finetune_demo/      # 训练输出（weights/best.pt）
  output/                  # 推理/跟踪可视化结果

~/.cache/ultralytics/      # 自动下载的预训练权重缓存
```

---

*生成时间：2026-03-24 | ultralytics 8.4.24 | YOLO11*
