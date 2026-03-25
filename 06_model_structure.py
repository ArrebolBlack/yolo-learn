"""
实验6：修改模型结构
学习目标：
  1. 读懂 YOLO YAML 模型定义语法
  2. 改动1：修改 nc（类别数）——最简单的改法
  3. 改动2：修改网络深度/宽度（scaling）
  4. 改动3：修改 YAML 增加/替换模块（增加一个检测头 P2，适合小目标）
  5. 改动4：注册自定义 Python 模块并在 YAML 中使用
"""

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C3k2
from ultralytics.nn.tasks import DetectionModel
import torch
import torch.nn as nn
from pathlib import Path

YAML_DIR = Path("/vepfs-mlp2/queue010/20252203765/yolo_learn")
YAML_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 改动1：只改 nc（最常见操作——迁移到自己的类别）
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("改动1：修改 nc（类别数）")
print("=" * 60)

# 方法A：训练时直接在 dataset.yaml 里指定 nc，YOLO 自动适配
# 方法B：从 YAML 构建时覆盖
model_nc2 = YOLO("yolo11n.yaml")           # 从 YAML 构建（随机初始化）
model_nc2.model.nc = 2                     # 直接改属性（不推荐，head 不会自动重建）

# 正确方法：在 YAML 里写好 nc，或训练时传 data=... 自动处理
# 实际使用时，只需：
#   model = YOLO("yolo11n.pt")
#   model.train(data="my_dataset.yaml")   ← nc 从 dataset.yaml 的 nc 字段自动设置
print("✅ nc 会在 model.train(data=...) 时自动从 dataset.yaml 读取，无需手动修改")

# ═══════════════════════════════════════════════════════════
# 改动2：通过 YAML scales 调整深度/宽度
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("改动2：自定义 scale（深度/宽度）")
print("=" * 60)

# 原始 yolo11n 使用 scale 'n': [depth=0.50, width=0.25, max_channels=1024]
# 我们写一个更小的 'tiny' scale
custom_scale_yaml = """\
# 自定义 scale 示例：比 nano 更小的 tiny 模型
nc: 2
scales:
  tiny: [0.33, 0.25, 512]   # depth=0.33, width=0.25, max_ch=512（比 nano 更浅）
  nano: [0.50, 0.25, 1024]  # 与官方 yolo11n 相同

backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]          # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]          # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]         # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]            # 9
  - [-1, 2, C2PSA, [1024]]              # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]         # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]         # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]         # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]         # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]]
"""

yaml_path = YAML_DIR / "yolo11_custom_scale.yaml"
yaml_path.write_text(custom_scale_yaml)

# 加载 tiny scale
model_tiny = YOLO(str(yaml_path))
info = model_tiny.info(verbose=False)
print(f"tiny 模型参数量: {sum(p.numel() for p in model_tiny.model.parameters()):,}")

# 加载 nano scale（对比）
model_nano = YOLO("yolo11n.yaml")
print(f"nano 模型参数量: {sum(p.numel() for p in model_nano.model.parameters()):,}")

# ═══════════════════════════════════════════════════════════
# 改动3：增加 P2 检测头（适合小目标检测）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("改动3：增加 P2 检测头（4 个检测尺度，适合小目标）")
print("=" * 60)

# 原始 yolo11n 有 3 个检测头：P3(8x), P4(16x), P5(32x)
# 增加 P2(4x) 后可以检测更小的目标（如流水线上的小零件）
p2_head_yaml = """\
# yolo11n + P2 小目标检测头
# 适用场景：检测距离较远或尺寸较小的目标（如流水线小零件）
nc: 80
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1-P2/4  ← 新增引用点
  - [-1, 2, C3k2, [256, False, 0.25]]   # 2
  - [-1, 1, Conv, [256, 3, 2]]          # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]   # 4
  - [-1, 1, Conv, [512, 3, 2]]          # 5-P4/16
  - [-1, 2, C3k2, [512, True]]          # 6
  - [-1, 1, Conv, [1024, 3, 2]]         # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]         # 8
  - [-1, 1, SPPF, [1024, 5]]            # 9
  - [-1, 2, C2PSA, [1024]]              # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 6], 1, Concat, [1]]                    # 12 cat P4
  - [-1, 2, C3k2, [512, False]]                  # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 4], 1, Concat, [1]]                    # 15 cat P3
  - [-1, 2, C3k2, [256, False]]                  # 16 (P3/8)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 17 ← 新增
  - [[-1, 2], 1, Concat, [1]]                    # 18 cat P2 ← 新增
  - [-1, 2, C3k2, [128, False]]                  # 19 (P2/4) ← 新增检测头

  - [-1, 1, Conv, [128, 3, 2]]                   # 20
  - [[-1, 16], 1, Concat, [1]]                   # 21 cat P3
  - [-1, 2, C3k2, [256, False]]                  # 22 (P3/8)

  - [-1, 1, Conv, [256, 3, 2]]                   # 23
  - [[-1, 13], 1, Concat, [1]]                   # 24 cat P4
  - [-1, 2, C3k2, [512, False]]                  # 25 (P4/16)

  - [-1, 1, Conv, [512, 3, 2]]                   # 26
  - [[-1, 10], 1, Concat, [1]]                   # 27 cat P5
  - [-1, 2, C3k2, [1024, True]]                  # 28 (P5/32)

  - [[19, 22, 25, 28], 1, Detect, [nc]]          # Detect(P2, P3, P4, P5)
"""

yaml_p2 = YAML_DIR / "yolo11n_p2.yaml"
yaml_p2.write_text(p2_head_yaml)

model_p2 = YOLO(str(yaml_p2))
print(f"P2 模型参数量: {sum(p.numel() for p in model_p2.model.parameters()):,}")
print(f"标准模型参数量: {sum(p.numel() for p in model_nano.model.parameters()):,}")
print("P2 头增加了少量参数，但能检测更小目标")

# ═══════════════════════════════════════════════════════════
# 改动4：注册自定义 Python 模块
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("改动4：注册自定义 Python 模块（以简单的 DepthwiseSepConv 为例）")
print("=" * 60)

# 步骤：
# 1. 定义 nn.Module 子类
# 2. 注册到 ultralytics.nn.modules.__init__ 的导出列表
# 3. 在 YAML 中引用

# 自定义模块：深度可分离卷积（Depthwise Separable Conv）
# 参数量约为普通卷积的 1/9，适合轻量化模型
class DSConv(nn.Module):
    """Depthwise Separable Convolution：深度卷积 + 逐点卷积"""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)  # depthwise
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)                       # pointwise
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

# 注册到 ultralytics 的模块查找表
import ultralytics.nn.modules as ult_modules
ult_modules.DSConv = DSConv

# 同时需要加入 tasks.py 的解析白名单（开发模式下直接修改）
from ultralytics.nn import tasks as ult_tasks
ult_tasks.DSConv = DSConv

# 验证自定义模块可以正常使用
test_input = torch.randn(1, 64, 32, 32)
ds = DSConv(64, 128, k=3, s=2)
out = ds(test_input)
print(f"DSConv 测试: input {test_input.shape} → output {out.shape} ✅")

# 计算参数节省
std_conv_params = 64 * 128 * 3 * 3   # 标准卷积
ds_conv_params  = 64 * 3 * 3 + 64 * 128  # 深度可分离卷积
print(f"参数对比: 标准卷积={std_conv_params:,}  DSConv={ds_conv_params:,}  "
      f"节省 {(1 - ds_conv_params/std_conv_params)*100:.0f}%")

print("""
【在 YAML 中使用自定义模块】
  backbone:
    - [-1, 1, DSConv, [128, 3, 2]]   # 用 DSConv 替换 Conv
  注意：YAML 中的模块名必须与注册到 ultralytics.nn.modules 的类名完全一致

【正式注册方法（持久化）】
  1. 在 ultralytics/nn/modules/block.py 末尾添加 DSConv 类定义
  2. 在 ultralytics/nn/modules/__init__.py 的 __all__ 列表中添加 "DSConv"
  3. 这样每次 import ultralytics 时自动可用，无需运行时注册
""")

# ═══════════════════════════════════════════════════════════
# 总结：查看模型结构
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("模型结构信息查看方式")
print("=" * 60)

model = YOLO("yolo11n.pt")
# model.info() 打印层信息
model.info(detailed=False, verbose=True)
