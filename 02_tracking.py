"""
实验2：YOLO 目标跟踪（Tracking）
学习目标：
  1. track() vs predict() 的区别
  2. 获取持久 track_id
  3. 理解两种跟踪器：ByteTrack vs BoT-SORT
  4. 模拟视频帧循环（机械臂实时抓取的核心模式）
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

OUT_DIR = Path("/vepfs-mlp2/queue010/20252203765/yolo_learn/output")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. 加载模型 ──────────────────────────────────────────────
model = YOLO("yolo11n.pt")

# ── 2. 用两张内置图片模拟"视频帧"，演示跨帧 ID 持久性 ────────
ASSETS = Path("/vepfs-mlp2/queue010/20252203765/ultralytics/ultralytics/assets")
frames = [
    cv2.imread(str(ASSETS / "bus.jpg")),
    cv2.imread(str(ASSETS / "bus.jpg")),   # 同一张图重复，模拟连续帧
    cv2.imread(str(ASSETS / "zidane.jpg")),
]

print("=" * 60)
print("【跟踪器对比】ByteTrack vs BoT-SORT")
print("=" * 60)

for tracker_name in ["bytetrack.yaml", "botsort.yaml"]:
    print(f"\n--- 使用跟踪器: {tracker_name} ---")

    for frame_idx, frame in enumerate(frames):
        results = model.track(
            source=frame,
            tracker=tracker_name,
            persist=True,    # ★ 关键：跨帧保持 track 状态，必须为 True
            conf=0.25,
            iou=0.7,
            device=0,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes

        print(f"\n  Frame {frame_idx}: 检测到 {len(boxes)} 个目标")

        if boxes.id is not None:
            # boxes.id 是 track_id，shape: (N,)
            for i in range(len(boxes)):
                track_id  = int(boxes.id[i].item())
                class_id  = int(boxes.cls[i].item())
                class_name = r.names[class_id]
                confidence = boxes.conf[i].item()
                cx, cy, w, h = boxes.xywh[i].tolist()

                print(f"    ID={track_id:3d}  {class_name:10s}  "
                      f"conf={confidence:.2f}  center=({cx:.0f},{cy:.0f})")
        else:
            print("    （本帧无有效 track，可能是首帧初始化中）")

    # 每个跟踪器测完后重置（避免 ID 状态污染下一个跟踪器的测试）
    model.predictor = None

# ── 3. 演示：机械臂抓取场景的帧处理循环 ─────────────────────
print("\n" + "=" * 60)
print("【机械臂场景】实时帧处理循环示例")
print("=" * 60)

model2 = YOLO("yolo11n.pt")

# 模拟从 Realsense 相机读取帧的循环
# 实际使用时替换为：
#   pipeline = rs.pipeline()  # pyrealsense2
#   frames = pipeline.wait_for_frames()
#   color_frame = frames.get_color_frame()
#   frame = np.asanyarray(color_frame.get_data())

target_class = "person"   # 你要抓取的目标类别，换成你的实际类别
tracked_objects = {}      # {track_id: {"class": ..., "center": ..., "history": [...]}}

for frame_idx, frame in enumerate(frames):
    results = model2.track(
        source=frame,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.3,
        device=0,
        verbose=False,
    )

    r = results[0]

    if r.boxes.id is not None:
        for i in range(len(r.boxes)):
            track_id   = int(r.boxes.id[i].item())
            class_name = r.names[int(r.boxes.cls[i].item())]
            cx, cy, w, h = r.boxes.xywh[i].tolist()
            conf = r.boxes.conf[i].item()

            # 更新跟踪字典
            if track_id not in tracked_objects:
                tracked_objects[track_id] = {
                    "class": class_name,
                    "history": [],
                }
            tracked_objects[track_id]["center"] = (cx, cy)
            tracked_objects[track_id]["size"]   = (w, h)
            tracked_objects[track_id]["conf"]   = conf
            tracked_objects[track_id]["history"].append((cx, cy))

    # 保存带 ID 标注的帧
    annotated = r.plot()
    cv2.imwrite(str(OUT_DIR / f"track_frame_{frame_idx}.jpg"), annotated)

print(f"\n共跟踪到 {len(tracked_objects)} 个唯一目标：")
for tid, info in tracked_objects.items():
    print(f"  ID={tid}  class={info['class']}  "
          f"最终位置=({info['center'][0]:.0f},{info['center'][1]:.0f})  "
          f"出现帧数={len(info['history'])}")

print(f"\n标注帧已保存到: {OUT_DIR}/track_frame_*.jpg")

# ── 4. 关键参数说明 ──────────────────────────────────────────
print("""
【跟踪器选择建议 - 机械臂场景】
  ByteTrack : 轻量快速，纯运动关联（IoU），适合目标不重叠、相机固定的场景
  BoT-SORT  : 加入全局运动补偿（GMC），适合相机移动或目标快速运动的场景
              with_reid=True 可开启外观特征，遮挡恢复更好，但需要额外计算

【关键参数调优 - 高速移动物体】
  track_buffer=30   : 目标消失后保留 ID 的帧数，高速物体可适当调大
  track_high_thresh : 降低可以检测更多目标，但会引入噪声
  match_thresh=0.8  : IoU 匹配阈值，高速物体位移大时可适当降低
""")
