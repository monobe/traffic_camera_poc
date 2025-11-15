import time
import cv2
import numpy as np
from ultralytics import YOLO

print("YOLOv8n 推論速度テスト")
print("=" * 50)

# ダミー画像生成
img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

# モデルロード
print("モデルロード中...")
model = YOLO("yolov8n.pt")

# ウォームアップ
print("ウォームアップ中...")
for _ in range(3):
    model(img, conf=0.25, imgsz=640, verbose=False)

# 推論速度測定
print("\n推論速度測定（10回）:")
times = []
for i in range(10):
    start = time.time()
    results = model(img, conf=0.25, imgsz=640, verbose=False)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  {i+1}: {elapsed:.3f}秒")

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time

print("\n結果:")
print(f"  平均推論時間: {avg_time:.3f}秒")
print(f"  実効FPS: {fps:.1f} FPS")
print(f"  カメラFPS 20との比較: {'✓ 余裕' if fps >= 10 else '✗ 遅い（フレームドロップ発生）'}")
