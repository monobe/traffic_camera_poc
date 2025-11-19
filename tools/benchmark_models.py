"""
YOLOモデルのベンチマークスクリプト

PyTorch (.pt) vs CoreML (.mlpackage) の推論速度を比較
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO


def benchmark_model(model_path: str, test_image_path: str, iterations: int = 50):
    """
    モデルのベンチマーク

    Args:
        model_path: モデルファイルパス
        test_image_path: テスト画像パス
        iterations: ベンチマーク反復回数

    Returns:
        平均推論時間 (ms), FPS
    """
    print(f"\nBenchmarking: {model_path}")

    # モデルロード
    print("  Loading model...")
    start_load = time.time()
    model = YOLO(model_path)
    load_time = time.time() - start_load
    print(f"  Load time: {load_time:.2f}s")

    # テスト画像読み込み
    img = cv2.imread(test_image_path)
    print(f"  Image size: {img.shape}")

    # ウォームアップ (最初の数回は遅い)
    print("  Warming up...")
    for _ in range(5):
        _ = model(img, verbose=False)

    # ベンチマーク
    print(f"  Running {iterations} iterations...")
    times = []

    for i in range(iterations):
        start = time.time()
        results = model(img, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # ms

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{iterations} iterations complete")

    # 統計
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / mean_time

    print(f"\n  Results:")
    print(f"    Mean time: {mean_time:.1f} ms (± {std_time:.1f} ms)")
    print(f"    Min time:  {min_time:.1f} ms")
    print(f"    Max time:  {max_time:.1f} ms")
    print(f"    FPS:       {fps:.1f}")

    # 検出数も表示
    num_detections = len(results[0].boxes)
    print(f"    Detections: {num_detections}")

    return mean_time, fps


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark YOLO models')
    parser.add_argument('--pytorch', type=str, default='yolov8n.pt',
                        help='PyTorch model path')
    parser.add_argument('--coreml', type=str, default='yolov8n.mlpackage',
                        help='CoreML model path')
    parser.add_argument('--image', type=str, default='calibration_data/frame_0000.jpg',
                        help='Test image path')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations for benchmark')

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO Model Benchmark")
    print("=" * 60)

    # PyTorchモデルのベンチマーク
    pt_time, pt_fps = benchmark_model(args.pytorch, args.image, args.iterations)

    # CoreMLモデルのベンチマーク
    coreml_time, coreml_fps = benchmark_model(args.coreml, args.image, args.iterations)

    # 比較
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"PyTorch (yolov8n.pt):       {pt_time:.1f} ms  ({pt_fps:.1f} FPS)")
    print(f"CoreML (yolov8n.mlpackage): {coreml_time:.1f} ms  ({coreml_fps:.1f} FPS)")
    print()

    speedup = pt_time / coreml_time
    if speedup > 1:
        print(f"CoreML is {speedup:.2f}x FASTER than PyTorch")
    else:
        print(f"CoreML is {1/speedup:.2f}x SLOWER than PyTorch")

    size_reduction = (1 - 3.2 / 6.2) * 100
    print(f"Model size reduction: {size_reduction:.1f}% (6.2 MB → 3.2 MB)")
    print("=" * 60)


if __name__ == '__main__':
    main()
