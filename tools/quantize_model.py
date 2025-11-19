"""
YOLOv8モデルのINT8量子化スクリプト

INT8量子化により:
- モデルサイズが約4分の1に削減
- CPU推論速度が向上（特にIntel CPUで効果的）
- 精度の低下は通常1-2%程度

使用方法:
    python tools/quantize_model.py --model yolov8n.pt --data calibration_data/
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_calibration_frames(video_source: str, num_frames: int = 100, output_dir: str = "calibration_data"):
    """
    キャリブレーション用のフレームを収集

    Args:
        video_source: RTSP URLまたはビデオファイルパス
        num_frames: 収集するフレーム数
        output_dir: フレームを保存するディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    logger.info(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        logger.error("Failed to open video source")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        # ビデオファイルの場合、均等に分散してサンプリング
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # ストリームの場合、最初のN フレームを収集
        frame_indices = range(num_frames)

    collected = 0
    frame_count = 0

    logger.info(f"Collecting {num_frames} calibration frames...")

    while collected < num_frames:
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Failed to read frame {frame_count}")
            break

        # ビデオファイルの場合、指定されたインデックスのみ保存
        if total_frames > 0:
            if frame_count not in frame_indices:
                frame_count += 1
                continue

        # フレームを保存
        frame_path = output_path / f"frame_{collected:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        collected += 1

        if collected % 10 == 0:
            logger.info(f"Collected {collected}/{num_frames} frames")

        frame_count += 1

    cap.release()
    logger.info(f"✓ Collected {collected} calibration frames to {output_dir}")
    return True


def export_optimized_model(model_path: str, output_format: str = "coreml", int8: bool = True):
    """
    YOLOv8モデルを最適化してエクスポート

    Args:
        model_path: 元のYOLOモデルパス (.pt)
        output_format: 出力形式 ("coreml", "ncnn", "engine")
        int8: INT8量子化を使用するか
    """
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # 基本パラメータ
    export_args = {
        'format': output_format,
        'imgsz': 640,
    }

    # 形式ごとの設定
    if output_format == 'coreml':
        # CoreML (Apple Silicon最適化)
        if int8:
            export_args['int8'] = True
        export_args['nms'] = True  # NMSを含める
        logger.info(f"Exporting CoreML model (int8={int8})...")

    elif output_format == 'ncnn':
        # NCNN (CPU/モバイル最適化)
        export_args['half'] = False  # NCNNはFP16をサポート
        logger.info(f"Exporting NCNN model...")

    elif output_format == 'engine':
        # TensorRT (NVIDIA GPU必須)
        if int8:
            export_args['int8'] = True
        export_args['workspace'] = 4  # GB
        export_args['verbose'] = True
        logger.info(f"Exporting TensorRT model (int8={int8})...")

    else:
        logger.error(f"Unsupported format: {output_format}")
        return None

    try:
        export_path = model.export(**export_args)
        logger.info(f"✓ Model exported to: {export_path}")
        return export_path
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_calibration_yaml(calibration_dir: str, output_path: str = "calibration.yaml"):
    """
    キャリブレーション用のYAML設定ファイルを作成

    Args:
        calibration_dir: キャリブレーション画像のディレクトリ
        output_path: YAML出力パス
    """
    yaml_content = f"""# Calibration dataset for INT8 quantization
path: {Path(calibration_dir).absolute()}
train: .
val: .

# Classes (COCO dataset subset for traffic monitoring)
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  5: bus
  7: truck
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    logger.info(f"Created calibration YAML: {output_path}")
    return output_path


def compare_models(original_model: str, quantized_model: str, test_image: str):
    """
    元のモデルと量子化モデルのパフォーマンスを比較

    Args:
        original_model: 元のモデルパス
        quantized_model: 量子化モデルパス
        test_image: テスト画像パス
    """
    import time

    logger.info("Comparing original vs quantized model...")

    # 元のモデル
    logger.info("Testing original model...")
    model_orig = YOLO(original_model)

    # ウォームアップ
    img = cv2.imread(test_image)
    for _ in range(5):
        _ = model_orig(img, verbose=False)

    # ベンチマーク
    times_orig = []
    for _ in range(50):
        start = time.time()
        results = model_orig(img, verbose=False)
        times_orig.append(time.time() - start)

    avg_time_orig = np.mean(times_orig) * 1000  # ms
    fps_orig = 1000 / avg_time_orig

    logger.info(f"Original model: {avg_time_orig:.1f}ms ({fps_orig:.1f} FPS)")

    # 量子化モデル
    logger.info("Testing quantized model...")
    model_quant = YOLO(quantized_model)

    # ウォームアップ
    for _ in range(5):
        _ = model_quant(img, verbose=False)

    # ベンチマーク
    times_quant = []
    for _ in range(50):
        start = time.time()
        results = model_quant(img, verbose=False)
        times_quant.append(time.time() - start)

    avg_time_quant = np.mean(times_quant) * 1000  # ms
    fps_quant = 1000 / avg_time_quant

    logger.info(f"Quantized model: {avg_time_quant:.1f}ms ({fps_quant:.1f} FPS)")

    speedup = avg_time_orig / avg_time_quant
    logger.info(f"Speedup: {speedup:.2f}x")

    return {
        'original': {'time_ms': avg_time_orig, 'fps': fps_orig},
        'quantized': {'time_ms': avg_time_quant, 'fps': fps_quant},
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser(description='Quantize YOLOv8 model to INT8')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 model')
    parser.add_argument('--collect-calibration', action='store_true',
                        help='Collect calibration frames from video source')
    parser.add_argument('--video-source', type=str,
                        help='Video source for calibration (RTSP URL or file)')
    parser.add_argument('--calibration-frames', type=int, default=100,
                        help='Number of calibration frames to collect')
    parser.add_argument('--calibration-dir', type=str, default='calibration_data',
                        help='Directory for calibration frames')
    parser.add_argument('--format', type=str, default='coreml',
                        choices=['coreml', 'ncnn', 'engine'],
                        help='Export format (coreml/ncnn for CPU, engine for NVIDIA GPU)')
    parser.add_argument('--no-int8', action='store_true',
                        help='Disable INT8 quantization (use FP32/FP16)')
    parser.add_argument('--test-image', type=str,
                        help='Test image for comparing models')

    args = parser.parse_args()

    # キャリブレーションフレームの収集
    if args.collect_calibration:
        if not args.video_source:
            logger.error("--video-source is required when --collect-calibration is specified")
            return

        success = collect_calibration_frames(
            args.video_source,
            args.calibration_frames,
            args.calibration_dir
        )

        if not success:
            logger.error("Failed to collect calibration frames")
            return

        # 収集のみの場合はここで終了
        if not args.format:
            logger.info("Calibration frames collected. Run again with --format to export model.")
            return

    # 最適化モデルのエクスポート
    use_int8 = not args.no_int8
    quantized_model = export_optimized_model(
        args.model,
        args.format,
        int8=use_int8
    )

    if quantized_model is None:
        logger.error("Quantization failed")
        return

    # パフォーマンス比較（テスト画像が指定されている場合）
    if args.test_image:
        compare_models(args.model, quantized_model, args.test_image)

    logger.info("Done!")


if __name__ == '__main__':
    main()
