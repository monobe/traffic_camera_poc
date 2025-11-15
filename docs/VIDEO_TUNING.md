# 動画を使った検出パラメータチューニングガイド

## 概要

リアルタイムでの試行錯誤は非効率なため、一度撮影した動画を使ってパラメータを最適化します。

## 手順

### 1. テスト動画を撮影

スマホやカメラで道路を30秒〜1分撮影してください。

**撮影のポイント：**
- 実際のカメラ位置と同じアングルで撮影
- 複数の車両が通過する様子を含める
- 速度が速い車・遅い車両の両方を含める
- 歩行者や自転車も含めるとベター

### 2. 動画を配置

撮影した動画を `test_videos/` ディレクトリに保存：

```bash
# 例: スマホから動画を転送
cp ~/Downloads/traffic_video.mp4 test_videos/
```

### 3. 基本的な実行

```bash
# 動画を処理して統計を表示
python test_video_tuning.py test_videos/traffic_video.mp4 --stats

# リアルタイムで表示しながら実行
python test_video_tuning.py test_videos/traffic_video.mp4 --display

# 結果を動画として保存
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --output output_annotated.mp4
```

### 4. パラメータを変えてテスト

同じ動画で異なるパラメータを試して比較します：

#### 検出感度を上げる（より多く検出）

```bash
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --confidence 0.2 \
  --stats
```

#### 検出感度を下げる（誤検出を減らす）

```bash
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --confidence 0.4 \
  --stats
```

#### トラッキングの追従性を上げる（速い車両に対応）

```bash
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --iou-threshold 0.5 \
  --stats
```

#### 速度測定の最小フレーム数を変える

```bash
# より短いトラックでも速度測定（速い車両向け）
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --min-track-length 3 \
  --stats

# より長いトラックが必要（精度重視）
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --min-track-length 10 \
  --stats
```

#### 複数パラメータを同時に調整

```bash
python test_video_tuning.py test_videos/traffic_video.mp4 \
  --confidence 0.25 \
  --iou-threshold 0.5 \
  --min-track-length 3 \
  --display \
  --stats
```

### 5. 結果を比較

各パラメータで以下を確認：

**良い結果の指標：**
- ✅ 通過した車両のほとんどが検出されている
- ✅ 誤検出が少ない（存在しない車を検出していない）
- ✅ トラックIDが安定（同じ車が途中で別IDにならない）
- ✅ 速度測定の数が適切（検出した車両の多くで速度が計測できている）

**悪い結果の兆候：**
- ❌ 明らかに通過した車が検出されていない
- ❌ 誤検出が多い（影や背景を検出）
- ❌ トラックIDが頻繁に変わる（1台の車が複数のIDを持つ）
- ❌ 速度測定がほとんどできていない

### 6. 最適なパラメータを config.yaml に反映

最も良い結果が得られたパラメータを `config.yaml` に設定：

```yaml
detection:
  confidence: 0.25  # ここを調整

tracking:
  iou_threshold: 0.5  # ここを調整

speed_estimation:
  min_track_length: 3  # ここを調整
```

### 7. ダッシュボードで確認

```bash
# ダッシュボードを再起動して設定を反映
lsof -ti :8000 | xargs kill -9 2>/dev/null
sleep 2
source venv/bin/activate
python dashboard/app.py
```

## パラメータの意味

### confidence (検出信頼度閾値)
- **範囲**: 0.1 〜 0.9
- **低い値 (0.2)**: より多く検出するが、誤検出も増える
- **高い値 (0.5)**: 確実なものだけ検出するが、見逃しが増える
- **推奨**: 0.25 〜 0.3

### iou_threshold (IoU閾値 - トラッキング)
- **範囲**: 0.2 〜 0.7
- **低い値 (0.3)**: ゆっくり動く車向け、厳しい重なり判定
- **高い値 (0.5)**: 速い車向け、緩い重なり判定
- **推奨**: 0.4 〜 0.5

### min_track_length (最小トラック長)
- **範囲**: 3 〜 15 フレーム
- **短い (3フレーム = 0.15秒)**: 速い車でも測定可能、精度は低め
- **長い (10フレーム = 0.5秒)**: 精度高いが、速い車は測定できない
- **推奨**: 3 〜 5 (速度重視の場合)

## 例：速い車が多い道路の場合

```bash
python test_video_tuning.py test_videos/fast_cars.mp4 \
  --confidence 0.25 \
  --iou-threshold 0.5 \
  --min-track-length 3 \
  --display \
  --stats
```

## 例：歩行者・自転車が多い場合

```bash
python test_video_tuning.py test_videos/pedestrians.mp4 \
  --confidence 0.3 \
  --iou-threshold 0.4 \
  --min-track-length 7 \
  --display \
  --stats
```

## トラブルシューティング

### 車がほとんど検出されない
→ `--confidence 0.2` で感度を上げる

### 誤検出が多すぎる
→ `--confidence 0.4` で感度を下げる

### トラックIDが頻繁に変わる
→ `--iou-threshold 0.5` で追従性を上げる

### 速度測定がほとんどできない
→ `--min-track-length 3` で最小フレーム数を減らす

## ヒント

1. **複数の動画で試す**: 晴れ・曇り・夜間など、条件が異なる動画で確認
2. **比較表を作る**: 各パラメータでの検出数・速度測定数を記録
3. **出力動画を保存**: `--output` で結果を保存して後で見返す
4. **統計を確認**: `--stats` で詳細な統計を確認

## 次のステップ

最適なパラメータが見つかったら：

1. `config.yaml` を更新
2. ダッシュボードを再起動
3. 実際の運用で検証
4. 必要に応じて微調整
