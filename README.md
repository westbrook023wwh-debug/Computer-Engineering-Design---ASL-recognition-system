# SignVision（本地 ASL 手势识别系统复现）

本项目复现：**MediaPipe 提取关键点 → 1D CNN + ECA + Transformer → 本地摄像头实时识别（Tkinter 窗口）**。

## 0) 环境安装

```powershell
cd "D:\Computer Science\ASL Recognition System"
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 1) 数据集（Kaggle asl-signs）

训练全量数据需要这些文件都在同一个目录下（例如 `data/asl-signs/`）：
- `train.csv`
- `sign_to_prediction_index_map.json`
- `train_landmark_files/`（大量 `*.parquet`；`train.csv` 的 `path` 列会引用它）

只放了 `train.csv` + `sign_to_prediction_index_map.json` 是**不能训练全量**的（会找不到 parquet）。

### 1.1 你当前的放置方式

如果你把 `sign_to_prediction_index_map.json` 下载到了 `data/` 根目录，需要复制到训练目录：

```powershell
New-Item -ItemType Directory -Force ".\data\asl-signs" | Out-Null
Copy-Item ".\data\sign_to_prediction_index_map.json" ".\data\asl-signs\sign_to_prediction_index_map.json" -Force
```

## 2) 训练

### 2.1 子集训练（推荐先跑通）

当 `kaggle competitions download -c asl-signs` 总是卡住时，用子集脚本下载一小部分 parquet：

```powershell
python -m signvision.kaggle_subset --out data\asl-signs-subset --max-samples 2000
python -m signvision.train --data data\asl-signs-subset --out checkpoints --feature-set hands --epochs 1 --batch-size 16
```

如果你想继续“多下载一点再训练”，重复运行 `kaggle_subset` 并把 `--max-samples` 调大即可（会自动保留已下载的 parquet，并生成只包含本地文件的 `train.csv`）：

```powershell
python -m signvision.kaggle_subset --out data\asl-signs-subset --max-samples 500 --cooldown-s 1.0
python -m signvision.kaggle_subset --out data\asl-signs-subset --max-samples 2000 --cooldown-s 1.0
```

### 2.2 全量训练（需要 train_landmark_files）

确认以下为 True：

```powershell
Test-Path ".\data\asl-signs\train.csv"
Test-Path ".\data\asl-signs\sign_to_prediction_index_map.json"
Test-Path ".\data\asl-signs\train_landmark_files"
```

然后训练：

```powershell
python -m signvision.train --data data\asl-signs --out checkpoints --feature-set hands --epochs 10
```

## 3) 启动（Tkinter 实时 UI）

```powershell
python -m signvision.realtime_tk --checkpoint checkpoints\best.pt --label-map checkpoints\label_map.json --feature-set hands
```

### 摄像头打不开怎么办

先扫描你机器上哪个索引/后端能打开：

```powershell
python -m signvision.camera_check
```

然后按输出提示启动，例如：

```powershell
python -m signvision.realtime_tk --checkpoint checkpoints\best.pt --label-map checkpoints\label_map.json --camera 1 --backend dshow
```

### MediaPipe 版本说明

如果你安装的 `mediapipe` 没有 `mp.solutions`（只包含 `mp.tasks`），程序会自动切到 **MediaPipe Tasks**（Hand/Face/Pose Landmarker）并在首次运行时把 `.task` 模型下载到 `assets/mediapipe/`（可用 `--mp-model-dir` 改目录，`--no-face/--no-pose` 关闭模块）。
