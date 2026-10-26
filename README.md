# RGB-D-YOLO

面向无人扫地车的视觉感知实验工程：**RGB 单类别垃圾实例分割、人工复核预标注流水线、独立深度几何与双目实验工具**。

当前 RGB 模型为 YOLO11s-seg，类别只有 `0: sweepable_litter`。RGB 与深度目前独立处理，`spatial_fusion=false`；仓库名称不代表已经完成 RGBD 配准、融合训练或实时机器人闭环。

## 能力与边界

| 模块 | 已有能力 | 限制 |
|---|---|---|
| RGB | 单图实例分割、候选多边形、CVAT 复核包、教师微调 | 仍有鞋子、反光等误检及垃圾漏检；预测不是训练真值 |
| 深度 | 深度解码、反投影、地面拟合、突出物与掩膜几何 | 必须提供正确内参、深度单位和像素网格；不决定清扫动作 |
| 双目 | FoundationStereo 视差、条件性米制深度/XYZ/点云与诊断 | 依赖外部源码与匹配标定；历史脚本有实验路径依赖 |
| Jetson | Orin Nano 实机离线图片推理与自检 | 尚未完成实时相机、RGBD 配准、长期运行与稳定 FPS 验收 |

正类包括散落瓶罐、碎纸、揉皱纸板、薄膜、包装碎片和可清扫杂物。**完整规整纸箱、机器人、设备、人员、地坪接缝、污渍与反光属于背景。** 货物尺寸实验是独立的几何研究，不把完整纸箱改成 RGB 垃圾正类。

## 目录

```text
garbage_auto_annotation/     数据转换、来源、校验、预标注工具库
depth_perception/            独立深度几何模块
config/                     参数、配置模板与历史标定
schemas/                    深度结果 JSON Schema
tests/                      CPU 回归测试
tools/                      数据准备、下载、实验、报告、发布工具
deployment/jetson/          Docker 与板端自检入口
docs/                       流水线说明及历史实验说明
assets/manifest.json        下载分组、体积和 SHA256
run_perception.py           统一图片推理入口
train_garbage_teacher.py    RGB 教师训练入口
predict_garbage_teacher.py  RGB 预标注与 CVAT 包入口
```

下载后另外生成 `models/`、`datasets/`、`data/`；这些大型二进制数据不进入 Git 提交。实验输出默认放 `outputs/<独立运行名>/`。

## 安装

建议 Python 3.10 或 3.11。先为你的 CPU/CUDA 安装匹配的 PyTorch 与 torchvision，再安装项目依赖；不要在 Jetson 上用桌面版 wheel 替换 NVIDIA 版本。

```bash
git clone https://github.com/ZRJ-H/RGB-D-YOLO.git
cd RGB-D-YOLO
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch torchvision
python -m pip install -r requirements-dev.txt
```

上述 torch 安装命令不保证 GPU 可用，CUDA 环境需选择相应发行版本。核心 RGB 依赖固定为 Ultralytics 8.3.179。`requirements-stereo.txt` 仅列出双目附加库，不是完整 FoundationStereo 环境锁文件。

## 原始数据与权重

**发布状态（2026-10-26）：代码、原始数据和权重均已上线。data-v1 共 22 个数据分卷（10,628,861,101 字节）和 5 个清单文件；全部 27 个资产的服务器 SHA256 与本地一致，已验证清单可匿名下载。**

大型资产放在本仓库的 [data-v1 Release](https://github.com/ZRJ-H/RGB-D-YOLO/releases/tag/data-v1)。下载器按清单验证每个分卷、合并文件和恢复文件的 SHA256；相同内容在包内去重。分卷不是独立 ZIP，请使用下载器恢复。

| 分组 | 恢复内容 |
|---|---|
| `models` | 当前 `models/best.pt`、历史 RGB 实验最佳权重、FoundationStereo `23-51-11` 权重和配置 |
| `reviewed` | reviewed20_r10、round3_r3、overnight reviewed Copy-Paste 数据快照，含图片、标签、划分与来源记录 |
| `raw` | 两份相机 ROS bag、47组双目采集及可用 Color/Depth、相机参数、已有 RGB 抽帧 |
| `public` | 原始 TACO.zip、DSWD.zip 与来源记录，保留原压缩包内容 |

```bash
# 只获取权重；缓存应放在有足够空间的数据盘。
python tools/fetch_assets.py models --cache E:/gpt/_release/rgbd_download_cache
# 获取训练数据，或将 reviewed 替换为 raw / public / all。
python tools/fetch_assets.py reviewed --cache E:/gpt/_release/rgbd_download_cache
```

Linux 下将 `--cache` 换成自己的绝对路径。空间需同时容纳分卷、合并包和展开数据；精确大小见 [资产清单](assets/manifest.json)。恢复遇到内容不同的已有文件会拒绝覆盖。公开源压缩包恢复至 `datasets/source/` 后按原目录结构解压；不同版本之间不要混用图片与标签。

当前 RGB 基准权重 SHA256：

```text
06c69e0ed6ca59a5b19f8d5ee4b7de5a226bfdd2787e7eb0ee07da2a9464f910
```

FoundationStereo 原权重 SHA256：

```text
60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1
```

`models/experiments/` 中的候选权重不代表已晋升为基准模型。资产不包含 Docker 大镜像、Python 环境、全部重复推理输出或历史点云发布包。下载的 `.pt/.pth` 只应从可信来源加载。

## 快速运行

所有命令在仓库根目录执行，输出目录必须使用新名字。先做不需要模型和真实相机的合成验证：

```bash
python run_perception.py depth-synthetic --output outputs/synthetic_01
```

RGB 推理：

```bash
python run_perception.py rgb --rgb path/to/rgb.jpg --weights models/best.pt --device cpu --imgsz 960 --conf 0.25 --iou 0.35 --output outputs/rgb_01
```

用 NVIDIA GPU 时改成 `--device 0`。原始倒置相机图需要 `--rotate 180`，已经旋转的图片不要再次旋转。

未标定 RGB/深度并排预览：

```bash
python run_perception.py paired-preview --rgb path/to/rgb.jpg --depth path/to/depth_raw.png --weights models/best.pt --device cpu --output outputs/paired_01
```

深度输入应为单通道 uint16。该模式只展示独立原始深度，不产生 RGB 实例米制几何。

有真实深度标定时，复制并填写 `config/intrinsics.template.yaml`：

```bash
python run_perception.py depth-calibrated --depth path/to/depth.png --intrinsics path/to/intrinsics.yaml --depth-scale 0.001 --output outputs/depth_01
```

`0.001` 仅适用于已确认原始单位为毫米的设备。不能凭图像格式或灰度范围推断单位。内参必须对应输入深度网格，不能把 RGB 内参直接用于未配准深度。

输出包含 `result.json`、预览及对应图像/掩膜（依模式不同）。深度模块的光学坐标方向为 X 向右、Y 向下、Z 向前；米制几何与 RGB 语义置信度分开表达。详细字段见 [结果 Schema](schemas/depth_perception_result.schema.json) 和 [深度使用说明](docs/depth-baseline-usage.md)。

## 训练与人工复核

归档 YAML 保留原开发机路径。先生成本机配置，不修改数据快照：

```bash
python tools/configure_dataset.py --root datasets/garbage_seg_factory_reviewed20_r10 --output config/dataset.local.yaml
python validate_garbage_dataset.py --data config/dataset.local.yaml
python train_garbage_teacher.py --data config/dataset.local.yaml --model models/best.pt --project outputs/training --name reviewed_finetune_01 --optimizer AdamW --lr0 0.00005 --warmup-epochs 0 --warmup-bias-lr 0 --epochs 10 --imgsz 960 --batch 8 --device 0
```

通过 `--dry-run` 可先检查训练参数。数据约定是 YOLO segmentation：每行 `0 x1 y1 x2 y2 ...`，坐标归一化到 `[0,1]`；空标签文件表示人工确认的负样本。训练前校验图片、标签、类别、来源与划分，防止同源帧泄漏。仓库的早期校验器面向 `images/train`、`images/val` 目录型划分，不能用它验证训练列表型 Copy-Paste 配置。

Copy-Paste 快照可用 `--source dataset_baseline.yaml` 或 `--source dataset_synthetic.yaml` 生成本地配置；配置工具同时重定位图片列表。增强图必须保持合成来源标记，不应加入独立验收集。

生成预标注：

```bash
python predict_garbage_teacher.py --model models/best.pt --source path/to/rgb_frames --output outputs/preannotation_01 --conf 0.25 --iou 0.35 --imgsz 960 --device 0 --include-images
```

默认选20帧试标，确认后可用 `--all-after-pilot` 处理全部输入。流程是：**采集 → 抽帧/去重 → 模型建议 → CVAT 人工纠错 → 校验及来源记录 → 独立训练 → 固定验证与未见场景复核**。禁止直接把模型预测当作训练真值；历史矩形提示也不等同于精确实例掩膜。

公开数据转换、人工复核导入和困难负样本工具详见 [标注流水线](docs/garbage-auto-annotation.md)。其中历史路径和指标有日期语境，新的运行以显式 CLI 参数为准。

## 双目、点云与 Jetson

双目实验使用外部 FoundationStereo 标准网络，并未在本项目重新训练。`tools/foundationstereo_runtime.py` 提供既有 Windows SDPA 兼容加载路径；`tools/run_stereo_disparity_trial.py` 等脚本依赖实验索引和原目录。迁移前按 [复现范围说明](docs/REPRODUCIBILITY.md) 调整路径并安装匹配的外部源码，不能仅下载权重就运行完整双目流程。

深度公式一般为 `Z = fx * baseline / disparity`，其成立依赖校正后的投影参数与主点关系；不要对未校正原图直接套用某个固定焦距/基线。点云公式核验不代表实物尺寸准确。历史 Raw1 数据存在标定与图片坐标状态不匹配的证据，相关参数仅保留实验记录。

2026-10-08 曾在 Orin Nano、Ubuntu 22.04、L4T 36.4.7 上完成 Docker GPU 自检、30项发布测试及6组图片推理。镜像环境观察值为 CUDA 12.6、NVIDIA PyTorch 2.5.0a0、Ultralytics 8.3.179；这不是所有 JetPack 版本兼容性承诺。原启动器限定 L4T 36.4.4，历史 [Jetson README](deployment/jetson/README.md) 的“未验收”描述属于镜像构建时状态；不要原样套用旧启动器于 36.4.7。

重新打包的所有暂存、压缩包、校验侧文件应在 `E:/gpt/_release`，例如：

```bash
python tools/build_preview_release.py --destination E:/gpt/_release/rgbd_preview_new --weights models/best.pt --samples path/to/verified_samples
```

构建后先按发布工具执行验证，再封包；不要覆盖旧发布目录。

## 测试与历史指标

```bash
python -m pytest tests -q
```

测试覆盖数据转换/来源、几何、CLI 保护、发布归档等，不要求下载实际模型或数据。GPU、实际相机和精度验收需单独执行。

历史当前 RGB 基准在公共验证集上 box mAP50-95 约 0.338，mask mAP50-95 约 0.296。这些值来自历史记录，并非此次整理重新训练得到，也不能代表工厂现场精度。PC GPU 测速、单次冷启动时间均不能当作 Jetson 稳定帧率。

## 常见问题

- **下载失败/404**：确认 Release 中对应分卷已上传完成；可重跑下载器，已通过校验的分卷会复用。
- **数据路径仍指向 C 盘**：先运行 `tools/configure_dataset.py`，训练使用新生成的 YAML。
- **输出目录已存在**：改用新的运行编号，保留旧结果。
- **没有 CUDA**：先用 CPU 验证，再检查与硬件匹配的 torch；Jetson 使用 NVIDIA 环境。
- **深度全空/没有地面**：检查原始编码、内参、比例与地面 ROI，质量标志不能用伪造数据消除。
- **分割与深度错位**：并排预览没有建立空间配准，不能把两幅不同网格的像素直接对应。

## 来源与许可

项目使用 Ultralytics、PyTorch、OpenCV、FoundationStereo 等第三方组件；第三方源码、数据和权重继续遵循各自许可。本次汇总不替这些材料重新授权，也未指定原创代码的统一开源许可。

公开数据来源记录随 `public` 分组提供：TACO（DOI `10.5281/zenodo.3587843`）与 DSWD（DOI `10.17632/gr99ny6b8p.1`），本地来源记录均标注 CC BY 4.0。原始档案及其归属信息保留；转换、单类合并、人工纠错和 Copy-Paste 属于后续处理。现场数据与模型由仓库所有者明确要求公开发布；这不构成对第三方材料许可的扩大。
