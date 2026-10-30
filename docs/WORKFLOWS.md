# 日常工作流

以下命令在仓库根目录执行。先完成 [项目首页的安装步骤](../README.md#安装)。`path/to/...` 和 `YOUR_VERIFIED_SCALE` 是需要替换的参数；每次运行使用新的输出名称。

## 下载需要的资产

图片推理只需先恢复权重；训练还需恢复审核数据。下载缓存放在空间充足的目录，下面的路径请替换为本机实际位置：

```bash
python tools/fetch_assets.py models --cache path/to/download_cache
python tools/fetch_assets.py reviewed --cache path/to/download_cache
```

默认恢复到仓库根目录，得到 `models/best.pt` 和清单中的数据目录。`--destination` 可指定另一根目录，此后运行命令使用对应的实际路径。完整分组与体积见 [资产清单](../assets/manifest.json)。下载器已有分卷与恢复检查，不需要为日常编辑额外重算全部文件哈希。

## 看一张 RGB 图

```bash
python run_perception.py rgb --rgb path/to/rgb.jpg --weights models/best.pt --device cpu --output outputs/rgb_01
```

支持 CUDA 的环境可改为 `--device 0`。默认 `imgsz=960`、`conf=0.25`、`iou=0.35`。

| 输出 | 内容 |
|---|---|
| `result.json` | 实例类别、框、置信度、掩膜文件名和推理参数 |
| `rgb_annotated.jpg` | 预测叠加图 |
| `mask_*.png` | 原 RGB 尺寸的 0/255 实例掩膜，无实例时不生成 |

要求权重是 `sweepable_litter` 单类分割模型。候选仅用于查看或人工复核，不直接作为训练真值。

## 并排查看 RGB 与原始深度

```bash
python run_perception.py paired-preview --rgb path/to/rgb.jpg --depth path/to/depth_raw.png --weights models/best.pt --device cpu --output outputs/paired_01
```

深度应为单通道 uint16。该模式额外生成 `preview.jpg`、`depth_colour.png` 和 `depth_nonzero.png`，只显示原始统计和预览，不产生配准后的实例几何。颜色按每帧分位数拉伸，不用于跨帧比较绝对距离。

原始图倒置时加 `--rotate 180`；已经旋转的图片不重复旋转。

## 运行独立深度

无需模型、相机或 GPU 的合成接口样例：

```bash
python run_perception.py depth-synthetic --output outputs/synthetic_01
```

结果包含 `result.json`、`synthetic_depth.png` 和 `object_mask.png`。这组输入是人工构造的平面与突出物，用于理解接口。

真实数据使用 [内参模板](../config/intrinsics.template.yaml)，填写对应深度网格的 `width`、`height`、`fx`、`fy`、`cx`、`cy`，并给出确认过的单位比例：

```bash
python run_perception.py depth-calibrated --depth path/to/depth.png --intrinsics path/to/intrinsics.yaml --depth-scale YOUR_VERIFIED_SCALE --output outputs/depth_01
```

原始数值已确认是毫米时比例为 `0.001`，数值本身已是米时为 `1`。不要从扩展名、灰度范围或图像尺寸猜单位。该模式也支持单通道 `.npy`。原入口 `run_depth_baseline.py` 从配置读取比例，默认值 `0.001` 同样需要与实际数据确认。接口细节见 [深度使用说明](depth-baseline-usage.md)。

## 恢复审核数据并训练候选

下载的 YAML 保留历史配置，先生成本机版本：

```bash
python tools/configure_dataset.py --root datasets/garbage_seg_factory_reviewed20_r10 --output config/dataset.local.yaml
python validate_garbage_dataset.py --data config/dataset.local.yaml
```

生成的配置使用本机路径，原快照保持不变。训练入口可先用 `--dry-run` 查看参数，实际训练示例：

```bash
python train_garbage_teacher.py --data config/dataset.local.yaml --model models/best.pt --project outputs/training --name reviewed_finetune_01 --optimizer AdamW --lr0 0.00005 --warmup-epochs 0 --warmup-bias-lr 0 --epochs 10 --imgsz 960 --batch 8 --device 0
```

这是候选微调配置示例，不是改善精度的承诺。保留已有权重和固定评估划分，根据公共验证与未见场景结果判断是否采用候选。

Copy-Paste 快照使用 `configure_dataset.py --source dataset_baseline.yaml` 或 `--source dataset_synthetic.yaml`，工具会同时重定位训练图片列表。早期 `validate_garbage_dataset.py` 面向目录型划分，不能直接验证这种列表型配置；不要将合成图加入独立验收集。

## 预标注与人工审核

```bash
python predict_garbage_teacher.py --model models/best.pt --source path/to/rgb_frames --output outputs/preannotation_01 --conf 0.25 --iou 0.35 --imgsz 960 --device 0 --include-images
```

默认选择 20 帧试标，适合先检查表现；确定处理范围后可用 `--all-after-pilot` 处理全部输入。数据主线为：采集与抽帧 → 模型建议 → CVAT 人工纠错 → 审核数据集 → 独立训练 → 同口径评估。

审核标签通过根目录 [add_reviewed_factory_dataset.py](../add_reviewed_factory_dataset.py) 导入，数据转换与来源记录等详细命令见 [标注流水线](garbage-auto-annotation.md)。历史矩形提示不等同精确实例掩膜。

## 双目与点云研究

| 输入或目标 | 入口 |
|---|---|
| 历史 Orbbec 左右 IR | [run_stereo_disparity_trial.py](../tools/run_stereo_disparity_trial.py)，原生输入显式指定 `--scale 1`，默认是 `0.5` |
| Raw1 拼接图 ZIP | [run_stereo_zip_batch.py](../tools/run_stereo_zip_batch.py) 的 `prepare` / `infer` 模式 |
| 指定标定重跑 Raw1 单组 | [run_corrected_stereo_sample.py](../tools/run_corrected_stereo_sample.py) |
| 已有原生视差导出深度/XYZ/点云 | [export_native_stereo_batch.py](../tools/export_native_stereo_batch.py) |

这些入口保留具体批次的输入索引和机器路径，先读 [可复现范围](REPRODUCIBILITY.md)，准备匹配的外部 FoundationStereo 源码与环境，再适配输入。标定需匹配图像模式、校正/裁剪状态和单位。完整算法流程见 [源码说明](foundationstereo-process-notes-20260928.md)。

## 保存结果与维护

普通运行使用 `outputs/<独立名称>/`，在结果说明中记录目的、命令、输入与输出位置。发布包和暂存文件使用明确的发布目录；原 Windows 开发机统一使用 `E:/gpt/_release`，包括打包工具的输出参数。历史包保留原位。

维护检查按改动选择：文档检查相关链接与命令；参数或算法修改检查受影响入口与行为。不默认执行全量回归、冒烟、训练或大文件哈希。已有下载和发布校验仍服务于对应工作流，无需为文档改动重复执行。
