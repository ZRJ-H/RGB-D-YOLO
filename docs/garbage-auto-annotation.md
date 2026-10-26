# 垃圾自动预标注流水线

## 目标和边界

这套流水线先用公开垃圾分割数据训练一个较大的离线教师 `YOLO11s-seg`，再让教师给工厂视频帧生成候选多边形。候选标注必须在 CVAT 中人工复核，不能直接当真值。最终部署到无人扫地车的模型可以另训为 `YOLO11n/YOLO11s`，并导出 ONNX 或 TensorRT 给 C++ 使用。

统一类别只有一个：

```text
0 sweepable_litter
```

它表示无人扫地车应处理的散落瓶罐、碎纸、包装碎片和杂物；完整、规整、仍在正常使用位置的纸箱、垃圾桶、托盘和设备属于背景。

## 当前目录

```text
目标工程：C:\Users\leg\Documents\Codex\2026-09-03\yolo-e-26-9-3-yolo
Python：D:\softwares\Anaconda3\envs\yolov8\python.exe
公开源数据：<目标工程>\datasets\source
阶段性 DWSD 数据：<目标工程>\datasets\garbage_seg_dwsd_interim
最终 TACO+DWSD 数据：<目标工程>\datasets\garbage_seg
训练结果：<目标工程>\outputs\training
推理结果：<目标工程>\outputs\inference
```

## 2026-09-07 当前状态

公开 TACO+DWSD 数据加入 17 张人工确认的工厂负样本后，基础数据集为 2301 张图、8940 个实例。现有 CVAT 检测框导出 `C:\cvat-share\garbage` 含 32 张图、24 张正样本、8 张负样本和 261 个框。程序已把检测框转换成四角矩形分割提示，并在不修改基础集的前提下生成：

```text
<目标工程>\datasets\garbage_seg_factory_seed32
总计：2333 张图，9201 个实例
训练：1883 张图，7506 个实例
验证：450 张图，1695 个实例
```

数据校验结果为 `valid: true`、`errors: []`。对应训练结果：

```text
<目标工程>\outputs\training\garbage_teacher_yolo11s_seg_factory_seed32_lr1e4_e12
最佳权重：weights\best.pt
box mAP50-95：约 0.327
mask mAP50：约 0.480
mask mAP50-95：约 0.287
```

固定工厂帧试标仍未过自动标注门槛：模型会漏掉明显纸堆，并误检机器人轮子、蓝色箱体、设备边缘、地面接缝和反光。把置信度从 0.10 提高到 0.25 会同时删除许多真垃圾，不能靠阈值解决。纯工厂聚焦微调还会快速遗忘公开垃圾形态，因此当前不应对全部视频直接批量生成“真值”。

下一轮独立纠错批次已经生成：

```text
C:\Users\leg\Documents\ChatGPT\yolo\active_batches\factory_holdout20_20260907
```

其中 `cvat_yolo_seg_proposals.zip` 已包含 20 张原图和 66 个候选多边形，可直接用于 CVAT 复核。这 20 张已排除先前参与训练的 17 张硬负样本。

## 2026-09-07 第二轮人工复核结果

CVAT 导出的第二轮文件包含 20 张图、13 张正样本、7 张空负样本和 80 个合法分割多边形。人工多边形没有再次转换成矩形。为避免它们在 1900 多张训练图里只占约 1%，导入时仅把这 20 张训练图重复 10 次，公开验证集保持不变：

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\add_reviewed_factory_dataset.py `
  --base-dataset .\datasets\garbage_seg_factory_seed32 `
  --archive 'C:\cvat-share\garbage-round2\task_2_dataset_2026_09_07_05_50_14_ultralytics yolo segmentation 1.0.zip' `
  --output .\datasets\garbage_seg_factory_reviewed20_r10 `
  --repeat 10
```

生成数据集共 2533 张图、10001 个实例；训练集为 2083 张图、8306 个实例，验证集仍为 450 张图、1695 个实例。校验结果为 `valid: true`、`errors: []`。

从上一版最佳权重继续训练 10 轮：

```text
<目标工程>\outputs\training\garbage_teacher_yolo11s_seg_reviewed20_r10_lr5e5_e10
optimizer=AdamW
lr0=0.00005
warmup_epochs=0
warmup_bias_lr=0
```

最终 `best.pt` 的公开验证结果约为 box mAP50-95 0.338、mask mAP50-95 0.295；上一版约为 0.327 和 0.288。另一组未参与训练的 20 张工厂帧上，旧模型在 `conf=0.25` 时只有 9 个候选，新模型在相同阈值下有 49 个候选；使用 `IoU=0.35` 去重后为 45 个。目视对比显示新模型补回了多处旧模型漏掉的纸片和纸堆，并消除了第 32 帧完整纸箱的误报，但仍需要人工删除脚、亮斑或相邻重复候选。

全 91 帧推理结果：

```text
<目标工程>\outputs\inference\reviewed20_r10_all91_conf025_iou035
91 张，221 个候选
```

排除已经人工复核的 20 张和先前确认的 17 张硬负样本后，下一轮实际需要复核的包为：

```text
<目标工程>\outputs\inference\reviewed20_r10_remaining54_conf025_iou035\cvat_yolo_seg_proposals.zip
54 张，144 个候选
```

这个模型已经从“候选很少且漏检严重”进步到“可减少从零描边”，但尚不能无人审核生成真值。

## 标准操作顺序

### 1. 转换公开数据

DWSD 阶段性转换：

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\prepare_dwsd_dataset.py `
  --source '<目标工程>\datasets\source\dwsd\extracted' `
  --output '<目标工程>\datasets\garbage_seg_dwsd_interim'
```

TACO 与 DWSD 都下载并解压后，生成最终合并数据：

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\prepare_garbage_dataset.py `
  --taco-source '<目标工程>\datasets\source\taco\extracted' `
  --dwsd-source '<目标工程>\datasets\source\dwsd\extracted' `
  --output '<目标工程>\datasets\garbage_seg'
```

### 2. 校验数据

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\validate_garbage_dataset.py `
  --data '<目标工程>\datasets\garbage_seg\dataset.yaml'
```

只有输出 `"valid": true` 且 `"errors": []` 才进入训练。

### 3. 加入人工确认的工厂负样本

先生成接触表并逐图确认，再把文件名写入 JSON。程序不会自动把任何图像判定为纯负样本。

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\select_hard_negatives.py `
  --source '<目标工程>\work\extracted_frames\test_camera1_1fps' `
  --approved-json '.\config\approved_hard_negatives.json' `
  --dataset-root '<目标工程>\datasets\garbage_seg'
```

### 4. 训练离线教师

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\train_garbage_teacher.py `
  --data '<目标工程>\datasets\garbage_seg\dataset.yaml' `
  --model yolo11s-seg.pt `
  --imgsz 960 `
  --batch 8 `
  --epochs 100 `
  --patience 20 `
  --device 0 `
  --workers 0 `
  --name garbage_teacher_yolo11s_seg
```

从现有最佳权重低学习率精调时，应显式指定优化器和学习率：

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\train_garbage_teacher.py `
  --data '<目标工程>\datasets\garbage_seg_factory_seed32\dataset.yaml' `
  --model '<已有训练结果>\weights\best.pt' `
  --optimizer AdamW `
  --lr0 0.0001 `
  --epochs 12 `
  --imgsz 960 `
  --batch 8 `
  --device 0
```

小数据聚焦训练必须额外关闭默认 warmup：

```text
--warmup-epochs 0 --warmup-bias-lr 0
```

原因是默认 `warmup_bias_lr=0.1` 对每轮只有几个 batch 的小数据集过大，实测会在数十步内让公开验证 mAP 接近零。

训练由 Ultralytics/PyTorch 执行。训练集用于反向传播更新参数，验证集只用于计算 precision、recall、mAP 和选择 `best.pt`。公开验证集指标只能说明模型会不会公开数据，不能证明工厂现场效果。

### 5. 固定 20 帧试标

```powershell
& 'D:\softwares\Anaconda3\envs\yolov8\python.exe' .\predict_garbage_teacher.py `
  --model '<目标工程>\outputs\training\garbage_teacher_yolo11s_seg\weights\best.pt' `
  --source '<目标工程>\work\extracted_frames\test_camera1_1fps' `
  --output '<目标工程>\outputs\inference\garbage_teacher_factory_pilot20' `
  --conf 0.10 `
  --iou 0.50 `
  --imgsz 960 `
  --device 0
```

输出包括：

- `annotated/`：叠加多边形的图片；
- `contact_sheet.jpg`：固定 20 帧总览；
- `predictions.json`：类别、置信度、多边形和拒绝原因；
- `labels_with_confidence/`：审计用，不可直接作为训练标签；
- `labels/`：不含置信度的标准 YOLO 分割标签；
- `cvat_yolo_seg_proposals.zip`：CVAT 导入包，默认不重复包含图片。

如果试标仍大量识别地坪磨损、反光、纸箱或设备，应停止全量导出，加入这些误检场景的纯负样本或人工纠正样本后再训练。

### 6. CVAT 复核

1. 在 CVAT 中创建与原始 91 帧对应的任务，类别名设为 `sweepable_litter`。
2. 导入 `cvat_yolo_seg_proposals.zip`；如果任务中没有原图，重新推理时加 `--include-images` 生成含图压缩包。
3. 删除地坪磨损、反光、完整纸箱、设备等误检。
4. 补画遗漏的瓶罐、碎纸和散落杂物。
5. 边界贴合实际可清扫物，不把大片地面或阴影包含进去。
6. 导出为 Ultralytics YOLO Segmentation，并把复核数据作为“工厂域种子集”参与下一轮训练。

当前应优先复核独立批次 `active_batches\factory_holdout20_20260907\cvat_yolo_seg_proposals.zip`。具体规则：

1. 完整、规整、仍在正常位置的纸箱不标；第 31 帧专门用于校正这一边界。
2. 删除机器人、轮子、垃圾桶、托盘车、设备、地坪接缝、污渍和反光上的候选。
3. 保留并修正散落碎纸、揉皱纸板、薄膜、包装碎片和小杂物。
4. 对第 9、58、78、80、90 等明显漏检帧补画多边形。
5. 没有可清扫垃圾的帧必须保留为空标签，作为困难负样本。

这 20 张的目的不是一次把模型训练好，而是提供目标相机下更有信息量的错误样本。复核后再与当前数据合并训练，才有依据判断自动预标注是否真正节省人工。

## Python 跑通对未来 C++ 的意义

Python 阶段验证的是数据定义、训练方式、精度上限、阈值、输入尺寸和后处理逻辑。C++ 阶段通常不重新发明模型，而是加载 Python 训练后导出的 ONNX/TensorRT 权重，实现一致的 resize/letterbox、归一化、NMS、分割掩码恢复和坐标映射。

因此应把链路分成两层：

```text
离线 Python：数据与标注 → 训练 → 验证 → 选 best.pt → 导出 ONNX/TensorRT
车端 C++：相机帧 → 同样预处理 → 推理 → 同样后处理 → 垃圾位置/掩码
```

现在选择 `YOLO11s-seg` 当教师，不会锁死以后部署模型。教师追求预标注质量；部署学生模型再根据车端算力选择 `YOLO11n-seg` 或 `YOLO11s-seg`，必要时用教师生成并经人工复核的数据训练学生。
