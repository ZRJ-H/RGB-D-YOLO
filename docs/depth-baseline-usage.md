# 深度检测第一版使用说明

工程入口见 [项目首页](../README.md)，常用命令见 [日常工作流](WORKFLOWS.md)。真实数据优先使用 `run_perception.py depth-calibrated`，它要求显式提供已确认的深度单位比例；下文保留原有深度入口的接口说明。

## 定位

这一版只负责深度几何，不判断垃圾语义，也不决定清扫或避障。输出中的
`geometry_confidence`表示几何测量质量，不能与YOLO的语义置信度混用。

核心入口：

```python
from depth_perception import CameraIntrinsics, DepthObstacleDetector, FrameMetadata

result = DepthObstacleDetector(config).process(
    depth_image,
    intrinsics,
    metadata=FrameMetadata(
        sequence_id=sequence_id,
        timestamp_ns=timestamp_ns,
        source_id="camera_03",
        frame_id="camera_03_depth_optical_frame",
    ),
)
```

未来YOLO分割结果接入时，先将实例掩膜配准到深度图的像素坐标系，再将布尔掩膜传给（尺寸相同本身不代表已配准）：

```python
measurement = detector.measure_mask(depth_image, intrinsics, instance_mask)
```

该接口会返回目标的相机坐标、地面轮廓、面积、高度、有效深度像素数和几何
置信度。RGB类别、语义置信度和深度测量仍保持为不同字段。

## 文件试跑

先从相机标定结果建立一个YAML文件，必须包含以下六个字段：

```yaml
width: 640
height: 400
fx: 由深度相机标定结果填写
fy: 由深度相机标定结果填写
cx: 由深度相机标定结果填写
cy: 由深度相机标定结果填写
```

确认 `config/depth_baseline.yaml` 中的 `depth_scale` 与实际深度单位一致后运行。默认 `0.001` 只适用于原始数值已确认是毫米的输入；米制数组应设为 `1`。

```bash
python run_depth_baseline.py --depth path/to/depth.png --intrinsics path/to/camera_03_depth_intrinsics.yaml --config config/depth_baseline.yaml --rotate 180 --output outputs/depth/depth_baseline_run_name
```

输出目录不能已存在，避免覆盖历史结果。输出包含：

- `result.json`：地面模型、质量标志和全部突出物几何结果；
- `object_mask.png`：独立深度算法识别到的突出物掩膜。

## 当前真实数据限制

`test_camera1`中已确认深度为`16UC1`、640×400；RGB为640×480，首个观测帧
时间差约8 ms。bag没有记录`CameraInfo`、RGB/深度对齐关系或外参。因此，在
拿到原始标定参数之前，只能验证数据读取和算法结构，不能对真实bag声称准确的
米制位置、面积或RGB掩膜融合结果。

相机画面需要旋转180度。命令行的`--rotate 180`会同时旋转深度像素并修正标定
主点，不能只旋转图像而继续使用未旋转的内参。

## 第一版算法边界

- 能检测高于拟合地面的、深度可见的连续突出物；
- 平铺薄膜、薄纸等可能低于深度噪声，不应把“深度未检出”解释成“RGB误检”；
- 地面ROI、深度范围、高度阈值和连通域阈值全部来自配置文件；
- 地面拟合失败或有效深度不足时输出质量标志，不伪造目标；
- 输出坐标当前明确为深度相机光学坐标系，不冒充地图坐标。

