# YOLO + Depth v0.1.0-preview 测试发布包

这是本机交接与部署准备使用的源码/权重包。它包含 RGB 实例分割、无标定的原始深度预览、合成深度几何测试，以及需调用者提供标定的独立深度入口。
本版没有空间融合、时间跟踪、板端加速或精度提升声明，不是自动清扫/避障控制程序。

## 环境

推荐 Python 3.10。已测试的开发机环境详见 `environment.json` 和 `VALIDATION.md`。
安装与目标硬件匹配的 PyTorch/torchvision 后执行：

```text
python -m pip install -r requirements.txt
```

`requirements.txt` 是安装约束，不是跨平台锁文件；`environment.json` 记录本次实际测试版本。
现有开发机的 OpenCV 4.12 与 NumPy 1.24 组合可运行，但与 OpenCV 包声明的依赖范围不一致；新环境由 pip 解析兼容版本，不应直接照抄该组合。
未提供离线 wheel 包；首次安装依赖可能需要联网。板卡型号未知，未打包 ARM 专用 wheel、TensorRT/RKNN 引擎或系统镜像。
默认 CPU；支持 CUDA 的环境可指定 `--device 0`。所有运行输出目录必须是新目录。
Ultralytics 配置默认写入包内 `.runtime/`，避免依赖旧账号的用户配置；因此解压目录应可写，也可以通过 `YOLO_CONFIG_DIR` 指定可写目录。

## 快速开始

在解压后的包根目录执行：

```text
python verify_release.py
python run_perception.py depth-synthetic --output runs/synthetic_01
python run_perception.py paired-preview --rgb samples/sample_00_rgb.jpg --depth samples/sample_00_depth_raw.png --output runs/preview_01
python run_perception.py rgb --rgb samples/sample_00_rgb.jpg --output runs/rgb_01
python smoke_test.py --device cpu --output runs/acceptance_01
```

也可以从任意工作目录通过脚本绝对路径调用；默认模型和配置相对于脚本定位。输入及输出参数仍相对于调用时的工作目录。

`paired-preview` 输出：

- `result.json`：RGB 实例、独立深度非零比例/原始值分位数及明确的能力标志；
- `mask_*.png`：每个实例的原始 RGB 分辨率二值掩膜（0/255）；无检测时没有这些文件；
- `rgb_annotated.jpg`：预标注叠加图；
- `depth_colour.png` / `depth_nonzero.png`：原始深度伪彩色及非零像素图；
- `preview.jpg`：两路并排显示，仅为可视化，不表示像素对应。

原始深度颜色按每帧2/98分位数缩放，不能根据不同帧的颜色比较绝对距离；非零比例也不等同于测量准确率。
JSON 的 `spatial_fusion=false`；RGB 实例的 `geometry=null`。类别固定为 `0 sweepable_litter`。
默认 `imgsz=960, conf=0.25, iou=0.35, batch=1`，输出完整分割掩膜。

## 样例来源与方向

6组样例来自本机 `test_camera1` bag 的既有预览，已旋转180度，因此默认 `--rotate 0`。
新相机原始画面若仍倒置，使用 `--rotate 180`；不要对已有样例再次旋转。
RGB 为640×480，深度为640×400。仅做时间最近邻配对；既有记录时间差约5.47～14.05 ms，详见 `samples/provenance.json`。
CLI 本身不核对时间戳，也不估计配准；禁止把深度直接缩放后作为垃圾掩膜的几何测量依据。
样例不是独立标注评估集，测试通过只说明程序可运行，不能推导模型准确率或融合效果。

## 有真实标定时的独立深度模式

复制 `config/intrinsics.template.yaml`，填入真实的深度相机内参。显式提供已确认的深度单位到米的比例：

```text
python run_perception.py depth-calibrated --depth YOUR_DEPTH.png --intrinsics YOUR_INTRINSICS.yaml --depth-scale YOUR_VERIFIED_SCALE --output runs/calibrated_01
```

模板中的 null 不能直接使用；不得拿合成测试内参或未经确认的0.001比例代替实机参数。
标定模式仅输出深度相机坐标系几何，仍不进行 RGB 配准。原 `run_depth_baseline.py` 为兼容保留，真实数据优先用要求显式比例的新入口。

## 验收与范围

`smoke_test.py` 运行随包单元测试、合成0.2m突出物、全部6组真实样例，并检查掩膜、图像尺寸、深度统计和无标定保护。结果写入指定目录。
`verify_release.py` 使用 SHA-256 检查发布清单覆盖的原始文件；新增 `runs/` 不影响校验。校验用于完整性检查，不是发布者数字签名。
`validation/` 与 `VALIDATION.md` 保存本次实际执行证据。延迟包含冷启动，不能当作稳定帧率基准。

已知限制：鞋子/反光等误检仍可能存在；不自动删除 RGB 候选，不将预标注当训练真值；没有板端、干净环境安装或真实 RGBD 精度验收。
本包包含现场图片及原有模型，仅在授权范围内交接。未指定新的开源许可；第三方代码/模型和数据的原有许可继续适用。
