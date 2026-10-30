# Jetson Orin Nano 8GB 环境镜像

## 实机结果与适用范围

2026-10-08 已在 Orin Nano Super、Ubuntu 22.04、L4T 36.4.7 上完成 Docker GPU 自检和图片推理。实际运行的是 RGB 分割与独立深度预览，输出 `spatial_fusion=false`、RGB 实例 `geometry=null`；实时相机、RGBD 配准和稳定帧率尚未验收。

本目录保留原交付的构建与启动脚本，`run_jetson.sh` 固定检查 L4T 36.4.4。上述 36.4.7 实机验证通过单独的运行脚本调用容器完成，该脚本未包含在本仓库中；不要直接把下方旧启动命令套到 36.4.7 环境。

容器内代码位于 `/opt/yolo-depth`，权重位于 `/opt/yolo-depth/models/best.pt`；运行结果写入绑定到 `/outputs` 的宿主机目录。新的运行使用独立输出目录。原开发机的新镜像包、构建暂存及交付材料统一放在 `E:/gpt/_release`，其他机器使用对应的数据盘位置。

项目入口见 [README](../../README.md)，模块关系见 [工程架构](../../docs/ARCHITECTURE.md)。

## 原交付的构建流程

配套应用：YOLO + Depth `v0.1.0-preview`。目标主机：JetPack **6.2.1** / Jetson Linux **36.4.4**，ARM64。

这是装好应用与依赖的 **Docker 离线应用镜像**。板子先烧录 NVIDIA JetPack，再导入本镜像；不能把 Docker tar 写入SD卡当系统镜像启动。
初次镜像准备时板卡尚未烧录，下列步骤供重新准备环境时参考。第三方载板使用其厂商对应的 BSP 与烧录步骤。

## 固定的环境来源

- 基础镜像：`ultralytics/ultralytics:8.3.179-jetson-jetpack6`。
- 固定摘要：`sha256:ddde06a91b658955e203499a92308ccce3ba4fdf8ebbfa8ad65a2573902b1761`。
- 基础系统：Ubuntu 22.04 / ARM64，CUDA 12.6；上游镜像构建历史包含Jetson专用PyTorch 2.5.0a0（nv24.08）及torchvision 0.20.0a0。
- 沿用基础镜像中的完整依赖，不用Windows的CUDA wheel，也不在Jetson上通过普通pip替换torch。
- 包含当前最佳 `best.pt`、独立深度模块、6组样例、全部发布文件和板端自检入口。
- 基础镜像内库的实际版本以 `doctor` 输出为准；主机JetPack版本与容器库版本应分别记录。

## 第一步：烧录与主机准备

1. 根据实际开发套件/载板，按照厂商流程烧录JetPack 6.2.1。开发套件初始固件若为JetPack 5.x，还要按NVIDIA说明更新固件。
2. 启动板子，配置网络与SSH，确认 `/etc/nv_tegra_release` 显示R36、REVISION 4.4，`uname -m` 为aarch64。
3. 安装或确认Docker、NVIDIA Container Toolkit，`docker info` 的 Runtimes 包含 `nvidia`。已有Docker时，可用 `sudo nvidia-ctk runtime configure --runtime=docker` 配置运行时，再按需重启Docker服务；该操作在目标板进行。
4. 在保存离线包的磁盘和Docker数据磁盘预留足够空间。完整CUDA开发基础镜像较大，建议总空闲空间至少40GB；容量以最终交付清单为准。

## 第二步：离线导入

将本次生成的 `yolo-depth_0.1.0-preview-jp6.2.1-arm64.tar` 及 `.sha256` 文件复制到板子。

```bash
sha256sum -c yolo-depth_0.1.0-preview-jp6.2.1-arm64.tar.sha256
sudo docker load -i yolo-depth_0.1.0-preview-jp6.2.1-arm64.tar
sudo docker image inspect yolo-depth:0.1.0-preview-jp6.2.1-arm64 --format '{{.Os}}/{{.Architecture}}'
```

应显示 `linux/arm64`。在完成板端自检前，该镜像仅为待验收版本。

## 第三步：板端自检

交付包中的 `run_jetson.sh` 检查CPU架构和目标L4T版本，并使用全新的输出目录；请在有Docker权限的账号运行，或使用sudo。

```bash
sudo bash run_jetson.sh ./jetson_diagnostics_01 doctor --require-cuda --output /outputs/environment.json
sudo bash run_jetson.sh ./jetson_acceptance_01
```

第二条命令执行 `self-test`：发布文件校验 → Python依赖导入 → CUDA矩阵运算与torchvision NMS → 30项单元测试 → 合成深度 → 6组真实RGB/深度推理 → 错误输入与覆盖保护。
结果位于 `jetson_acceptance_01/acceptance/report.json`。自检使用 `--network none`，验证运行不依赖在线下载。
默认不授予容器 `--privileged` 或相机访问；实时相机/ROS接入属于后续工作。

## 单张图片试跑

```bash
sudo bash run_jetson.sh ./jetson_preview_01 paired-preview \
  --rgb /opt/yolo-depth/samples/sample_02_rgb.jpg \
  --depth /opt/yolo-depth/samples/sample_02_depth_raw.png \
  --device 0 --output /outputs/result
```

自带图片已经旋转180度，不再旋转。输出实例掩膜、叠加图、独立深度预览与JSON；没有相机标定时不产生实例米制几何，不声明RGBD空间融合。

## 重建镜像

交付的 `context/` 目录有完整Dockerfile、应用与容器入口，可在拥有足够空间且能访问基础镜像的Docker环境重建：

```bash
docker build --platform linux/arm64 -t yolo-depth:0.1.0-preview-jp6.2.1-arm64 ./context
docker save -o yolo-depth_0.1.0-preview-jp6.2.1-arm64.tar yolo-depth:0.1.0-preview-jp6.2.1-arm64
```

本机C盘容量有限时，使用 `download_jetson_layers.py` 分段下载固定基础镜像，随后通过 `assemble_jetson_image.py` 将已校验镜像层和应用层组装为标准Docker镜像归档。其语义与Dockerfile的COPY、ENV、WORKDIR和ENTRYPOINT相同，不执行依赖安装，也不上传模型或现场图片到远程镜像仓库。镜像配置保留官方基础层的全部环境，逐层解压摘要与发布文件摘要会在封装完成前核对。

## 验证边界

已有 Windows 发布包的 CPU/GPU 测试不等于 ARM64 容器的运行验收。2026-10-08 的实机记录证明当时环境中的 GPU 自检和图片推理通过；新的镜像或设备仍需检查其实际运行情况。`BUILD_STATUS.json` 记录的是对应交付时的状态，不能替代新环境的时延、内存或散热测量。
模型仍是预标注用途，原有误检漏检未因打包自动改善。基础镜像的AGPL及各第三方许可沿用原条款。

## 官方参考

- [NVIDIA JetPack 6.2.1及烧录入口](https://developer.nvidia.com/embedded/jetpack-sdk-621)
- [Ultralytics Jetson部署指南](https://docs.ultralytics.com/guides/nvidia-jetson/)
- [Docker镜像归档格式规范](https://github.com/moby/docker-image-spec/blob/main/spec.md)
