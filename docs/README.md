# 文档导航

第一次阅读建议按 [项目首页](../README.md) → [工程架构](ARCHITECTURE.md) → [日常工作流](WORKFLOWS.md) 的顺序。需要某个脚本时查 [工具索引](../tools/README.md)，迁移历史实验前查 [可复现范围](REPRODUCIBILITY.md)。

## 工程与使用

| 文档 | 内容 |
|---|---|
| [工程架构](ARCHITECTURE.md) | RGB、深度与双目分工，数据流、接口与代码阅读顺序 |
| [日常工作流](WORKFLOWS.md) | 下载资产、本地数据配置、推理、人工复核和训练入口 |
| [工具索引](../tools/README.md) | 按任务查找仓库实际提供的工具 |
| [可复现范围](REPRODUCIBILITY.md) | 可移植入口、外部依赖和历史实验路径 |
| [独立深度使用说明](depth-baseline-usage.md) | 深度算法接口、内参与质量标记 |
| [RGB 标注流水线](garbage-auto-annotation.md) | 数据转换、CVAT 审核集导入、训练与预测的历史记录 |
| [Jetson 部署说明](../deployment/jetson/README.md) | 镜像构建和板端启动材料；实际验收状态以项目首页为准 |

## 算法与实验专题

| 文档 | 内容与适用范围 |
|---|---|
| [FoundationStereo 流程](foundationstereo-process-notes-20260928.md) | 左右图输入、网络视差、尺寸、深度与点云换算 |
| [货物尺寸测量方法](cargo_measurement_v3_dimension_method_20260928.md) | 特定批次的人工 ROI、拟合平面和长宽高计算 |
| [Copy-Paste 训练工作流](overnight-training-workflow-20260923.md) | 2026-09-23 实验的组织方式与数据使用范围 |
| [仓库汇总验证记录](repository_validation_20261026.json) | 2026-10-26 整理发布时的检查记录 |

带日期的文档保留当时的路径、条件和过程。其中的旧运行状态或机器绝对路径不是新环境配置，也不代表报告和数据已随 Git 克隆。下载内容以 [资产清单](../assets/manifest.json) 和项目首页的说明为准。

## 维护文档

稳定接口与日常用法写入本目录；单次实验的图表、日志和完整报告放在该实验输出目录，导航中只保留入口和结论。新增工具时更新 [工具索引](../tools/README.md)。日常维护只检查本次改动直接影响的部分，不默认重跑全部实验、回归、冒烟或文件哈希。
