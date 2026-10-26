# 可复现范围与历史脚本

仓库于 2026-10-26 汇总现有 RGB 标注训练代码、独立深度模块和实验工具。

- 主要入口：`run_perception.py`、`run_depth_baseline.py`、`train_garbage_teacher.py`、`predict_garbage_teacher.py`，输入路径均可显式提供。
- `tools/` 中带 Raw1、Orbbec、overnight、report、archive 的脚本保留原实验逻辑。部分默认路径、输入索引和报告依赖原开发机，运行前查看 `--help` 和源码；不属于克隆后立即可运行的通用 API。
- `config/*historical.json`、历史标定 JSON、历史文档中的绝对路径是实验记录，不是新环境配置。不能把某台设备的标定直接用于另一设备或不同像素网格。
- FoundationStereo 源码及其完整环境不内嵌。权重与源码版本/配置必须匹配；本项目没有重新训练该网络。
- 原始 bag 与双目数据用于保留输入，未审核的模型输出不属于训练真值。
- 发布包、暂存文件、校验文件的本机默认归档位置为 `E:/gpt/_release`，不写入原工程 C 盘 releases 目录。
- 历史文档中的旧验收状态保留时间语境；最新板端结论见主 README。
