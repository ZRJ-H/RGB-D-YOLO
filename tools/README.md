# 工具索引

日常推理、训练和预标注使用根目录入口，本目录提供下载、配置、研究、报告与发布工具。先看 [项目首页](../README.md) 或 [日常工作流](../docs/WORKFLOWS.md)，再按下面的任务找脚本。

研究脚本保留特定采集批次的路径与假设，参见 [可复现范围](../docs/REPRODUCIBILITY.md)。新实验使用独立配置与输出名称；阅读已有报告不需要重新运行整套工具。

## 下载与本机配置

| 工具 | 作用 |
|---|---|
| [fetch_assets.py](fetch_assets.py) | 按 Release 清单下载 `models`、`reviewed`、`raw`、`public` 并恢复文件 |
| [configure_dataset.py](configure_dataset.py) | 从数据快照生成本机 YAML，重定位路径与训练图片列表 |

根目录的 [train_garbage_teacher.py](../train_garbage_teacher.py)、[predict_garbage_teacher.py](../predict_garbage_teacher.py) 分别训练候选、生成预标注；[add_reviewed_factory_dataset.py](../add_reviewed_factory_dataset.py) 导入人工审核数据。

## RGB 对照实验

| 工具 | 作用 |
|---|---|
| [build_overnight_dataset.py](build_overnight_dataset.py) | 从审核素材建立 Copy-Paste/真实重采样对照数据 |
| [run_overnight_training.py](run_overnight_training.py) | 按配置编排训练与评估，保存进度 |
| [evaluate_overnight_candidate.py](evaluate_overnight_candidate.py) | 基线与候选的公共验证、现场诊断和定性对照 |
| [summarize_overnight_training.py](summarize_overnight_training.py) | 汇总候选结果与选择依据 |
| [render_overnight_video.py](render_overnight_video.py) | 新旧模型的原时序视频对照 |

阅读背景：[2026-09-23 训练工作流](../docs/overnight-training-workflow-20260923.md)。日期配置属于历史实验，不直接作为新运行配置。

## 双目输入、视差与米制导出

| 工具 | 作用 |
|---|---|
| [preview_depth_bag.py](preview_depth_bag.py) | 从 bag 查看 RGB/深度材料 |
| [foundationstereo_runtime.py](foundationstereo_runtime.py) | 标准模型加载与本机 SDPA 适配，供其他脚本导入 |
| [run_stereo_disparity_trial.py](run_stereo_disparity_trial.py) | 历史左右 IR 的视差推理，可指定编号和输入比例 |
| [run_stereo_zip_batch.py](run_stereo_zip_batch.py) | Raw1 ZIP 的准备与批量推理 |
| [prepare_raw1_rectified.py](prepare_raw1_rectified.py) | 为已有 Raw1 批次准备校正后的输入 |
| [run_corrected_stereo_sample.py](run_corrected_stereo_sample.py) | 使用指定标定重跑一组 Raw1 样本 |
| [export_orbbec_ir_calibration.py](export_orbbec_ir_calibration.py) | 整理 Orbbec IR 标定材料 |
| [apply_stereo_calibration_trial.py](apply_stereo_calibration_trial.py) | 历史缩小推理的标定后处理与原网格恢复 |
| [export_native_stereo_batch.py](export_native_stereo_batch.py) | 原生视差批次的深度、XYZ 与点云导出 |
| [export_stereo_xyz_sample.py](export_stereo_xyz_sample.py) | 单组逐像素 XYZ 和点云样例 |
| [export_raw1_metric_release.py](export_raw1_metric_release.py) | Raw1 米制结果导出，显式区分平移单位 |
| [check_stereo_depth_trial.py](check_stereo_depth_trial.py) | 检查历史双目结果 |
| [summarize_stereo_batch.py](summarize_stereo_batch.py) | 汇总双目批次 |

## 几何诊断与报告

| 工具 | 作用 |
|---|---|
| [measure_cargo_review.py](measure_cargo_review.py) | 基于人工 ROI 与平面拟合的货物尺寸测量 |
| [diagnose_stereo_sample_outputs.py](diagnose_stereo_sample_outputs.py) | 检查单组视差、深度和点云输出 |
| [diagnose_raw1_rectification_hypotheses.py](diagnose_raw1_rectification_hypotheses.py) | 对比 Raw1 校正假设与稀疏匹配 |
| [build_cargo_measurement_report.py](build_cargo_measurement_report.py) | 货物测量阅读报告 |
| [build_stereo_batch_markdown_report.py](build_stereo_batch_markdown_report.py) | 全批次输入—处理—输出 Markdown 报告 |
| [build_raw1_stereo_report.py](build_raw1_stereo_report.py) | Raw1 双目批次报告 |
| [build_raw1_diagnosis_report.py](build_raw1_diagnosis_report.py) | Raw1 校正与点云诊断报告 |
| [build_stereo_rerun_review.py](build_stereo_rerun_review.py) | 单组独立重跑的报告与交付整理 |

## 结果打包

新包和暂存文件使用明确的发布目录；原 Windows 开发机统一写入 `E:/gpt/_release`。历史脚本中的旧输入路径需先适配。

| 工具 | 作用 |
|---|---|
| [build_xyz_review_package.py](build_xyz_review_package.py) | XYZ 单样例阅读包 |
| [build_stereo_pair_review_package.py](build_stereo_pair_review_package.py) | DOE 分组样例阅读包 |
| [build_stereo_batch_archive.py](build_stereo_batch_archive.py) | 双目全批次归档 |
| [package_corrected_stereo_sample.py](package_corrected_stereo_sample.py) | 指定标定单组结果包 |
| [build_preview_release.py](build_preview_release.py) | RGB + 独立深度预览应用包 |
| [preview_release_README.md](preview_release_README.md) | 上述应用包的历史 README 模板 |
| [verify_release.py](verify_release.py)、[verify_preview_archive.py](verify_preview_archive.py) | 对应发布格式的文件/归档完整性检查 |
| [preview_smoke_test.py](preview_smoke_test.py) | 预览发布包验收入口，打包后名为 `smoke_test.py` |

## Jetson 镜像

| 工具 | 作用 |
|---|---|
| [prepare_jetson_image.py](prepare_jetson_image.py) | 从已有应用发布包准备镜像上下文和材料 |
| [download_jetson_layers.py](download_jetson_layers.py) | 下载指定基础镜像层 |
| [assemble_jetson_image.py](assemble_jetson_image.py) | 组装离线应用镜像 |
| [check_jetson_archive.py](check_jetson_archive.py) | 检查镜像归档 |

实际运行方式见 [Jetson 部署说明](../deployment/jetson/README.md)，历史验收状态见项目首页。

检查脚本服务于对应实验或发布流程，不是日常改文档或小改动的必跑清单。只检查本次改动涉及的内容，不默认全量回归、冒烟或文件哈希。
