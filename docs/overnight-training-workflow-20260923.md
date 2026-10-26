# 2026-09-23 低人工垃圾分割训练

用户授权：不希望大量新增人工标注，由自动工作流训练候选，次日验收。

本轮复用已审核标注进行程序式 Copy-Paste 增强，不引入未审核的模型预测，不调用文生图服务，也不把候选自动替换为当前最佳模型。原始数据和原权重保持不变。

## 输入与数据

- Python：`D:\softwares\Anaconda3\envs\yolov8\python.exe`，PyTorch 2.7.1+cu128，Ultralytics 8.3.179；GPU RTX 5070 Ti 16GB。
- 起点：旧工程 `outputs/training/garbage_teacher_yolo11s_seg_reviewed20_r10_lr5e5_e10/weights/best.pt`，SHA256 `06c69e0ed6ca59a5b19f8d5ee4b7de5a226bfdd2787e7eb0ee07da2a9464f910`。
- 独立新数据目录：`datasets/garbage_overnight_reviewed_cp300_20260923`。
- 1834张公共训练图，17张已确认工厂负样本，38张CVAT审核图（重复6次）。旧seed32矩形提示数据不加入本轮训练；起始权重历史仍接触过该来源，不能抹除其影响。
- 基线组另加300条已审核真实图重采样；增强组另加300张合成图，共693实例、79个审核实例素材。两组每轮均2379条；同样18轮、batch 8、imgsz 960、seed 20260923、AdamW lr0 8e-5，无warmup、无提前停止。
- 审核标签中的严格轴对齐矩形不作合成素材；合成标签为审核多边形的确定性几何变换。只粘贴到已查看的17张确认负样本中的地面区域。
- 公共验证集450张保留原字节。8张round3审核图、46实例仅作同视频未见图回归，未进入本轮训练及合成来源。
- 这8帧与基线过去训练帧相邻约1–2秒，不能宣称独立场景测试；公开val也已经用于历史模型选择。审计见 `docs/overnight-data-audit-20260923.md`。
- 142张定性对照包含91张旧相机帧、47张新Orbbec RGB、4张鞋子回归帧；不得把候选数量当作精度。

## 工作流

1. `tools/build_overnight_dataset.py`：审核来源核验、数据隔离、增强、坐标/哈希/图像解码验证、掩膜总览。
2. `tools/run_overnight_training.py`：两组分进程训练和评估，状态/逐轮指标原子落盘，限制6小时，低磁盘空间时停止当前子任务；仅在运行期间申请系统保持唤醒，允许屏幕休眠。
3. `tools/evaluate_overnight_candidate.py`：相同参数比较公共验证、有限现场真值、142张定性图像；输出中文HTML、三栏图片和明确非原始时间的幻灯MP4。
4. `tools/summarize_overnight_training.py`：按重新测量的公共mask mAP50-95排名选候选，给出退化提示，不根据8张现场诊断挑模型，不修改当前best。
5. `tools/render_overnight_video.py`：909帧、10FPS完整相机视频的新旧模型并排回归，保存首中尾并全量解码验收。

配置：`config/overnight_training_20260923.json`。主输出：`outputs/overnight/20260923_reviewed_synthesis_v1`。

该配置用于已经启动的一次实验，不能原地重复启动。重跑必须使用新的输出目录和实验名；不能删除运行标记后覆盖现有结果。

## 验收与状态

- `STATUS.json`：主进程、子进程、当前阶段、更新时间、错误、完成状态；只有最终状态和实际文件校验能证明交付完成。
- `progress_reviewed_control_e18.json` / `progress_reviewed_copypaste_e18.json`：每轮进度和验证指标。
- `logs/`：完整训练/评估/打包日志。
- `acceptance.html`：最终验收入口；主流程尚在训练时该文件可以不存在。
- `SUMMARY.json`：候选选择和筛查结果；`selected_candidate.pt`与SHA256文件是候选交付。
- `full_video/comparison.mp4`：完整原时序并排视频；`video_report.json`为解码验收结果。
- 不存在真正独立工厂测试集；由用户次日结合真实场景验收，不能仅凭公开mAP自动晋升生产模型。

已完成前置验证：8图1轮GPU训练试跑；同一权重作为基线/候选的评估烟雾测试，公共与8帧回归所有指标差值均为0；对照图和幻灯视频生成成功。该小批次只验证流程，不代表模型精度。

当前任务附加15分钟心跳巡检（automation id `yolo`），只在完成、失败、明显停滞或需要动作时通知。完成后检查成品并停止巡检；最多两次必要恢复，2026-09-24 08:00前给出已有结果与状态。保持电脑通电和应用开启。
