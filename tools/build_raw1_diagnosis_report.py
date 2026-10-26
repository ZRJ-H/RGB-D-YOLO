"""Preserve read-only diagnostic evidence and verified historical point-cloud paths."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from foundationstereo_runtime import sha256

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('E:/gpt/_release')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if BASE.resolve() not in out.parents or out.exists():
        raise ValueError('Use a new output under E:/gpt/_release')
    evidence={}
    for label,script,extra in (
        ('disparity_geometry','diagnose_stereo_sample_outputs.py',[]),
        ('calibration_hypotheses','diagnose_raw1_rectification_hypotheses.py',['--detailed'])):
        result=subprocess.run([sys.executable,'-B',str(ROOT/'tools'/script),*extra],capture_output=True,
                              text=True,encoding='utf-8',check=True,cwd=ROOT)
        evidence[label]=json.loads(result.stdout)
    groups=[
        ('本次新参数第1组，彩色',BASE/'foundationstereo_raw1_pair01_corrected_20260928_v1'),
        ('此前Raw1旧参数15组，T按毫米换算',BASE/'foundationstereo_raw1_metric_assumed_mm_20260928_v2'),
        ('历史47组中的46份点云，IR灰度',ROOT/'outputs/depth/foundationstereo_full_batch_20260928_v1/geometry'),
        ('第12组独立重跑，IR灰度',BASE/'foundationstereo_left_12_doe6_rerun_20260928_v1/geometry'),
        ('更早第25组半分辨率恢复样例',ROOT/'outputs/depth/orbbec_xyz_sample25_20260924'),
        ('更早DOE6/DOE0两组半分辨率恢复样例',ROOT/'outputs/depth/orbbec_cargo_pair2_xyz_20260924'),
        ('历史Raw1按32.4841米假设导出，非本次0.18025米结果',BASE/'foundationstereo_raw1_metric_assumed_m_20260928_v1'),
    ]
    inventory=[]
    for label,folder in groups:
        files=sorted([*folder.rglob('*.ply'),*folder.rglob('*.pcd')])
        inventory.append({'label':label,'root':str(folder),'files':[{'path':str(p),'bytes':p.stat().st_size} for p in files]})
    h=evidence['calibration_hypotheses']['results']
    g=evidence['disparity_geometry']
    names=[('原始图像','original'),('使用新K/D/R/T','new_as_supplied'),('将畸变D置零','new_D_zero'),
           ('将旋转R置为单位阵','new_R_identity'),('T只保留水平分量','new_T_horizontal'),
           ('完整反转外参','new_extrinsics_inverse'),('旧参数历史校正','old_historical_focal')]
    table='\n'.join(f"| {title} | {h[key]['abs_p50']:.3f} | {h[key]['abs_p90']:.3f} |" for title,key in names)
    labels={'raw_no_rectification':'原始图直接推理','previous_calibration':'旧参数校正后推理','corrected_calibration':'新参数校正后推理'}
    dtable='\n'.join(f"| {labels[v['variant']]} | {v['common_subset']['horizontal_abs_px']['p50']:.3f} | {v['common_subset']['horizontal_abs_px']['p90']:.3f} | {v['common_subset']['horizontal_gt3px_fraction']:.2%} |" for v in g['variants'])
    paths='\n'.join(f"- **{item['label']}**（{len(item['files'])}个PLY/PCD）：[{item['root']}]({Path(item['root']).as_posix()})" for item in inventory)
    report=f'''# Raw1第1组点云异常排查与历史路径

## 当前判断

首要异常出现在**这份标定与图片实际坐标状态的适用关系，以及由此产生的双目校正/匹配**。当前证据不足以把责任单独锁定为内参K，也不能直接断言原图已由SDK校正。应优先核对采集接口、是否去畸变/校正、相机模式与标定是否对应，再继续评估深度。

前次“核验通过”验证的是数值公式、坐标/颜色读回与文件一致性，不能证明空间形状或测量精度正确。本文没有将任何试验性参数改成正式标定，也没有覆盖点云；复用已保存的三套网络结果，未重新调用模型。

## 1. 已核验的历史点云位置

{paths}

当前新参数点云：[PLY](../foundationstereo_raw1_pair01_corrected_20260928_v1/cloud_rgb_m.ply) / [PCD](../foundationstereo_raw1_pair01_corrected_20260928_v1/cloud_rgb_m.pcd)。此前15组的第1组为[旧参数PLY](../foundationstereo_raw1_metric_assumed_mm_20260928_v2/pair_01/cloud_rectified_left_gray_provisional.ply)；第2～15组在同根目录 `pair_02`～`pair_15`，文件名相同。

历史47组中第24组缺匹配标定，没有PLY，因此该批只有46份。全部实际文件名、完整绝对路径和字节数见[path_inventory.json](path_inventory.json)。旧参数结果保留用于对照，不视为正确真值。`assumed_m`历史目录也不是本次基线换成0.18025米的结果，请勿混淆。

## 2. 单位和点云导出：没有发现实现错误

- 本次T=(-0.18025,0.000436713,-0.000468661)m；名义基线0.18025m，完整T长度0.1802511383081319m。两者只差约0.001138mm，不能解释明显变形。
- 重新用Q反投影全部有效点，与XYZ最大坐标差 {g['geometry_and_ply_recheck']['Q_max_abs_coordinate_error_m']:.3g}m（浮点舍入量级）。PLY的{g['geometry_and_ply_recheck']['valid_points']:,}个XYZ与RGB全部逐值相符。
- P1/P2/Q和校正图均采用同一个596px焦距；并非仅改深度公式、却遗漏校正映射。原始K约581/582与校正后596的区别本身不构成错误。
- 固定视差、投影参数时，仅将基线统一乘/除1000会整体缩放XYZ，不会单独制造局部弯折。这里确认的是**代码换算一致**，并不独立证明所给物理基线适用于当前图像。

## 3. 校正异常已经有直接图像证据

对同一组374对SIFT互检/RANSAC筛选的匹配点做变换，不在变换后重新挑点。它们是诊断特征，不是人工量测真值。

| 处理假设 | 垂直偏差绝对值P50(px) | P90(px) |
|---|---:|---:|
{table}

新校正后的左/中/右三区P50约 **5.494 / 2.890 / 0.951px**，原图为 **1.056 / 0.716 / 0.706px**。新增误差具有明显水平位置相关倾斜。R=I减少这部分倾斜，却留下约3.9px整体偏移；因此简单把R或D删掉不是可靠修复。全部假设和区域分布见[calibration_hypotheses.json](calibration_hypotheses.json)。

这些对照支持：不是单纯D畸变、T的微小Y/Z分量或外参简单转置造成；K/主点、D、R/T与图片坐标状态需要成套核对。旧参数的校正也没有证明正确。焦距不同的两种图像网格中像素残差不直接等价于物理误差，完整JSON另含旧参数统一596px的诊断。

水平双目正确校正应使对应点位于同一行；P/Q应与校正网格一致，见[OpenCV官方定义](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)。

## 4. 新校正后的视差也更不稳定

对三套结果共同位于图内、可采样且正视差的同一342对点，比较网络视差与稀疏匹配的水平位移。双线性采样采用OpenCV INTER_LINEAR。这不是稠密真值精度评测，但可定位劣化发生在导出点云之前。

| 已保存网络结果 | 视差差异P50(px) | P90(px) | 差异>3px比例 |
|---|---:|---:|---:|
{dtable}

新结果差异>3px的比例约为17.84%，原图直接推理约5.56%。原始图直接推理在这些点更一致，**不表示可直接给原图套596/0.18025的米制反投影**；若SDK已经校正，必须取得那个输出网格实际对应的P1/P2或Q。

另在白板内部、排除箱子/键盘/边缘的固定区域检查视差与二维仿射平面的残差：原始/旧/新结果RMS约0.020/0.023/0.051px，三者白板内部都相当平滑。这一局部检查不能解释或排除边缘、遮挡、整幅相对几何的问题，也不能仅凭截图把所有表面判为弯曲。[disparity_geometry.json](disparity_geometry.json)保存ROI与完整数据。

## 5. 下一步如何区分具体原因

1. 查清保存 `Image_Color1.png` 的相机SDK接口与开关：原始图、去畸变图还是双目校正图，是否resize/crop。用户此前说未校正，本次把它作为待复核事实；目前没有采集源码/SDK设置来独立确认。
2. 同一设备同一1280×720模式，同时保存一组传感器原始左右图及SDK校正后的左右图，并导出对应的K/D/R/T与P1/P2或Q。若原图已校正，避免再次使用原始K/D/R/T重映射。
3. 用独立棋盘图分别检查左右内参/畸变的重投影误差，再检查双目R/T的垂直角点残差。这样才能区分K/D错误、R/T错误与图片坐标状态不匹配。
4. 校正对应关系核实后，重新运行同一组，再用有实测距离/尺寸的物体验收。不要通过观察形状手调焦距或基线，把比例“调得像”。

本轮仅诊断，未声称点云问题已经修复。用户截图没有显示原文件路径，不能独立确认它打开的具体版本；以上数值针对本次 `foundationstereo_raw1_pair01_corrected_20260928_v1` 及明确列出的历史结果。
'''
    out.mkdir(parents=True)
    for name,value in {**evidence,'path_inventory':inventory}.items():
        (out/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    (out/'provenance').mkdir()
    for name in ('diagnose_stereo_sample_outputs.py','diagnose_raw1_rectification_hypotheses.py','build_raw1_diagnosis_report.py'):
        shutil.copyfile(ROOT/'tools'/name,out/'provenance'/name)
    evidencefiles=[p for p in out.rglob('*') if p.is_file()]
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'report':str(out/'REPORT.md'),
             'diagnostic_only':True,'old_outputs_modified':False,'new_network_calls':0,
             'files':{p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in evidencefiles}}
    (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(out/'REPORT.md'),'groups':len(inventory),'point_cloud_files':sum(len(i['files']) for i in inventory),'common_sparse_matches':342},ensure_ascii=False))


if __name__=='__main__':
    main()
