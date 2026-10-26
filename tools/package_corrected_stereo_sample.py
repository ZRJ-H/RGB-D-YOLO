"""Build the Chinese report, point-cloud preview, and verified single-pair archive."""
import argparse
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np
from foundationstereo_runtime import sha256
from run_corrected_stereo_sample import require, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    out = args.input.resolve()
    require(Path('E:/gpt/_release').resolve() in out.parents, 'Release path must be E:/gpt/_release')
    archive = out.with_suffix('.zip')
    require(not archive.exists() and not (out/'REPORT.md').exists(), 'Do not overwrite previous release')
    load = lambda name: json.loads((out/name).read_text(encoding='utf-8'))
    result, inference, diag, rect = [load(name) for name in ('result.json','inference.json','rectification_diagnostic.json','rectification.json')]
    require(inference['status']=='completed' and result['status']=='completed', 'Incomplete inference')
    require(load('verification.json')['status']=='passed', 'Geometry checks not passed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz = np.load(out/'xyz_m.npy', allow_pickle=False)
    valid = np.isfinite(xyz).all(axis=2)
    colour = cv2.cvtColor(cv2.imdecode(np.fromfile(out/'left_rectified_colour.png',np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    points, colours = xyz[valid],colour[valid]/255.
    step = max(1,int(np.ceil(len(points)/40000)))
    points, colours = points[::step],colours[::step]
    # Only the preview is thinned. Every valid point remains in PLY and PCD.
    fig = plt.figure(figsize=(10.5,8),facecolor='#f4f6f2')
    ax = fig.add_subplot(111, projection='3d', facecolor='#f4f6f2')
    ax.scatter(points[:,0],points[:,2],-points[:,1],c=colours,s=1.5,depthshade=False,linewidths=0)
    low, high = np.min(xyz[valid],axis=0),np.max(xyz[valid],axis=0)
    ax.set(xlabel='X right (m)',ylabel='Z forward (m)',zlabel='-Y up (m)',
           xlim=(low[0],high[0]),ylim=(low[2],high[2]),zlim=(-high[1],-low[1]))
    ax.set_box_aspect((high-low)[[0,2,1]])
    ax.view_init(elev=29,azim=-71)
    ax.set_title(f'Pair 01 | colour point cloud | {result["valid_points"]:,} exported points\nPreview: 1 in {step} points; optical coordinates, metres',pad=18)
    fig.tight_layout()
    fig.savefig(out/'point_cloud_preview.png',dpi=160)
    plt.close(fig)
    p = result['depth_percentiles_m']
    before,after = diag['original_vertical_abs_pixels'],diag['rectified_vertical_abs_pixels']
    text = f'''# 修正标定后，第1组双目深度与彩色点云

已使用本次提供的标定参数，重新校正 `Image_Depth_Raw1.zip` 的第1组 `Image_Color1.png`，并实际重新调用标准 FoundationStereo。输入单目和结果数组均为 **1280×720**，焦距统一为 **596 px**，所有深度、XYZ、PLY、PCD 坐标单位均为 **米**。

## 直接查看

- [彩色点云 PLY](cloud_rgb_m.ply) / [彩色点云 PCD](cloud_rgb_m.pcd)：{result['valid_points']:,} 个点，支持用 CloudCompare 等点云软件打开。两种文件表达同一份点云，选择其中一个即可。
- [深度数组](depth_m.npy)：720×1280，float32，数值为光轴深度 Z（米）。
- [逐像素 XYZ](xyz_m.npy)：720×1280×3，float32，`xyz[v,u]=[X,Y,Z]`（米）。无效位置三个通道均为 NaN。
- [视差数组](disparity_pixels.npy)、[有效掩膜](valid_mask.png)、[结果数据](result.json)。

![左右输入、视差与深度](preview.jpg)

![点云预览](point_cloud_preview.png)

点云预览为减少绘制量，每{step}点显示1点；PLY/PCD保留全部有效点。显示用坐标为X、Z、−Y，文件内始终是X右、Y下、Z前。

## 标定与单位

本次提供的新左右K、D、R以及T已完整写入[标定文件](calibration.json)。旧标定与旧输出均保留，没有覆盖。原始焦距约581/582用于去畸变，596用于校正后图像和三维几何，二者用途不同。

| 参数 | 本次采用值 |
|---|---:|
| 单目分辨率 | 1280×720 |
| 校正后 fx、fy | 596 px |
| 用户名义基线 | 180.25 mm = 0.18025 m |
| 原始 T（mm） | (−180.25, 0.436713, −0.468661) |
| 换算后 T（m） | (−0.18025, 0.000436713, −0.000468661) |
| 保留完整T的几何基线 ‖T‖ | {result['baseline_m']:.16f} m |
| 校正后共同主点 cx、cy | {rect['P1'][0][2]:.10f}、{rect['P1'][1][2]:.10f} px |

完整T的长度比名义水平分量多约0.001138毫米；为了P、Q及三维几何一致，实际反投影采用完整T的长度。所有结果已是米，**不要再除以1000**。

OpenCV `stereoRectify(CALIB_ZERO_DISPARITY)` 生成R1/R2与共同主点；再显式设置两目fx/fy=596，重建P1/P2/Q与校正映射。不能只把深度公式中的焦距换成596。完整矩阵保存在[rectification.json](rectification.json)，映射保存在[rectification_maps.npz](rectification_maps.npz)。

## 输入、处理和输出

1. [原始拼接图](source/Image_Color1.png)按1280列分为左右两目；沿用此前左半为左目、右半为右目的处理顺序。
2. 用本次完整K/D/R/T重新去畸变及双目校正；[左校正图](left_rectified.png)、[右校正图](right_rectified.png)。没有复用旧标定的视差。
3. 灰度重复为三通道输入标准FoundationStereo，原生scale=1、32迭代、max_disp=416、FP16/SDPA；内部补边到1280×736，网络后裁回1280×720。
4. 视差定义 `d=u_left-u_right`。当前主点相同，因此 `Z={result['f_times_baseline_px_m']:.12f}/d` 米，`X=(u-cx)Z/596`，`Y=(v-cy)Z/596`。
5. 掩膜要求正且有限的视差、左图校正源像素支持、右目对应位置及双线性邻点具有源图支持；这只是几何有效性，不是模型置信度。没有平滑、降采样或统计离群点清除。
6. 彩色取自[同一张左目彩色图的校正结果](left_rectified_colour.png)，与XYZ像素对应；不需要另一台RGB相机的跨相机配准。

坐标原点为校正后左相机光心，X向右、Y向下、Z向前。这里的Z是光轴深度，不是XYZ范数或货物尺寸。输出位于**校正后左图网格**，不能直接当原始像素网格或设备Raw图使用。

## 本次结果与核验

| 项目 | 结果 |
|---|---:|
| 实际网络调用 | 1次 |
| 网络耗时 | {inference['inference_seconds']:.3f}秒（不含加载、读写、几何导出） |
| 有效点数 / 比例 | {result['valid_points']:,} / {result['valid_fraction']:.2%} |
| 深度P2 / 中位数 / P98 | {p['p02']:.4f} / {p['p50']:.4f} / {p['p98']:.4f} m |
| 固定稀疏匹配点 | {diag['fixed_match_count']} |
| 校正前垂直偏差P50 / P90 | {before['p50']:.3f} / {before['p90']:.3f} px |
| 校正后垂直偏差P50 / P90 | {after['p50']:.3f} / {after['p90']:.3f} px |

**本组校正后的垂直偏差增大。** 这说明当前参数与该批图片的适用关系或图片预处理状态仍需核实；不能仅因程序完成，就判定新的标定已改善精度。原始未校正的说明沿用此前用户确认，但当前稀疏诊断不能独立确认其真实预处理状态，也不能单独定位原因。没有提供本组实物尺寸真值，本次不报告货物长宽高或测量精度。下一步应核对本组原始采集模式/校正状态与参数对应关系。

[极线预览](rectified_epipolar_lines.png)及[固定匹配诊断](rectification_diagnostic.json)供进一步检查。深度伪彩色按P2～P98映射，暖色近、冷色远、深灰无效；颜色饱和不表示数组截断。

已完成：全有效像素的Q独立反投影对照、XYZ-Z一致性、NaN与掩膜一致性、像素数值回投、PLY/PCD所有顶点与颜色读回核验；见[verification.json](verification.json)。这些是实现与文件核验，不是物理精度证明。

模型权重SHA256：`{inference['checkpoint_sha256']}`。运行环境见[inference.json](inference.json)，来源哈希见[source_manifest.json](source_manifest.json)。本次未训练、未改变原权重。

## 复现

运行脚本与依赖助手位于 `provenance/`。在本项目及已部署模型环境中执行以下命令，输出路径必须使用新的独立名称：

```powershell
& 'D:\\softwares\\Anaconda3\\envs\\foundation_stereo\\python.exe' -B tools/run_corrected_stereo_sample.py --source-zip 'C:\\Users\\leg\\Downloads\\Image_Depth_Raw1.zip' --calibration config/stereo_corrected_1280x720_20260928.json --output 'E:\\gpt\\_release\\NEW_UNIQUE_NAME' --number 1
& 'D:\\softwares\\Anaconda3\\envs\\foundation_stereo\\python.exe' -B tools/package_corrected_stereo_sample.py --input 'E:\\gpt\\_release\\NEW_UNIQUE_NAME'
```

归档包含本组原图、校正图、视差、深度、XYZ、两种点云、标定/映射、诊断、报告和脚本；不包含模型大权重或其完整Python环境。`SHA256.json`记录包内内容哈希，压缩包旁另有SHA256与归档核验文件。
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    shutil.copyfile(Path(__file__),out/'provenance'/Path(__file__).name)
    links=re.findall(r'\]\(([^)]+)\)',text)
    require(all((out/link).is_file() for link in links),'Broken report link')
    write_json(out/'report_verification.json',{'status':'passed','relative_links':links,'point_cloud_preview_sample_step':step})
    files=sorted(p for p in out.rglob('*') if p.is_file())
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files}
    write_json(out/'SHA256.json',manifest)
    files.append(out/'SHA256.json')
    partial=out.with_suffix('.partial.zip')
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as zf:
        for pth in files:
            zf.write(pth,out.name+'/'+pth.relative_to(out).as_posix())
    with zipfile.ZipFile(partial) as zf:
        require(zf.testzip() is None,'ZIP CRC failed')
        require(len(zf.namelist())==len(set(zf.namelist()))==len(files),'ZIP count mismatch')
        for name,item in manifest.items():
            import hashlib
            data=zf.read(out.name+'/'+name)
            require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'ZIP content hash failed')
        require(json.loads(zf.read(out.name+'/SHA256.json'))==manifest,'ZIP manifest mismatch')
    partial.rename(archive)
    receipt={'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),
             'entries':len(files),'hashed_content_files':len(manifest),'valid_points':result['valid_points']}
    write_json(out.with_suffix('.archive_verification.json'),receipt)
    archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
