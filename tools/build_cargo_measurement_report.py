"""Build an offline, reviewable report and verify numerical artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from html.parser import HTMLParser
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/depth/cargo_measurement_review_20260927_v3'

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    report=json.loads((OUT/'measurements.json').read_text(encoding='utf-8'))
    rows=report['rows'];target=next(r for r in rows if r['number']==33)
    evidence=ROOT/'outputs/depth/cargo_measurement_20260927_v1'
    for name in ['screenshot_retrieval.json','screenshot_match_best.json','screenshot_matches_frame33.jpg']:
        shutil.copy2(evidence/name,OUT/name)
    shutil.copy2(ROOT/'outputs/depth/cargo_measurement_review_20260927_v2/new_camera_diagnostic.json',OUT/'new_camera_diagnostic.json')
    shutil.copy2(Path(r'C:/Users/leg/AppData/Local/Temp/codex-clipboard-3c2a3fc6-3636-4c5e-9efc-bde8f732a39a.jpg'),OUT/'user_reference_screenshot.jpg')
    checks=[]
    k=json.loads((OUT/'calibration.json').read_text())['streams']['ir_left']['intrinsic']
    yy,xx=np.indices((800,1280))
    for r in rows:
        folder=f"left_{r['number']:02d}_doe{r['doe']}";p=OUT/folder
        d=np.load(p/'disparity_pixels.npy');z=np.load(p/'depth_m_provisional.npy');mask=cv2.imread(str(p/'validity_mask.png'),0)>0
        assert d.shape==z.shape==mask.shape==(800,1280)
        np.testing.assert_array_equal(mask,np.isfinite(z))
        np.testing.assert_allclose(z[mask],620.138794*.0952326965/d[mask],rtol=1e-6)
        c=np.load(p/'top_corners_xyz_m.npy');lengths=np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1)*1000
        np.testing.assert_allclose(sorted(lengths[:2],reverse=True),[r['length_mm'],r['width_mm']],atol=.001)
        bn=np.array(r['board_plane']['normal']);bd=r['board_plane']['offset_m']
        np.testing.assert_allclose(np.mean(c@bn+bd)*1000,r['height_above_board_mm'],atol=1e-6)
        assert cv2.imread(str(p/'top_support_mask.png'),0).astype(bool).sum()==r['top_support_points']
        with (p/'top_and_reference_board.ply').open('rb') as f:
            header=[]
            while True:
                line=f.readline();header.append(line.decode().strip())
                if line==b'end_header\n':break
            vertex_count=int(next(s for s in header if s.startswith('element vertex')).split()[-1])
            payload=f.read();assert len(payload)==vertex_count*15
        assert vertex_count==r['review_cloud_points']
        for image in p.glob('*.png'):assert cv2.imread(str(image),-1) is not None
        checks.append(dict(frame=r['number'],native_grid=True,formula=True,rectangle_edges=True,height=True,ply_vertices=vertex_count))
    values=np.array([target['length_mm'],target['width_mm'],target['height_above_board_mm']])
    np.testing.assert_allclose(values-[546,227,224],target['conditional_reference_comparison']['signed_error_mm'])
    summary='''# 货物尺寸测量试验 · 2026-09-27

用户最新指定的点云截图对应目标为546×227×224毫米。对47张原IR进行特征检索后，第33组DOE0最符合：23个单应性内点，其中22个位于货物区域；次高仅9个。已目视核对纹理、方向与周围场景。原PLY文件名未由用户确认，因此保留截图来源识别依据。

## 本轮结果

第33组原生1280×800重新推理，估计 **545.2×225.0×224.7 mm**。
与参考546×227×224 mm相比，测量值减参考值约 **-0.8、-2.0、+0.7 mm**（约-0.14%、-0.90%、+0.32%）。
同一方法测原640×400推理再恢复的结果约 **527.7×230.0×230.9 mm**。两次采用同一标准FoundationStereo权重、32迭代，未训练。

这是一组已知货物上的条件性估计，不是系统精度验收，也不是独立测试集结果。不能承诺所有货物误差在2毫米以内。

## 做法与边界

1. 深度由原生视差及原IR内参/95.2326965mm基线换算，保持1280×800网格；未使用RGB配准。
2. 人工圈定货物排除区域和白色承载板区域；货物内部区域向内收缩8像素，拟合顶面；承载板单独拟合。使用迭代SVD及MAD异常点过滤。
3. 在排除区域内提取距离顶面小于10mm的最大连通区域，沿像素射线投影到拟合顶面，计算最小包围矩形长宽。高度为该矩形中心到承载板的法向距离。必须假设箱体平放、底部接触板面且顶面轮廓代表外形尺寸。
4. 所有4组统一使用10mm阈值，546/227/224仅用于最后计算误差，不作为尺寸拟合或比例校正输入。不过本试验方法是在已知真值后开发，不属于盲测；后续需锁定参数后用新货物验收。
5. 第33组将顶面阈值改为3/5/20mm，长约541.6～547.1mm、宽约220.8～228.4mm。这是参数敏感性，不是置信区间。图像ROI也来自人工，尚不是全自动测量系统。
6. 原IR历史校正状态仍未确认；所给JSON呈现理想校正几何。本次米制结果以其确实适用于历史图像为前提。平面残差小可能反映网络平滑，不能当成真实精度。

## 补充货物与边缘问题

第6组DOE6约539.5×234.5×226.0mm，本轮不将截图真值自动套到它。
第16/17组大箱分别约1507.0×815.5×142.0mm、1509.4×815.4×137.5mm。没有这只大箱实测真值；两帧一致性不等于精度，摆放变化也不是严格重复性试验。
DOE6及大箱在严格顶面阈值下会丢失部分边缘，表明阈值策略仍有局限。单纯放大旧深度或增加PLY点数不能恢复观测细节。这里重新做了原生推理，并以内部平面和顶面边界估计尺寸，没有修改原深度去强行补齐真值。

过程版v1直接采用人工顶面四角，存在内缩选边问题（第33组约509.6×203.8mm），不能作为交付测量。最终v3用区域内顶面支持点求矩形，保留manual_corner_diagnostic供审计；直接读取边角深度的结果也仅作不稳定性诊断。

## 新相机

新ZIP有15组。抽查1/8/15组，按所给参数再次校正后，稀疏特征的垂直偏差中位数由约0.8px增至1.0～1.3px。这不能证明参数错误，但说明不能直接假定输入是待校正原图；可能存在已校正、模式或坐标关系差异。Raw含义、比例和标定T单位仍缺明确声明，所以本轮不生成新相机毫米测量。详见new_camera_diagnostic.json；其统计以左右拼接顺序及Raw像素对应等未验证假设为前提，不能据此确定Raw编码。

## 文件

打开START_HERE.html查看标记图、剖面和对照表。measurements.csv可用Excel打开；每组measurement.json含完整参数、敏感性和来源哈希。
depth_m_provisional.npy是float32光轴深度，单位米，无效NaN；disparity_pixels.npy单位为原图像素。top_and_reference_board.ply包含货物ROI与承载板参考点，XYZ单位米、左IR光学坐标系、灰度颜色。不是全场景点云，不是RGB配准。
annotations.json保存人工区域；原始数据与旧权重均保留，没有加入训练集。
measure_cargo_review_source.py是工程内运行脚本副本，重跑需原项目及数据、NumPy/OpenCV/Matplotlib；工具不会覆盖已有输出。

## 下一步

先确认历史IR与标定的适用关系，并用相同锁定参数测多件新货物、多个距离和摆放位置；记录三轴绝对误差、最大误差及成功率。大箱需补实测尺寸；新相机需补Raw定义/比例、左右顺序/校正状态、T单位和校正后P1/P2或Q。
'''
    (OUT/'README.md').write_text(summary,encoding='utf-8')
    table=''
    cards=''
    for r in rows:
        name=f"left_{r['number']:02d}_doe{r['doe']}";old=r['half_resolution_comparison']['dimensions_mm']
        dimension=' × '.join(f'{r[key]:.1f}' for key in ['length_mm','width_mm','height_above_board_mm'])
        table+=f'<tr><td>第{r["number"]}组 · DOE{r["doe"]}</td><td>{dimension}</td><td>{" × ".join(f"{v:.1f}" for v in old)}</td><td>{"546 × 227 × 224" if r["number"]==33 else "未绑定实测真值"}</td></tr>'
        threshold=' / '.join(f'{v["fixed_plane_residual_tolerance_mm"]:.0f}mm阈值：{v["length_mm"]:.1f} × {v["width_mm"]:.1f}mm' for v in r['top_support_threshold_sensitivity'])
        cards+=f'''<article><h2>第{r['number']}组 · DOE{r['doe']}　{dimension} mm</h2><p>黄色：估计顶面矩形；青色：参考板区域；细红线：人工排除区域。</p><a href="{name}/measurement_overlay.jpg"><img src="{name}/measurement_overlay.jpg" alt="货物与参考区域标记"></a><img src="{name}/edge_profile.png" alt="新旧高度剖面对比"><p>{threshold}</p><p><a href="{name}/measurement.json">详细结果</a> · <a href="{name}/top_and_reference_board.ply">货物与参考板点云 PLY</a> · <a href="{name}/depth_m_provisional.npy">原尺寸深度 NPY</a> · <a href="{name}/top_support_mask.png">顶面支持区域</a> · <a href="{name}/height_0_to_350mm.png">高度图（0～350mm，超范围饱和）</a></p></article>'''
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>货物尺寸测量 · 第33组复核</title><style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;background:#f4f6f7;color:#20373b;line-height:1.8}}h1{{font-size:32px}}article,.box{{background:white;padding:24px;border-radius:14px;margin:24px 0}}img{{max-width:100%;height:auto}}.notice{{background:#fff2d7;padding:18px;border-left:4px solid #ad762b}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:13px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#006e73}}.big{{font-size:34px;font-weight:bold}}small{{color:#53656a}}@media(max-width:700px){{table{{font-size:12px}}th,td{{padding:5px}}.big{{font-size:24px}}}}</style><body><small>2026-09-27 · FoundationStereo · 原生1280×800推理 · 本地复核</small><h1>你指定的货物：第33组 DOE0</h1><div class="box"><div class="big">545.2 × 225.0 × 224.7 mm</div><p>实测参考：546 × 227 × 224 mm<br>测量值减实测：<b>−0.8 / −2.0 / +0.7 mm</b></p></div><p class="notice">这是单个已知货物上的条件性结果，不能据此承诺系统毫米级精度。人工选择区域；历史校正与标定适用关系仍待确认。顶面阈值改变时，第33组长宽仍会变化数毫米。方法开发时已知道真值，后续需要锁定参数进行独立验证。</p><p><a href="README.md">完整中文说明</a> · <a href="measurements.csv">结果表CSV</a> · <a href="measurements.json">结构化报告</a> · <a href="annotations.json">人工区域</a> · <a href="verification.json">核验记录</a></p><h2>原尺寸推理与旧半分辨率结果</h2><p>单位均为毫米；顺序为长 × 宽 × 相对承载板高度。所有组采用相同10mm顶面阈值。</p><table><tr><th>图像</th><th>本次原生推理</th><th>旧半分辨率恢复</th><th>实测参考</th></tr>{table}</table><article><h2>截图来源核对</h2><p>47张原IR检索中，第33组有23个几何一致匹配，22个位于货物区域；次高9个。纹理与摆放经目视复核吻合。截图对应关系有较强支持，原始PLY文件名尚未确认。</p><img src="screenshot_matches_frame33.jpg" alt="用户截图与原IR第33组特征匹配"><a href="screenshot_retrieval.json">全部47组检索结果</a></article>{cards}<article><h2>新相机尚不输出毫米尺寸</h2><p>15组材料已检查。抽查再次校正后的垂直匹配偏差反而增大，需要确认输入是否已校正、Raw定义及比例、T单位。原样保留数据，未套用Orbbec标定。</p><a href="new_camera_diagnostic.json">新相机诊断</a></article><p>7项几何与网格回归通过；数组公式、矩形边长、高度、PLY读回已核验。这些是实现检查，不等于物理精度验收。</p></body></html>'''
    (OUT/'START_HERE.html').write_text(page,encoding='utf-8')
    save('verification.json',dict(status='passed_implementation_checks_not_physical_accuracy',geometry_tests=7,frames=checks,truth_comparison_arithmetic=True))
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['src','href']:assert (OUT/value).is_file(),value
    Links().feed(page)
    for version in ['v1','v2']:
        p=ROOT/f'outputs/depth/cargo_measurement_review_20260927_{version}'
        (p/'SUPERSEDED.md').write_text('过程结果，请使用 ../cargo_measurement_review_20260927_v3/START_HERE.html。v1人工边界内缩，且真值绑定假设已被用户更正；v3仅对有截图匹配证据的第33组做实测误差对照。保留旧文件供审计。',encoding='utf-8')
    files={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256.json'}
    save('SHA256.json',files)
    archive=Path(r'E:\gpt\_release')/'cargo_measurement_review_20260927_v3.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise ValueError('Archive already exists')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    verification=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),content_files=len(files),all_hashes_verified=True,zip_crc_verified=True)
    archive.with_suffix('.verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
