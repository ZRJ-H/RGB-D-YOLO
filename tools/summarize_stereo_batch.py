"""Validate the 47-pair trial and build contact sheets."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/depth/orbbec_standard_all47_20260923'
report = json.loads((out/'report.json').read_text(encoding='utf-8'))
rows = report['results']
assert len(rows) == 47 and len({r['number'] for r in rows}) == 47
for row in rows:
    folder = out / f"left_{row['number']:02d}_doe{row['doe']}"
    disp = np.load(folder/'disparity_pixels.npy', allow_pickle=False)
    inv = np.load(folder/'inverse_disparity_nonmetric.npy', allow_pickle=False)
    assert disp.shape == (row['input_size'][1], row['input_size'][0])
    valid = np.isfinite(disp) & (disp>0)
    assert valid.any() and np.allclose(inv[valid]*disp[valid], 1, atol=1e-6)
    for name in ['disparity_colour.png','positive_finite_mask.png','horizontal_overlap_mask.png','left_input.png','right_input.png','preview.jpg']:
        with Image.open(folder/name) as image:
            image.verify()
for start in range(0,47,12):
    canvas = Image.new('RGB',(1200,880),(24,24,24))
    draw = ImageDraw.Draw(canvas)
    for j,row in enumerate(rows[start:start+12]):
        folder = out / f"left_{row['number']:02d}_doe{row['doe']}"
        im = Image.open(folder/'disparity_colour.png').convert('RGB')
        im.thumbnail((390,185))
        x,y = (j%3)*400,(j//3)*220
        canvas.paste(im,(x,y))
        label = f"{row['number']:02d} | DOE={row['doe']} | {row['input_size']}"
        if row['timestamp_delta_raw']:
            label += ' | PAIR?'
        draw.text((x+5,y+190),label,fill='white')
    canvas.save(out/f"overview_{start+1:02d}_{min(start+12,47):02d}.jpg",quality=93)
summary = {'verified_pairs':len(rows),'doe6':sum(r['doe']==6 for r in rows),'doe0':sum(r['doe']==0 for r in rows),
           'pairing_unverified_numbers':[r['number'] for r in rows if r['timestamp_delta_raw']],
           'small_resolution_numbers':[r['number'] for r in rows if r['size'] != [1280,800]],
           'metric_depth_and_pointcloud_generated':False}
(out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(out/'README.md').write_text('''# 标准FoundationStereo全部47组输出

权重：D:/RC/lege/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth。
32次迭代、半分辨率、PyTorch SDPA兼容层。原始采集文件未修改，未进行校正。

- disparity_pixels.npy：模型直接输出的float32视差，单位为本次输入图像的像素。
- inverse_disparity_nonmetric.npy：1/视差，单位为1/像素；不是米制深度。
- disparity_colour.png：每帧按2%和98%视差分位拉伸；跨帧同色不代表同距离。
- positive_finite_mask.png：正且有限的视差位置，不是准确性或置信度。
- horizontal_overlap_mask.png：进一步满足右图水平坐标范围的位置；不保证匹配正确。
- left_input.png / right_input.png：实际缩小后的输入灰度图。
- preview.jpg：从上到下为左图、右图、视差。
- result.json：来源、DOE、时间戳、配对状态、分辨率和推理时延。

overview联系表供浏览；report.json汇总各组信息；verification.json记录文件验收。
米制深度、三维坐标和PLY点云等待可信标定后生成，没有套用旧相机参数或拟合尺度。
最近时间戳相差1的那组仅作暂定配对，需复核；848×480组保持独立尺寸。
''',encoding='utf-8')
print(json.dumps(summary))
