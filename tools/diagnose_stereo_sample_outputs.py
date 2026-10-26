"""Read-only comparison of stored raw/old/new stereo disparities against fixed matches."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np
from prepare_raw1_rectified import transform_points

BASE = Path('E:/gpt/_release')
CURRENT = BASE/'foundationstereo_raw1_pair01_corrected_20260928_v1'
HISTORY = BASE/'foundationstereo_image_depth_raw1_20260928_v1'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(a):
    return dict(zip(('p50','p90','p98'),map(float,np.percentile(a,[50,90,98]))))


def sample(a, xy):
    return cv2.remap(a,xy[:,0].astype(np.float32).reshape(-1,1),xy[:,1].astype(np.float32).reshape(-1,1),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan')).ravel()


def main():
    diag=load(CURRENT/'rectification_diagnostic.json')
    pl,pr=[np.array(diag[k],np.float64) for k in ('original_points_left','original_points_right')]
    old=load(HISTORY/'rectified/rectification.json')
    oldcal=load(HISTORY/'rectified/calibration.json')
    oldpts=[transform_points(p,np.array(oldcal[k]),np.array(oldcal[d]),np.array(old[r]),np.array(old[proj]))
            for p,k,d,r,proj in ((pl,'cameraMatrixL','distCoeffL','R1','P1'),(pr,'cameraMatrixR','distCoeffR','R2','P2'))]
    newpts=[np.array(diag[k],np.float64) for k in ('rectified_points_left','rectified_points_right')]
    original_roi=np.zeros((720,1280),np.uint8)
    # Visible, mostly textureless interior of the white board; excludes box/keyboard/edges.
    roi_poly=np.array([[687,112],[997,87],[997,228],[705,263]],np.int32)
    cv2.fillPoly(original_roi,[roi_poly],255)
    with np.load(CURRENT/'rectification_maps.npz') as maps:
        newroi=cv2.remap(original_roi,maps['left_map_x'],maps['left_map_y'],cv2.INTER_NEAREST)>0
    with np.load(HISTORY/'rectified/rectification.npz') as maps:
        oldroi=cv2.remap(original_roi,maps['left_map_x'],maps['left_map_y'],cv2.INTER_NEAREST)>0
    variants=[('raw_no_rectification',HISTORY/'pair_01/disparity_pixels.npy',(pl,pr),original_roi>0),
              ('previous_calibration',HISTORY/'rectified/pair_01/disparity_pixels.npy',oldpts,oldroi),
              ('corrected_calibration',CURRENT/'disparity_pixels.npy',newpts,newroi)]
    common=np.ones(len(pl),bool)
    for _,path,(l,r),_ in variants:
        predicted=sample(np.load(path,allow_pickle=False),l)
        common &= np.isfinite(predicted)&(l[:,0]-r[:,0]>0)&(l[:,0]>=0)&(l[:,0]<1279)&(l[:,1]>=0)&(l[:,1]<719)&(r[:,0]>=0)&(r[:,0]<1279)&(r[:,1]>=0)&(r[:,1]<719)
    results=[]
    for name,path,(l,r),roi in variants:
        disp=np.load(path,allow_pickle=False)
        predicted=sample(disp,l)
        dx,dy=l[:,0]-r[:,0],l[:,1]-r[:,1]
        finite=np.isfinite(predicted)&(dx>0)&(l[:,0]>=0)&(l[:,0]<1279)&(l[:,1]>=0)&(l[:,1]<719)&(r[:,0]>=0)&(r[:,0]<1279)&(r[:,1]>=0)&(r[:,1]<719)
        abs_error=np.abs(predicted-dx)
        row={'variant':name,'sparse_count':int(finite.sum()),'vertical_abs_px':stats(np.abs(dy[finite])),
             'model_vs_sparse_horizontal_abs_px':stats(abs_error[finite]),
             'model_vs_sparse_horizontal_gt3px_fraction':float((abs_error[finite]>3).mean()),
             'disparity_model_over_sparse_median':float(np.median(predicted[finite]/dx[finite])),
             'disparity_model_px':stats(predicted[finite]),'disparity_sparse_px':stats(dx[finite])}
        row['common_subset']={'count':int(common.sum()),'vertical_abs_px':stats(np.abs(dy[common])),
            'horizontal_abs_px':stats(abs_error[common]),'horizontal_gt3px_fraction':float((abs_error[common]>3).mean())}
        row['vertical_residual_bins']=[]
        for low,high in ((0,1),(1,3),(3,100)):
            keep=finite&(np.abs(dy)>=low)&(np.abs(dy)<high)
            row['vertical_residual_bins'].append({'range_px':[low,high],'n':int(keep.sum()),
                'horizontal_abs_px':stats(abs_error[keep]) if keep.any() else None})
        yy,xx=np.where(roi&np.isfinite(disp)&(disp>0))
        A=np.column_stack((xx,yy,np.ones(len(xx))))
        values=disp[yy,xx].astype(float)
        coeff=np.linalg.lstsq(A,values,rcond=None)[0]
        residual=values-A@coeff
        row['white_board_disparity_affine_fit']={'n':len(xx),'median_disparity_px':float(np.median(values)),
            'residual_abs_px':stats(np.abs(residual)),'residual_rms_px':float(np.sqrt(np.mean(residual**2))),
            'coefficient_du_dv_offset':coeff.tolist(),
            'interpretation':'Flat-surface proxy, not independently measured plane or depth ground truth; raw input may not be rectified.'}
        results.append(row)
    disp=np.load(CURRENT/'disparity_pixels.npy')
    xyz=np.load(CURRENT/'xyz_m.npy')
    mask=cv2.imdecode(np.fromfile(CURRENT/'valid_mask.png',np.uint8),cv2.IMREAD_GRAYSCALE)>0
    rect=load(CURRENT/'rectification.json')
    Q=np.array(rect['Q'])
    independent=cv2.reprojectImageTo3D(disp,Q)
    err=np.abs(xyz[mask].astype(float)-independent[mask])
    plytype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    with (CURRENT/'cloud_rgb_m.ply').open('rb') as f:
        header=[]
        while True:
            line=f.readline()
            header.append(line)
            if line==b'end_header\n':
                break
            if not line:
                raise ValueError('Invalid PLY header')
        verts=np.fromfile(f,plytype)
    coords=np.column_stack((verts['x'],verts['y'],verts['z']))
    np.testing.assert_array_equal(coords,xyz[mask])
    rgb=cv2.cvtColor(cv2.imdecode(np.fromfile(CURRENT/'left_rectified_colour.png',np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(np.column_stack((verts['r'],verts['g'],verts['b'])),rgb[mask])
    summary={'source':'Existing saved arrays; no new network calls and no existing files modified',
        'same_original_sparse_matches':len(pl),'white_board_roi_original_vertices':roi_poly.tolist(),
        'sparse_matches_are_ground_truth':False,'variants':results,
        'geometry_and_ply_recheck':{'Q_max_abs_coordinate_error_m':float(err.max()),'PLY_XYZ_RGB_all_points_match':True,
            'nominal_baseline_m':.18025,'actual_full_T_baseline_m':rect['baseline_m'],
            'baseline_norm_difference_relative':rect['baseline_m']/.18025-1,
            'valid_points':int(mask.sum()),'unit_conversion_changes_uniform_scale_only':True}}
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
