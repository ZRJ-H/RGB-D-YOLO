"""Auditable semi-manual cargo measurements, without using nominal sizes in fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Image-reviewed top-face corners, clockwise, in original LEFT IR pixels.
# These are review annotations, not training labels. No size prior is used.
CASES = {
    6: dict(kind='small', corners=[[801,471],[982,472],[982,539],[807,539]], board=[[625,290],[1035,290],[1035,425],[625,425]]),
    33: dict(kind='small', corners=[[702,474],[865,450],[875,519],[708,537]], board=[[625,290],[1035,290],[1035,420],[625,420]]),
    16: dict(kind='large', corners=[[464,285],[927,281],[933,533],[464,533]], board=[[975,295],[1040,295],[1040,510],[975,510]]),
    17: dict(kind='large', corners=[[471,233],[933,273],[909,523],[443,480]], board=[[980,300],[1040,300],[1040,505],[980,505]]),
}
SILHOUETTES = {
    6: [[779,461],[984,467],[985,540],[807,541],[779,520]],
    33: [[691,463],[867,444],[881,520],[706,541]],
    16: [[458,277],[932,276],[940,537],[457,539]],
    17: [[468,227],[939,273],[914,531],[438,484]],
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def polygon_mask(poly, shape=(800,1280), erosion=0):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 1)
    if erosion:
        mask=cv2.erode(mask, np.ones((2*erosion+1,2*erosion+1),np.uint8))
    return mask.astype(bool)

def fit_plane(points):
    points=np.asarray(points, float)
    points=points[np.isfinite(points).all(1)]
    if len(points)<100:
        raise ValueError('Insufficient independent surface samples')
    keep=np.ones(len(points),bool)
    for _ in range(8):
        c=points[keep].mean(0)
        _,_,vt=np.linalg.svd(points[keep]-c,full_matrices=False)
        n=vt[-1]
        if n[2]>0:n=-n
        residual=(points-c)@n
        mad=np.median(np.abs(residual[keep]-np.median(residual[keep])))
        new=np.abs(residual-np.median(residual[keep]))<max(.001,4*1.4826*mad)
        if np.array_equal(new,keep):break
        if new.sum()<100:raise ValueError('Plane rejected too many points')
        keep=new
    c=points[keep].mean(0)
    _,_,vt=np.linalg.svd(points[keep]-c,full_matrices=False)
    n=vt[-1]
    if n[2]>0:n=-n
    residual=(points-c)@n
    return n, -float(c@n), dict(normal=n.tolist(),offset_m=-float(c@n),centroid_m=c.tolist(),samples=len(points),inliers=int(keep.sum()),rms_inlier_mm=float(np.sqrt(np.mean(residual[keep]**2))*1000),p95_all_abs_residual_mm=float(np.percentile(np.abs(residual),95)*1000))

def rays_at(pixels, k):
    pixels=np.asarray(pixels,float)
    return np.stack([(pixels[...,0]-k['cx'])/k['fx'],(pixels[...,1]-k['cy'])/k['fy'],np.ones(pixels.shape[:-1])],-1)

def intersect(pixels, k, n, d):
    rays=rays_at(pixels,k)
    t=-d/(rays@n)
    if not np.all(t>0):raise ValueError('Intersection behind camera')
    return rays*t[...,None]

def edge_dimensions(corners):
    edges=np.linalg.norm(np.roll(corners,-1,axis=-2)-corners,axis=-1)*1000
    dimensions=np.stack([(edges[...,0]+edges[...,2])/2,(edges[...,1]+edges[...,3])/2],-1)
    return np.sort(dimensions,axis=-1)[...,::-1],edges

def top_rectangle(xyz, k, silhouette, top, board, tolerance=.010):
    """Enclose observed top support, using the same fixed tolerance for all cases.

    Manual silhouette is only an exclusion ROI (keeps nearby platform/rail out).
    The interior fit and top-support boundary do not use nominal dimensions.
    """
    n=np.array(top['normal']);d=top['offset_m']
    residual=xyz@n+d
    mask=polygon_mask(silhouette)&np.isfinite(residual)&(np.abs(residual)<tolerance)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count<2:raise ValueError('No top component')
    mask=labels==(1+np.argmax(stats[1:,4]))
    v,u=np.where(mask)
    p=intersect(np.stack([u,v],-1),k,n,d)
    ex=np.array([1.,0,0]);ex-=n*(ex@n);ex/=np.linalg.norm(ex);ey=np.cross(n,ex)
    origin=-d*n
    xy=np.stack([(p-origin)@ex,(p-origin)@ey],-1).astype(np.float32)
    rect=cv2.minAreaRect(xy);corners2=cv2.boxPoints(rect)
    corners=origin+corners2[:,0,None]*ex+corners2[:,1,None]*ey
    pixels=np.stack([corners[:,0]/corners[:,2]*k['fx']+k['cx'],corners[:,1]/corners[:,2]*k['fy']+k['cy']],-1)
    dims=sorted([float(z*1000) for z in rect[1]],reverse=True)
    height=float(np.mean(corners@np.array(board['normal'])+board['offset_m'])*1000)
    return dict(length_mm=dims[0],width_mm=dims[1],height_above_board_mm=height,
        top_corners_xyz_m=corners.tolist(),rectangle_corners_uv=pixels.tolist(),
        top_support_points=len(p),fixed_plane_residual_tolerance_mm=tolerance*1000,
        method='largest top-support component within visual silhouette; rays projected onto interior plane; minimum-area enclosing rectangle'),mask

def measure(depth, k, case, erosion=8, sample_step=1):
    yy,xx=np.indices(depth.shape)
    xyz=rays_at(np.stack([xx,yy],-1),k)*depth[...,None]
    sample=(xx%sample_step==0)&(yy%sample_step==0)
    bm=polygon_mask(case['board']) & sample
    tm=polygon_mask(case['corners'],erosion=erosion) & sample
    bn,bd,board=fit_plane(xyz[bm]);tn,td,top=fit_plane(xyz[tm])
    corners=intersect(case['corners'],k,tn,td)
    dims,edges=edge_dimensions(corners)
    h=float(np.mean(corners@bn+bd)*1000)
    heights=xyz[tm]@bn+bd
    direct=[]
    for u,v in case['corners']:
        # Median 3x3 depth is diagnostic only; never used to fit dimensions.
        z=float(np.nanmedian(depth[v-1:v+2,u-1:u+2]))
        direct.append(rays_at([u,v],k)*z)
    dd,de=edge_dimensions(np.array(direct))
    rng=np.random.default_rng(20260927)
    jitter=np.asarray(case['corners'])[None]+rng.uniform(-2,2,(2000,4,2))
    jd,_=edge_dimensions(intersect(jitter,k,tn,td))
    result=dict(length_mm=float(dims[0]),width_mm=float(dims[1]),height_above_board_mm=h,
        opposite_edges_mm=edges.tolist(),top_corners_xyz_m=corners.tolist(),
        top_plane=top,board_plane=board,
        top_to_board_angle_deg=float(np.degrees(np.arccos(np.clip(tn@bn,-1,1)))),
        interior_height_p05_p50_p95_mm=(np.percentile(heights,[5,50,95])*1000).tolist(),
        corner_jitter_2px_p05_p95_length_width_mm=np.percentile(jd,[5,95],axis=0).tolist(),
        corner_jitter_note='Sensitivity to chosen pixels only; not a physical accuracy confidence interval.',
        direct_corner_depth_dimensions_mm=dd.tolist(),direct_corner_depth_edges_mm=de.tolist(),
        top_roi_erosion_pixels=erosion,plane_sample_step=sample_step)
    return result,xyz,xyz@bn+bd,tm,bm

def ply(path, xyz, gray, mask):
    valid=mask & np.isfinite(xyz).all(-1)
    p=xyz[valid];c=gray[valid]
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    a=np.empty(len(p),dtype)
    for j,s in enumerate('xyz'):a[s]=p[:,j]
    for s in ['red','green','blue']:a[s]=c
    header=f'ply\nformat binary_little_endian 1.0\ncomment provisional left IR optical coordinates, metres\nelement vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with path.open('wb') as f:f.write(header.encode());a.tofile(f)
    return len(p)

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    source=ROOT/'outputs/depth/cargo_native4_20260927_v1'
    calpath=ROOT/'outputs/depth/orbbec_cargo_pair2_metric_20260924/calibration.json'
    cal=json.loads(calpath.read_text());k=cal['streams']['ir_left']['intrinsic']
    kr=cal['streams']['ir_right']['intrinsic']
    # Use the same validated ideal calibration contract as the earlier package.
    from apply_stereo_calibration_trial import validate_calibration
    _,baseline=validate_calibration(cal)
    y,x=np.indices((800,1280))
    out.mkdir(parents=True);shutil.copy2(calpath,out/'calibration.json')
    write_json(out/'annotations.json',dict(source='assistant visual review on original left IR; corners select interior fit, silhouette excludes surrounding equipment',cases=CASES,silhouettes=SILHOUETTES,truth_used_for_annotation=False,assumptions=['carton rests on white board','visible top perimeter represents rectangular carton footprint','historical IR matches supplied ideal rectified calibration']))
    rows=[]
    for number,case in CASES.items():
        doe=6 if number<24 else 0;folder=f'left_{number:02d}_doe{doe}';src=source/folder;dest=out/folder;dest.mkdir()
        meta=json.loads((src/'result.json').read_text())
        disp=np.load(src/'disparity_pixels.npy');im=cv2.imread(str(src/'left_input.png'),0)
        assert disp.shape==im.shape==(800,1280) and meta['input_size']==[1280,800]
        valid=np.isfinite(disp)&(disp>0)&(x-disp>=0)&(x-disp<=1279)
        depth=np.full(disp.shape,np.nan,np.float32);depth[valid]=k['fx']*baseline/(disp[valid]-(k['cx']-kr['cx']))
        np.save(dest/'depth_m_provisional.npy',depth);cv2.imwrite(str(dest/'left_ir.png'),im)
        cv2.imwrite(str(dest/'validity_mask.png'),valid.astype(np.uint8)*255)
        shutil.copy2(src/'disparity_pixels.npy',dest/'disparity_pixels.npy')
        result,xyz,h,tm,bm=measure(depth,k,case)
        manual={s:result[s] for s in ['length_mm','width_mm','height_above_board_mm','top_corners_xyz_m']}
        automatic,support=top_rectangle(xyz,k,SILHOUETTES[number],result['top_plane'],result['board_plane'])
        result['manual_corner_diagnostic']=manual
        result.update(automatic)
        result['top_support_threshold_sensitivity']=[]
        for threshold in [.003,.005,.020]:
            variant,_=top_rectangle(xyz,k,SILHOUETTES[number],result['top_plane'],result['board_plane'],threshold)
            result['top_support_threshold_sensitivity'].append(variant)
        result.update(number=number,doe=doe,kind=case['kind'],status='conditional_measurement_not_accuracy_acceptance',physical_accuracy_verified=False,native_resolution_inference=True,input_size=[1280,800],calibration_sha256=digest(calpath),source_left=meta['file'],source_right=meta['right_file'],source_left_sha256=digest(meta['file']),source_right_sha256=digest(meta['right_file']),source_disparity_sha256=digest(src/'disparity_pixels.npy'),corners_uv=case['corners'],depth_unit='metre',coordinate_frame='left_IR_optical_assuming_rectified_input')
        sensitivities=[]
        for erosion in [5,12,16]:
            r,px,*_=measure(depth,k,case,erosion=erosion)
            r,_=top_rectangle(px,k,SILHOUETTES[number],r['top_plane'],r['board_plane'])
            sensitivities.append(dict(erosion_pixels=erosion,dimensions_mm=[r['length_mm'],r['width_mm'],r['height_above_board_mm']]))
        result['interior_roi_sensitivity']=sensitivities
        if number==33:
            # Only evaluation after measurements; does not feed geometry or annotation.
            nominal=np.array([546,227,224]);vals=np.array([result['length_mm'],result['width_mm'],result['height_above_board_mm']])
            result['conditional_reference_comparison']=dict(reference_mm=nominal.tolist(),user_confirmed_unit_mm=True,user_identified_target='latest supplied pointcloud screenshot',source_frame_identification='SIFT comparison over 47 originals: frame33 has 23 homography inliers, 22 in cargo area; next-best 9; visually checked',original_pointcloud_filename_confirmed=False,assumption='reference length/width/height order; screenshot corresponds to frame33',signed_error_mm=(vals-nominal).tolist(),absolute_error_mm=np.abs(vals-nominal).tolist(),signed_relative_error_percent=((vals/nominal-1)*100).tolist())
        old=ROOT/'outputs/depth/orbbec_metric_native47_20260924'/folder/'depth_m_provisional.npy'
        if number in [6,33]:old=ROOT/'outputs/depth/orbbec_cargo_pair2_metric_20260924'/folder/'depth_m_provisional.npy'
        if old.exists():
            older=np.load(old);oldr,oldxyz,oldh,_,_=measure(older,k,case,sample_step=2)
            oldauto,_=top_rectangle(oldxyz,k,SILHOUETTES[number],oldr['top_plane'],oldr['board_plane'])
            result['half_resolution_comparison']=dict(source=str(old),source_sha256=digest(old),dimensions_mm=[oldauto['length_mm'],oldauto['width_mm'],oldauto['height_above_board_mm']],top_plane_rms_mm=oldr['top_plane']['rms_inlier_mm'],note='Same masks and 10mm threshold, plane fits use every second restored pixel to avoid duplicate half-resolution samples; boundary still lies on restored grid. Difference alone is not accuracy improvement.')
        else:oldh=None
        # Local vertical profiles pass through the visible top at its centre.
        v=int(np.mean(np.array(case['corners'])[:,1]));lo=max(0,min(c[0] for c in case['corners'])-35);hi=min(1279,max(c[0] for c in case['corners'])+35)
        fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
        ax.plot(np.arange(lo,hi),np.nanmedian(h[v-2:v+3,lo:hi],axis=0)*1000,label='Native 1280x800 inference')
        if oldh is not None:ax.plot(np.arange(lo,hi),np.nanmedian(oldh[v-2:v+3,lo:hi],axis=0)*1000,label='640x400 inference restored',alpha=.8)
        ax.set(xlabel='Original left IR u (pixel)',ylabel='Height above fitted board (mm)',title=f'Frame {number} / DOE {doe} / profile v={v} +/-2 px');ax.grid(alpha=.25);ax.legend();fig.savefig(dest/'edge_profile.png',dpi=150);plt.close(fig)
        overlay=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        rectangle=np.rint(result['rectangle_corners_uv']).astype(np.int32)
        cv2.polylines(overlay,[np.array(SILHOUETTES[number])],True,(100,100,255),1)
        cv2.polylines(overlay,[rectangle],True,(0,220,255),2)
        cv2.polylines(overlay,[np.array(case['board'])],True,(255,220,0),2)
        for j,(u,vv) in enumerate(rectangle):cv2.circle(overlay,(u,vv),4,(0,0,255),-1);cv2.putText(overlay,str(j+1),(u+4,vv-5),0,.55,(0,0,255),2)
        cv2.putText(overlay,f'#{number} DOE{doe}: {result["length_mm"]:.1f} x {result["width_mm"]:.1f} x {result["height_above_board_mm"]:.1f} mm',(25,35),0,.7,(0,255,255),2)
        cv2.putText(overlay,'CONDITIONAL / manual ROI + fitted top rectangle / calibration pending',(25,65),0,.55,(0,255,255),1)
        cv2.imwrite(str(dest/'measurement_overlay.jpg'),overlay)
        heat=cv2.applyColorMap(np.uint8(np.nan_to_num(np.clip(h/.35,0,1))*255),cv2.COLORMAP_TURBO);heat[~valid]=0
        cv2.polylines(heat,[rectangle],True,(255,255,255),1);cv2.imwrite(str(dest/'height_0_to_350mm.png'),heat)
        cv2.imwrite(str(dest/'top_support_mask.png'),support.astype(np.uint8)*255)
        cloudmask=polygon_mask(SILHOUETTES[number])|bm
        result['review_cloud_points']=ply(dest/'top_and_reference_board.ply',xyz,im,cloudmask)
        np.save(dest/'top_corners_xyz_m.npy',np.array(result['top_corners_xyz_m']))
        write_json(dest/'measurement.json',result);rows.append(result)
        print(number, [round(result[s],2) for s in ['length_mm','width_mm','height_above_board_mm']],flush=True)
    report=dict(status='conditional_measurement_not_accuracy_acceptance',method='interior top-plane robust SVD; 10mm top support within manual silhouette; enclosing rectangle on plane; board normal distance at top centre',truth_used_for_fitting=False,RGB_registration_used=False,calibration_verified_for_historical_input=False,reference_applies_only_to_frame33=True,rows=rows)
    write_json(out/'measurements.json',report)
    with (out/'measurements.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['frame','DOE','kind','length_mm','width_mm','height_above_board_mm','physical_accuracy_verified'])
        for r in rows:writer.writerow([r['number'],r['doe'],r['kind'],r['length_mm'],r['width_mm'],r['height_above_board_mm'],False])
    shutil.copy2(__file__,out/'measure_cargo_review_source.py')

if __name__=='__main__':main()
