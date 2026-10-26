"""Physical geometry checks independent of the real cargo's nominal dimensions."""
import unittest
import numpy as np
from tools.measure_cargo_review import fit_plane, intersect, edge_dimensions, rays_at, top_rectangle


class CargoGeometryTests(unittest.TestCase):
    def test_tilted_plane_recovers_known_oblique_rectangle(self):
        k=dict(fx=700.,fy=710.,cx=640.,cy=400.)
        n=np.array([.12,-.2,-1.]);n/=np.linalg.norm(n)
        ex=np.array([1.,0,0]);ex-=n*(ex@n);ex/=np.linalg.norm(ex)
        ey=np.cross(n,ex);centre=np.array([.2,.1,2.])
        corners=centre+np.array([[-.3,-.15],[.3,-.15],[.3,.15],[-.3,.15]])@np.stack([ex,ey])
        pixels=np.stack([corners[:,0]/corners[:,2]*k['fx']+k['cx'],corners[:,1]/corners[:,2]*k['fy']+k['cy']],1)
        recovered=intersect(pixels,k,n,-centre@n)
        np.testing.assert_allclose(recovered,corners,atol=1e-12)
        dimensions,_=edge_dimensions(recovered)
        np.testing.assert_allclose(dimensions,[600,300],atol=1e-8)

    def test_robust_plane_rejects_depth_outliers(self):
        rng=np.random.default_rng(1);xy=rng.uniform(-.5,.5,(2000,2))
        z=2+.1*xy[:,0]-.04*xy[:,1]+rng.normal(0,.0002,2000)
        z[:100]+=.08
        points=np.column_stack([xy,z]);n,d,report=fit_plane(points)
        self.assertLess(abs(np.array([0,0,2])@n+d),.0001)
        self.assertLess(report['rms_inlier_mm'],.3)
        self.assertLess(report['inliers'],1950)

    def test_support_rectangle_height_and_invalid_background(self):
        k=dict(fx=620.,fy=620.,cx=637.5,cy=406.)
        v,u=np.indices((800,1280));xyz=rays_at(np.stack([u,v],-1),k)*2
        xyz[:]=np.nan
        uv=np.stack([u[350:451,500:701],v[350:451,500:701]],-1)
        xyz[350:451,500:701]=rays_at(uv,k)*2
        top=dict(normal=[0,0,-1],offset_m=2)
        board=dict(normal=[0,0,-1],offset_m=2.3)
        result,mask=top_rectangle(xyz,k,[[480,330],[720,330],[720,470],[480,470]],top,board)
        np.testing.assert_allclose([result['length_mm'],result['width_mm']],[200*2000/620,100*2000/620],atol=.001)
        self.assertAlmostEqual(result['height_above_board_mm'],300,places=6)
        self.assertEqual(int(mask.sum()),201*101)


if __name__=='__main__':unittest.main()
