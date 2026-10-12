"""Read COLMAP text models and per-image camera matrices."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from recon.common.camera import frame_intrinsics

def parse_model(path):
    intrinsics=frame_intrinsics(path)
    points={}
    for line in (path/'points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            f=line.split();points[int(f[0])]=np.array([float(x) for x in f[1:4]])
    lines=[line for line in (path/'images.txt').read_text().splitlines() if line and not line.startswith('#')]
    frames=[]
    for a,b in zip(lines[::2],lines[1::2]):
        f=a.split();q=np.array([float(x) for x in f[1:5]])
        R=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix();t=np.array([float(x) for x in f[5:8]])
        obs=np.asarray(b.split(),dtype=float).reshape(-1,3)
        frames.append((int(Path(f[9]).stem),R,t,obs))
    return intrinsics,points,sorted(frames,key=lambda f:f[0])

