#!/usr/bin/env python3
"""Export a COLMAP text model to this project's trajectory and colored PLY."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from recon.common.progress import done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--min-track',type=int,default=3)
    parser.add_argument('--max-error',type=float,default=2.0)
    parser.add_argument('--sequence',type=Path,
                        help='Keep only 3D tracks observed inside selected-object masks')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    lines=[line for line in (args.model/'images.txt').read_text().splitlines()
           if line and not line.startswith('#')]
    rows=[]
    masked_support={}
    for image_line, observations_line in zip(lines[::2], lines[1::2]):
        if args.sequence:
            image_fields=image_line.split()
            mask=cv2.imread(str(args.sequence/'masks'/image_fields[9]),cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(args.sequence/'masks'/image_fields[9])
            data=observations_line.split()
            if len(data)%3:
                raise ValueError('Invalid COLMAP 2D observation list')
            for n in range(0,len(data),3):
                point_id=int(data[n+2])
                if point_id<0:continue
                x,y=int(round(float(data[n]))),int(round(float(data[n+1])))
                if 0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]>0:
                    masked_support[point_id]=masked_support.get(point_id,0)+1
    for line in lines[::2]:
        fields=line.split()
        frame=int(Path(fields[9]).stem)
        q=np.array([float(fields[k]) for k in range(1,5)])
        w2c=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
        t=np.array([float(fields[k]) for k in range(5,8)])
        center=-w2c.T@t
        rows.append([frame,'tracked',*center,*Rotation.from_matrix(w2c.T).as_quat(),
                     0,0,0,0])
    with (args.output/'trajectory.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['frame','status','tx','ty','tz','qx','qy','qz','qw',
                         'matches','inliers','landmarks','processing_ms'])
        writer.writerows(sorted(rows,key=lambda row:row[0]))
    points=[];colors=[]
    for line in (args.model/'points3D.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        fields=line.split()
        if len(fields)<8+2*args.min_track or float(fields[7])>args.max_error:continue
        if args.sequence and masked_support.get(int(fields[0]),0)<args.min_track:continue
        points.append([float(value) for value in fields[1:4]])
        colors.append([int(value) for value in fields[4:7]])
    with (args.output/'landmarks.ply').open('w') as stream:
        stream.write('ply\nformat ascii 1.0\n')
        stream.write(f'element vertex {len(points)}\n')
        stream.write('property float x\nproperty float y\nproperty float z\n')
        stream.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for xyz,rgb in zip(points,colors):
            stream.write(' '.join(map(str,xyz+rgb))+'\n')
    summary={'registered_frames':len(rows),'supported_landmarks':len(points),
             'min_track':args.min_track,'max_reprojection_error_px':args.max_error,
             'object_mask_filter':bool(args.sequence)}
    (args.output/'export.json').write_text(json.dumps(summary,indent=2)+'\n')
    done(f"Exported {summary['registered_frames']} camera poses and {summary['supported_landmarks']} landmarks")

if __name__=='__main__':main()
