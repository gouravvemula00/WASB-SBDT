import os.path as osp
import pandas as pd
import numpy as np

from utils import Center

def load_csv_tennis(csv_path, visible_flags, frame_dir=None, bounce_column=None):
    df = pd.read_csv(csv_path)
    fnames, visis, xs, ys = df['file name'].tolist(), df['visibility'].tolist(), df['x-coordinate'].tolist(), df['y-coordinate'].tolist()
    
    # Check if bounce column exists, if not create default values
    if bounce_column and bounce_column in df.columns:
        bounces = df[bounce_column].tolist()
    else:
        # If no bounce column, try to infer from trajectory or set to False
        bounces = [False] * len(fnames)
    
    xyvs = {}
    for fname, visi, x, y, bounce in zip(fnames, visis, xs, ys, bounces):
        fid = int(osp.splitext(fname)[0])
        if frame_dir is not None:
            frame_path = osp.join(frame_dir, fname)
        else:
            frame_path = None

        if fid in xyvs.keys():
            raise KeyError('fid {} already exists'.format(fid ))
        
        if np.isnan(x) or np.isnan(y):
            if (int(visi) in visible_flags):
                print(visible_flags)
                print(fname)
                print(visi, x, y)
                print(int(visi))
                quit()

        xyvs[fid] = {'center': Center(x=float(x),
                                      y=float(y),
                                      is_visible=True if int(visi) in visible_flags else False,
                                      is_bounce=bool(bounce) if not pd.isna(bounce) else False,
                               ),
                     'file_name': fname,
                     'frame_path': frame_path
                     }

    return xyvs

