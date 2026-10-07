"""Privileged-state-free IL adapter. Each sample is 2 frames -> 8 actions."""
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

class FlipDataset:
    def __init__(self,root=ROOT,split='train',action_target='effective',image_size=96):
        self.root=Path(root)
        self.ids=json.loads((self.root/'data/splits.json').read_text())[split]
        if action_target not in ('effective','requested'):raise ValueError(action_target)
        self.action_target=action_target;self.image_size=image_size
    def __len__(self):return len(self.ids)*60
    def __getitem__(self,index):
        ep=self.ids[index//60];chunk=index%60;frames=[max(0,chunk-1),chunk]
        with h5py.File(self.root/'data/episodes'/f'episode_{ep:04d}.h5','r') as f:
            result={}
            for camera,key in [('global','image_overview'),('left_wrist','image_left'),('right_wrist','image_right')]:
                # Read only observations. Never expose diagnostics/config to policy.
                ims=[np.asarray(Image.fromarray(f['observations/rgb/'+camera][j]).resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)) for j in frames]
                result[key]=np.stack(ims).transpose(0,3,1,2)
            result['agent_pos']=np.stack([f['observations/proprio'][j] for j in frames]).astype(np.float32)
            result['action']=f['actions/'+self.action_target][chunk*8:(chunk+1)*8].astype(np.float32)
            result['remaining_time_s']=np.array([2-chunk/30],np.float32)
            result['reward']=np.float32(f['outcome/reward'][chunk*8:(chunk+1)*8].sum())
            result['terminated']=bool(f['outcome/terminated'][(chunk+1)*8-1])
            return result
