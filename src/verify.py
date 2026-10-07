"""Audit every shard and replay effective actions without expert feedback."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import h5py,numpy as np,mujoco
from collect import ROOT,CAMS,setup_env,task_state

def verify_one(path):
    with h5py.File(path,'r') as f:
        cfg=json.loads(f.attrs['config_json']);result=json.loads(f.attrs['result_json']);idx=int(f.attrs['episode_id'])
        assert result['success']
        assert np.all(f['diagnostics/stable'][-48:]) and np.min(f['diagnostics/bottom'][:])>=.74
        np.testing.assert_allclose(f['observations/timestamp'][:],np.arange(61)/30,atol=1e-12)
        np.testing.assert_allclose(f['actions/timestamp'][:],np.arange(480)/240,atol=1e-12)
        for c in CAMS:
            ds=f['observations/rgb/'+c];assert ds.shape==(61,240,320,3) and ds.dtype==np.uint8
            # Read every compressed chunk to detect corruption.
            assert ds[:].std()>0
            np.testing.assert_array_equal(f['observations/rgb_timestamp/'+c][:],f['observations/timestamp'][:])
        assert f['observations/proprio'].shape==(61,28)
        assert f['actions/effective'].shape==(480,14)
        for field in ['requested','effective']:
            assert np.isfinite(f['actions/'+field][:]).all()
            np.testing.assert_allclose(f['actions/'+field][:,[6,13]],.012,atol=1e-12)
        assert f['outcome/reward'][:].sum()==1 and f['outcome/reward'][-1]==1
        assert f['outcome/terminated'][:].sum()==1 and f['outcome/terminated'][-1]
        env=setup_env(cfg);e=env.physics;max_err=0.
        try:
            for tick,a in enumerate(f['actions/effective'][:]):
                np.testing.assert_allclose(e.proprio(),f['proprio/state'][tick],atol=1e-8,rtol=1e-8)
                previous=e.command.copy();e.step(a)
                assert np.max(abs(e.command-previous))<=3/240+1e-12
                saved=f['diagnostics/object_pose'][tick,:3]
                err=float(np.max(abs(e.data.xpos[e.model.body('parcel').id]-saved)));max_err=max(max_err,err)
                if tick%8==7:
                    np.testing.assert_allclose(e.proprio(),f['observations/proprio'][(tick+1)//8],atol=1e-8,rtol=1e-8)
            np.testing.assert_allclose(e.data.qpos,f['diagnostics/final_qpos'][:],atol=1e-8,rtol=1e-8)
            assert max_err<1e-8
        finally:env.close()
        if cfg['category']=='recovery':
            z=np.load(ROOT/'data/rejected'/f'paired_no_recovery_{idx:04d}.npz')
            cr=json.loads(str(z['result_json']));assert not cr['success']
            trigger=result['recovery_trigger_tick'];assert trigger is not None
            np.testing.assert_array_equal(z['requested'][:trigger],f['actions/requested'][:trigger])
            assert not np.allclose(z['requested'][trigger:],f['actions/requested'][trigger:])
        return dict(index=idx,category=cfg['category'],replay_max_position_error_m=max_err,
            first_stable_200ms_end_s=result['first_stable_200ms_end_s'],peak_gripper_force_N=result['peak_gripper_force_N'],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    paths=sorted((ROOT/'data/episodes').glob('episode_*.h5'));assert len(paths)==300
    with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(verify_one,paths))
    counts=dict(Counter(r['category'] for r in rows));assert counts==dict(regular=150,boundary=90,recovery=60)
    split={k:[] for k in ['train','validation','test']};rng=np.random.default_rng(810001)
    for cat,ids in [('regular',np.arange(150)),('boundary',np.arange(150,240)),('recovery',np.arange(240,300))]:
        rng.shuffle(ids);n=len(ids);a=int(.8*n);b=int(.9*n)
        for name,values in zip(split,[ids[:a],ids[a:b],ids[b:]]):split[name].extend(map(int,values))
    for values in split.values():values.sort()
    (ROOT/'data/splits.json').write_text(json.dumps(split,indent=2));(ROOT/'reports/splits.json').write_text(json.dumps(split,indent=2))
    assert len(set(sum(split.values(),[])))==300
    from dataset import FlipDataset
    for name in split:
        ds=FlipDataset(split=name)
        for i in [0,len(ds)-1]:
            sample=ds[i];assert sample['action'].shape==(8,14) and sample['image_overview'].shape==(2,3,96,96)
            assert set(sample)=={'image_overview','image_left','image_right','agent_pos','action','remaining_time_s','reward','terminated'}
    attempts=[]
    for p in (ROOT/'data/candidates').glob('*.json'):attempts.extend(json.loads(p.read_text())['attempts'])
    counter_reasons=Counter(json.loads(str(np.load(p)['result_json']))['reason'] for p in (ROOT/'data/rejected').glob('paired*.npz'))
    summary=dict(episodes=300,counts=counts,successful_replay=300,duration_s=2,physics_hz=240,camera_hz=30,
        camera_frames_per_episode=61,total_rgb_frames=54900,total_physics_transitions=144000,total_chunks=18000,
        bytes=sum(r['bytes'] for r in rows),max_replay_position_error_m=max(r['replay_max_position_error_m'] for r in rows),
        stable_completion_s=[min(r['first_stable_200ms_end_s'] for r in rows),max(r['first_stable_200ms_end_s'] for r in rows)],
        peak_force_N=[min(r['peak_gripper_force_N'] for r in rows),max(r['peak_gripper_force_N'] for r in rows)],
        split_counts={k:len(v) for k,v in split.items()},search_attempts=len(attempts),search_successes=sum(a['result']['success'] for a in attempts),
        paired_no_recovery_failure_reasons=dict(counter_reasons),
        limitations=['success-selected demonstration dataset, not a policy success-rate estimate','single trajectory family with state-conditioned control search; not human demonstrations','recovery subtype: measured under-rotation following reduced commanded turn progress; not slip/late-contact coverage','fixed SKU09 mass/friction/gain; high contact forces; no packaging damage model','all splits share procedural controller family; not an out-of-distribution benchmark'])
    (ROOT/'reports/verification.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'reports/episode_audit.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(summary,indent=2))
