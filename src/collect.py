"""State-conditioned scripted demonstrations, with strict 2 s validation.

No object forces or teleportation after reset. Privileged expert signals are
stored only in diagnostics. Import this module with MUJOCO_GL=egl for rendering.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor/logistics-sorting-sim/src'))
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
A = np.load(ROOT/'assets/reference_actions.npy')
RAW = json.loads((ROOT/'assets/reference_states.json').read_text())
ANGLES = np.interp(np.arange(480), np.arange(7,480,8), np.arccos(-np.clip([s['label_up'] for s in RAW],-1,1)))
CAMS = ['global','left_wrist','right_wrist']

def make_maps():
    out=ROOT/'assets/tcp_jacobian_maps.npy'
    if out.exists(): return np.load(out)
    maps=np.zeros((480,2,6,3))
    with LogisticsSortingEnv(render=False) as env:
        e=env.physics;m=e.model;d=mujoco.MjData(m)
        for k,a in enumerate(A):
            for arm in range(2):
                d.qpos[e.qidx[arm]]=a[7*arm:7*arm+6];mujoco.mj_forward(m,d)
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacSite(m,d,jp,jr,m.site(f'arm{arm}_tcp').id)
                J=np.vstack([jp[:,e.vidx[arm]],jr[:,e.vidx[arm]]])
                maps[k,arm]=np.linalg.solve(J.T@J+1e-6*np.eye(6),J.T)[:,:3]
    np.save(out,maps);return maps

def setup_env(cfg):
    env=LogisticsSortingEnv(render=False);env.reset(seed=0)
    e=env.physics;m,d=e.model,e.data
    adr=m.jnt_qposadr[m.joint('parcel_free').id]
    # Undo wrapper jitter: domain offsets refer to the XML nominal table pose.
    jitter=np.random.default_rng(0).uniform(-.005,.005,2)
    d.qpos[adr:adr+2]-=jitter
    d.qpos[adr:adr+2]+=np.array([cfg['dx'],.05+cfg['dy']])
    yaw=np.pi/2+np.deg2rad(cfg['yaw_deg'])
    d.qpos[adr+3:adr+7]=[0,np.cos(yaw/2),np.sin(yaw/2),0]
    m.actuator_gainprm[e.aidx.ravel(),0]=1200
    m.actuator_biasprm[e.aidx.ravel(),1]=-1200
    mujoco.mj_forward(m,d)
    return env

def simulate(cfg, capture=False, recovery_enabled=True, record=True):
    maps=make_maps();env=setup_env(cfg);e=env.physics;m,d=e.model,e.data
    renderer=mujoco.Renderer(m,height=240,width=320) if capture else None
    phase=0.;stable=[];peak=0.;dropped=False;trigger_tick=None
    logs={k:[] for k in ['requested','effective','proprio','qpos','qvel','ctrl','object_pose','object_velocity','contact_force','bottom','label_up','phase','stable','recovery_active']}
    rgb={c:[] for c in CAMS};obs_proprio=[]
    config=dict(cfg, initial_qpos=d.qpos.tolist(), initial_qvel=d.qvel.tolist(),
        initial_ctrl=d.ctrl.tolist(), parcel_mass=float(m.body_mass[m.body('parcel').id]),
        parcel_inertia=m.body_inertia[m.body('parcel').id].tolist(),
        geom_friction=m.geom_friction.tolist(), geom_names=[m.geom(i).name for i in range(m.ngeom)],
        gain=1200,physics_hz=240,action_hz=240,camera_hz=30,image_shape=[240,320,3],
        action_semantics='absolute_joint_position_targets_rad; columns 6,13 fixed_width_m=0.012',
        requested_to_effective='joint limit clipping then 3 rad/s target slew limit',
        task='SKU09_long_edge_flip_2s_not_full_sorting',expert_uses_privileged_state=True)
    def snapshot():
        obs_proprio.append(e.proprio().copy())
        if renderer:
            for c in CAMS:
                renderer.update_scene(d,camera=c);rgb[c].append(renderer.render().copy())
    snapshot()
    try:
        for tick in range(480):
            state=task_state(e);angle=float(np.arccos(-np.clip(state['label_up'],-1,1)))
            if cfg['category']=='recovery' and recovery_enabled and trigger_tick is None:
                # A measured deficit relative to the nominal turning angle.
                if 120<=tick<=250 and ANGLES[min(tick,479)]-angle>cfg['trigger_deficit_rad']:
                    trigger_tick=tick
            active=trigger_tick is not None
            i=min(478,int(phase));f=min(1,phase-i);a=A[i]*(1-f)+A[i+1]*f
            # Independent support timing plus state-conditioned nominal TCP shift.
            sp=np.clip(phase+cfg['support_phase_ticks']*np.sin(np.pi*min(phase,288)/288),0,479)
            si=min(478,int(sp));sf=min(1,sp-si);a[7:13]=A[si,7:13]*(1-sf)+A[si+1,7:13]*sf
            for arm in range(2):
                delta=np.array([cfg['tracking_fraction']*cfg['dx'],cfg['tracking_fraction']*cfg['dy'],cfg['height_m']])*min(1,tick/48)
                a[7*arm:7*arm+6]+=maps[i,arm]@delta
            if record:
                logs['proprio'].append(e.proprio().copy());logs['qpos'].append(d.qpos.copy());logs['qvel'].append(d.qvel.copy())
            e.step(a);s=task_state(e);v=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('parcel').id,v,0)
            ok=s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15
            stable.append(bool(ok));peak=max(peak,max(s['forces']));dropped|=s['bottom']<.74
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise AssertionError('External applied force')
            if record:
                vals=dict(requested=a.copy(),effective=e.history[-1][2],ctrl=d.ctrl.copy(),object_pose=np.r_[d.xpos[m.body('parcel').id],d.xquat[m.body('parcel').id]],object_velocity=v,contact_force=s['forces'],bottom=s['bottom'],label_up=s['label_up'],phase=phase,stable=ok,recovery_active=active)
                for k,val in vals.items():logs[k].append(val)
            if (tick+1)%8==0:snapshot()
            rate=cfg['approach_rate'] if phase<65 else cfg['turn_rate'] if phase<235 else cfg['exit_rate']
            if cfg['category']=='recovery' and 65<phase<230:
                rate*=cfg['disturbance_rate']
                if active:rate*=cfg['recovery_multiplier']
            elif 65<phase<270:
                rate*=np.clip(1+cfg['angle_feedback']*(angle-ANGLES[i]),.7,1.3)
            phase=min(479,phase+rate)
        times=[(j+48)/240 for j in range(433) if all(stable[j:j+48])]
        success=all(stable[-48:]) and not dropped
        summary=dict(success=bool(success),reason='success_2s' if success else 'drop' if dropped else 'wrong_face_or_unsettled_at_2s',
            peak_gripper_force_N=float(peak),first_stable_200ms_end_s=times[0] if times else None,
            final_label_up=float(s['label_up']),final_bottom_m=float(s['bottom']),
            final_linear_speed=float(np.linalg.norm(v[3:])),final_angular_speed=float(np.linalg.norm(v[:3])),
            recovery_trigger_tick=trigger_tick,external_object_wrench=False)
        if record:
            logs={k:np.asarray(v) for k,v in logs.items()};logs['final_qpos']=d.qpos.copy();logs['final_qvel']=d.qvel.copy()
        return summary,logs,rgb,np.asarray(obs_proprio),config
    finally:
        if renderer:renderer.close()
        env.close()

def proposal(rng, category, index, attempt):
    # Initial-state sequence independent from control search, recorded in full.
    edge=category=='boundary'
    initial_rng=np.random.default_rng(1710000+index)
    xy=initial_rng.uniform(-.005,.005,2)
    if edge:xy[initial_rng.integers(2)]=initial_rng.choice([-1,1])*initial_rng.uniform(.004,.005)
    yaw=initial_rng.uniform(-2,2)
    return dict(category=category,index=index,attempt=attempt,dx=float(xy[0]),dy=float(xy[1]),yaw_deg=float(yaw),
        approach_rate=float(rng.uniform(.96,1.05)),turn_rate=float(rng.uniform(.94,1.08)),exit_rate=float(rng.uniform(.97,1.05)),
        support_phase_ticks=float(rng.uniform(-3,3) if not edge else rng.choice([-1,1])*rng.uniform(3,6)),
        tracking_fraction=float(rng.uniform(0,.2)),height_m=float(rng.uniform(-.0005,.0005) if not edge else rng.uniform(-.0015,.0015)),
        angle_feedback=float(rng.uniform(.05,.35)),disturbance_rate=float(rng.uniform(.58,.83)),
        trigger_deficit_rad=float(rng.uniform(.2,.5)),recovery_multiplier=float(rng.uniform(1.35,1.9)),
        seed=710000+index)

def find_episode(index,max_attempts=100):
    category='regular' if index<150 else 'boundary' if index<240 else 'recovery'
    rng=np.random.default_rng(710000+index);attempts=[];best=None
    for attempt in range(max_attempts):
        cfg=proposal(rng,category,index,attempt)
        result,_,_,_,_=simulate(cfg,record=False)
        row=dict(config=cfg,result=result);attempts.append(row)
        if not result['success'] or result['peak_gripper_force_N']>350:continue
        if category=='recovery':
            if result['recovery_trigger_tick'] is None:continue
            counter,_,_,_,_=simulate(cfg,recovery_enabled=False,record=False)
            row['counterfactual']=counter
            if counter['success']:continue
        # First valid state-conditioned candidate. Not a uniform success estimate.
        best=row;break
    (ROOT/'data/candidates'/f'{index:04d}.json').write_text(json.dumps(dict(index=index,category=category,attempts=attempts,selected=best),indent=2))
    return dict(index=index,category=category,attempts=len(attempts),selected=best is not None,result=best['result'] if best else None)

def save_episode(index):
    import h5py
    row=json.loads((ROOT/'data/candidates'/f'{index:04d}.json').read_text())['selected']
    if not row:raise RuntimeError(f'No qualifying episode {index}')
    result,logs,rgb,proprio,cfg=simulate(row['config'],capture=True)
    assert result['success'] and abs(result['peak_gripper_force_N']-row['result']['peak_gripper_force_N'])<1e-7
    dest=ROOT/'data/episodes'/f'episode_{index:04d}.h5'
    with h5py.File(str(dest)+'.tmp','w') as f:
        f.attrs['schema_version']='1.0';f.attrs['config_json']=json.dumps(cfg);f.attrs['result_json']=json.dumps(result)
        f.attrs['provenance']='privileged_state_conditioned_script_search; reference trajectory family; not human teleoperation'
        f.attrs['category']=cfg['category'];f.attrs['episode_id']=index
        f.create_dataset('observations/timestamp',data=np.arange(61)/30)
        f.create_dataset('observations/proprio',data=proprio)
        for camera in CAMS:
            f.create_dataset('observations/rgb/'+camera,data=np.asarray(rgb[camera]),compression='gzip',compression_opts=1,chunks=(1,240,320,3))
            f.create_dataset('observations/rgb_timestamp/'+camera,data=np.arange(61)/30)
        f.create_dataset('actions/timestamp',data=np.arange(480)/240)
        f.create_dataset('actions/requested',data=logs.pop('requested'))
        f.create_dataset('actions/effective',data=logs.pop('effective'))
        f.create_dataset('actions/actuator_ctrl',data=logs.pop('ctrl'))
        f.create_dataset('proprio/timestamp',data=np.arange(480)/240)
        f.create_dataset('proprio/state',data=logs.pop('proprio'))
        reward=np.zeros(480,np.float32);reward[-1]=1
        done=np.zeros(480,np.bool_);done[-1]=True
        f.create_dataset('outcome/reward',data=reward);f.create_dataset('outcome/terminated',data=done)
        f.create_dataset('diagnostics/post_step_timestamp',data=np.arange(1,481)/240)
        f.create_dataset('diagnostics/pre_step_timestamp',data=np.arange(480)/240)
        for k,v in logs.items():f.create_dataset('diagnostics/'+k,data=v,compression='gzip' if np.asarray(v).ndim else None)
        if 'counterfactual' in row:f.attrs['no_recovery_counterfactual_json']=json.dumps(row['counterfactual'])
    os.replace(str(dest)+'.tmp',dest)
    if cfg['category']=='recovery':
        cr,cl,_,_,cc=simulate(row['config'],recovery_enabled=False)
        assert not cr['success']
        np.savez_compressed(ROOT/'data/rejected'/f'paired_no_recovery_{index:04d}.npz',**cl,config_json=json.dumps(cc),result_json=json.dumps(cr))
    return dict(index=index,path=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size,category=cfg['category'],result=result,config=row['config'])

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor,as_completed
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['search','render']);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=300);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    make_maps();fn=find_episode if args.mode=='search' else save_episode
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fn,i) for i in range(args.start,args.stop)]
        for fut in as_completed(futures):
            row=fut.result();print(json.dumps(row),flush=True)
            if args.mode=='render':
                with open(ROOT/'data/manifest.jsonl','a') as f:f.write(json.dumps(row)+'\n')
