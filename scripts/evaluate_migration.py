from pathlib import Path
import subprocess,os,json,time,math,hashlib
ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('BC_PYTHON','/data02/kemove/sku09-bc-migration/env/bin/python')
OUT=ROOT/'reports/closed_loop';OUT.mkdir(parents=True,exist_ok=False)
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2))
write('protocol.json',dict(checkpoint='models/bc_epoch252.pt',sampler='upstream',eta=1,ddim_steps=10,gpus=[1,2,3,6],cohorts={'reproduction':list(range(880000,880100)),'fresh_holdout':list(range(910000,910100))},training=False,final900000_seeds_used=False,action_hz=240,camera_hz=30,duration_s=2,position_half_range_m=.005,yaw_half_range_deg=2))
started=time.time();allrows={}
try:
 for cohort,start in [('reproduction',880000),('fresh_holdout',910000)]:
  jobs=[]
  for shard,gpu in enumerate([1,2,3,6]):
   dest=OUT/f'{cohort}_{shard}.json';log=open(OUT/f'{cohort}_{shard}.log','w')
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYTHONPATH=str(ROOT/'src'))
   cmd=[PYTHON,'-u',str(ROOT/'src/evaluate_bc.py'),'--checkpoint',str(ROOT/'models/bc_epoch252.pt'),'--output',str(dest),'--seed-start',str(start+25*shard),'--episodes','25','--sampler','upstream']
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log,dest))
  write('status.json',dict(stage=cohort,pids=[p.pid for p,_,_ in jobs],started=started))
  for p,log,dest in jobs:
   code=p.wait();log.close();assert code==0,(p.pid,code,str(dest))
  rows=[json.loads(s) for _,_,d in jobs for s in d.with_suffix('.episodes.jsonl').read_text().splitlines()]
  assert len(rows)==100 and len({r['seed'] for r in rows})==100
  allrows[cohort]=rows
 results={}
 for cohort,rows in allrows.items():
  k=sum(r['success'] for r in rows);n=len(rows);z=1.95996398454;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
  latency=sorted(v for r in rows for v in r['inference_latency_s'])
  results[cohort]=dict(successes=k,episodes=n,success_rate=p,wilson95=[c-h,c+h],failure_counts={reason:sum(r['reason']==reason for r in rows) for reason in ['drop','wrong_face','unsettled']},failed_seeds=[r['seed'] for r in rows if not r['success']],inference_latency_median_s=latency[len(latency)//2],inference_latency_p95_s=latency[int(.95*(len(latency)-1))])
 old={r['seed']:r for r in map(json.loads,(ROOT/'reports/source_baseline/episodes.jsonl').read_text().splitlines())}
 differences=[r['seed'] for r in allrows['reproduction'] if r['success']!=old[r['seed']]['success'] or r['reason']!=old[r['seed']]['reason']]
 report=dict(complete=True,root=str(ROOT),checkpoint_sha256=hashlib.sha256((ROOT/'models/bc_epoch252.pt').read_bytes()).hexdigest(),results=results,reproduction_outcome_mismatches=differences,elapsed_s=time.time()-started)
 write('report.json',report);write('status.json',dict(stage='complete',elapsed_s=report['elapsed_s']));print(json.dumps(report),flush=True)
except Exception as e:
 write('status.json',dict(stage='failed',error=repr(e)));raise
