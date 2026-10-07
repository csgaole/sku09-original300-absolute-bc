# SKU09 original-300 absolute-joint BC

Independent server-local Git repository containing the MuJoCo sorting simulation, 300 original scripted expert demonstrations, and the epoch-252 BC EMA policy. No V8/V14 RL checkpoint is used.

## Contents
- `vendor/logistics-sorting-sim`: complete simulation source and assets.
- `vendor/rl100`: vendored RL-100 components used by the BC host.
- `data/episodes`: 300 HDF5 episodes; `data/splits.json`: 240 train / 30 validation / 30 test.
- `models/bc_epoch252.pt`: frozen original checkpoint; SHA-256 c899a348be831018b1903399e2ea858e130202f7bc2038408f8606e3387d908a.
- `src`: policy loading, action semantics, closed-loop evaluator and demonstration tools.
- `migration_manifest.json`: copied payload hashes and source provenance.
- `requirements-lock.txt`: actual source environment package versions.
- `provenance`: original training source retained for reference; cached training inputs and a training-resume checkpoint are not part of this evaluation repository.

Binary payloads are physically present and independently copied in this checkout. Git excludes HDF5 and model binaries; a plain Git clone alone does not transport these payloads. Transfer the complete repository directory and verify the manifest. No remote repository has been published.

## Runtime
Current verified interpreter: `/data02/kemove/sku09-bc-migration/env/bin/python`. Source, simulation assets, demonstrations and checkpoint resolve within this repository; the Python environment is shared. For a separate machine, install `requirements-lock.txt` in a compatible CUDA environment.

## Closed-loop evaluation
```sh
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl /data02/kemove/sku09-bc-migration/env/bin/python src/evaluate_bc.py --checkpoint models/bc_epoch252.pt --output reports/manual_eval.json --seed-start 910000 --episodes 100 --sampler upstream
```
`python scripts/evaluate_migration.py` orchestrates four GPU shards on GPUs 1,2,3,6; refuses to overwrite `reports/closed_loop`. It runs 100 historical development conditions (880000–880099), followed by 100 fresh conditions (910000–910099). Reserved final seeds 900000–900999 are untouched. Fresh conditions share the original narrow initialization distribution; they are not an out-of-distribution robustness test.

Policy observations: three RGB cameras and measured 28-dimensional proprioception, two observation frames. Policy outputs: 12 absolute joint targets and two fixed 0.012 m gripper slots. No expert policy or parcel privileged state supplies inference actions. Evaluation uses 10-step upstream DDIM, eta=1, FP32 without TF32. Physics control is 240 Hz, camera updates 30 Hz, simulated duration 2 s.

Success requires label-up >0.98, parcel bottom >0.79 m, linear speed <0.03 m/s, angular speed <0.15 rad/s for the final 200 ms, and no bottom <0.74 m during the episode. Wall-clock inference latency is recorded; simulation correctness does not imply real-time deployability.

Source historical result: 98/100 with this exact checkpoint and sampler. New migration results appear in `reports/closed_loop/report.json`.
