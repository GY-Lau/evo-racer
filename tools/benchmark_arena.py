"""Fixed-work Jetson microbenchmark; never loads or writes the live championship."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from f1_config import circuit
parser=argparse.ArgumentParser();parser.add_argument('--baseline');parser.add_argument('--eager',action='store_true');parser.add_argument('--steps',type=int,default=512);args=parser.parse_args()
if args.baseline:
 spec=importlib.util.spec_from_file_location('arena_baseline',args.baseline);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
else:
 import grand_prix as module
assets=[]
for i in range(5):
 p,m,s,l=circuit(i);assets.append((torch.as_tensor(p,device='cuda'),torch.as_tensor(m,device='cuda'),s,l))
generator=torch.Generator(device='cuda').manual_seed(1234)
weights=torch.rand((704,112),generator=generator,device='cuda')*2-1
results=[]
for live in (704,176,22):
 arena=module.Arena(weights.clone(),0,1,assets,**({'optimized':False} if args.eager else {}))
 arena.reason[live:]=1
 # The same fixed number of steps and initial conditions for both execution paths.
 torch.cuda.synchronize();start=time.perf_counter()
 for _ in range(args.steps):
  arena.step()
  if arena.ticks%4==0 and arena.done():break
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 digest=hashlib.sha256(b''.join(getattr(arena,name).cpu().numpy().tobytes() for name in ('xy','angle','speed','best','reason','finish_tick'))).hexdigest()
 results.append(dict(stateDigest=digest,initialLive=live,steps=arena.ticks,seconds=round(elapsed,4),stepsPerSecond=round(arena.ticks/elapsed,1)))
print(json.dumps(dict(module=args.baseline or 'physics-v4',execution='eager' if args.eager else 'graph',results=results)))
