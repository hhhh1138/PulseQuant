#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from diffusers import UniPCMultistepScheduler


SUFFIXES=("attn1.to_q","attn1.to_k","attn1.to_v","attn1.to_out.0","attn2.to_q","attn2.to_k","attn2.to_v","attn2.to_out.0","ffn.net.0.proj","ffn.net.2")
p=argparse.ArgumentParser()
p.add_argument("--analysis",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--blocks",type=int,default=40)
p.add_argument("--inference-steps",type=int,default=40)
p.add_argument("--calibration-steps",type=int,default=6)
p.add_argument("--flow-shift",type=float,default=12.0)
p.add_argument("--tail-fraction",type=float,default=.25)
p.add_argument("--time-weight-gamma",type=float,default=.25)
a=p.parse_args()
rows=json.loads(a.analysis.read_text())["runs"]

def schedule(steps):
    s=UniPCMultistepScheduler(prediction_type="flow_prediction",use_flow_sigmas=True,num_train_timesteps=1000,flow_shift=a.flow_shift)
    s.set_timesteps(steps)
    return np.asarray([float(x) for x in s.timesteps]),np.asarray([float(x) for x in s.sigmas[:steps]])

infer_t,infer_s=schedule(a.inference_steps)
cal_t,cal_s=schedule(a.calibration_steps)
result_rows=[]
expert_summary={}
for expert,mask in (("high_noise",infer_t>=875),("low_noise",infer_t<875)):
    selected=[r for r in rows if r["expert"]==expert]
    if not selected: raise ValueError(f"no {expert} pulse rows")
    by_step=defaultdict(list)
    for row in selected: by_step[int(row["injection_step"])].append(row)
    normalized=defaultdict(list)
    for step,group in by_step.items():
        raw=np.asarray([float(r["finite_difference_jvp_gain"]) for r in group])
        scale=max(float(np.median(raw)),1e-20)
        for row,value in zip(group,raw/scale): normalized[int(row["block"])].append(float(value))
    observed=np.asarray(sorted(normalized),dtype=np.int64)
    observed_risk=np.asarray([np.mean(sorted(normalized[int(b)],reverse=True)[:max(1,math.ceil(len(normalized[int(b)])*a.tail_fraction))]) for b in observed])
    blocks=np.arange(a.blocks,dtype=np.int64)
    block_risk=np.interp(blocks,observed,observed_risk)
    raw_by_block=defaultdict(list)
    for row in selected: raw_by_block[int(row["block"])].append((int(row["injection_step"]),float(row["finite_difference_jvp_gain"])))
    observed_weights={}
    local_cal_indices=np.flatnonzero(cal_t>=875 if expert=="high_noise" else cal_t<875)
    for block,values in raw_by_block.items():
        ordered=sorted(values)
        anchor=np.asarray([x[0] for x in ordered],dtype=np.int64)
        log_risk=np.log(np.asarray([x[1] for x in ordered])+1e-20)
        weights=np.exp(np.interp(cal_s[local_cal_indices][::-1],infer_s[anchor][::-1],log_risk[::-1])[::-1])
        weights=np.power(weights,a.time_weight_gamma)
        weights/=max(float(weights.mean()),1e-20); weights=np.clip(weights,.25,4); weights/=float(weights.mean())
        observed_weights[block]=[float(x) for w in weights for x in (w,w)]
    def weights_for(block):
        if block in observed_weights:return observed_weights[block]
        left=max([int(x) for x in observed if x<block],default=int(observed[0])); right=min([int(x) for x in observed if x>block],default=int(observed[-1]))
        if left==right:return observed_weights[left]
        ratio=(block-left)/(right-left)
        return [float((1-ratio)*x+ratio*y) for x,y in zip(observed_weights[left],observed_weights[right])]
    mean=max(float(block_risk.mean()),1e-20)
    for block in blocks:
        for suffix in SUFFIXES:
            result_rows.append({"module":f"{expert}.blocks.{int(block)}.{suffix}","weight_bits":4,"activation_bits":4,"propagation_state_weights":weights_for(int(block)),"absolute_propagation_risk":float(block_risk[int(block)]/mean)})
    expert_summary[expert]={"observed_blocks":[int(x) for x in observed],"pulse_steps":sorted(by_step),"calibration_step_indices":[int(x) for x in local_cal_indices],"profile_modules":a.blocks*len(SUFFIXES)}
result={"schema_version":2,"method":"pulseprofile","uses_evaluation_scores":False,"model":"Wan2.2-T2V-A14B","calibration_source":str(a.analysis),"risk":"anchor-normalized CVaR of directional finite-difference propagation gain","risk_mode":"gain","tail_fraction":a.tail_fraction,"time_weight_gamma":a.time_weight_gamma,"scheduler":{"flow_shift":a.flow_shift,"inference_steps":a.inference_steps,"calibration_steps":a.calibration_steps,"boundary_ratio":.875},"experts":expert_summary,"modules":result_rows}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"modules":len(result_rows),"experts":expert_summary},indent=2))
