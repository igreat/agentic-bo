---
name: bo-write-report
description: Write clear, narrative reports (or sections) from Bayesian Optimization runs, including abstract, introduction, results, discussion, summary, and significance.
---

# BO Report Writing

Generate readable, insight-driven writing from Bayesian Optimization runs in `runs/<run_id>/`.

## Files to use
- `state.json` — problem setup and parameters  
- `report.json` — best result and summary metrics  
- `observations.jsonl` — optimization history  
- `convergence.pdf` — (optional) convergence plot  

## What to do

1. Understand the problem from `state.json`  
   - objective (min/max), parameters, bounds, optimizer  

2. Extract key results from `report.json`  
   - best value, best candidate, total evaluations  
   - oracle/proxy quality if available  

3. Analyse `observations.jsonl`  
   - how performance improved  
   - when best value was found  
   - convergence behaviour (fast, slow, plateau, unstable)

4. Optionally review `convergence.pdf` for visual insights.  

---

## Always include these sections

### Abstract
- What was optimized  
- Method used (BO + setup)  
- Best result  
- Key improvement or takeaway  

### Introduction
- What the problem is  
- Why it matters  
- Why Bayesian Optimization is suitable  

### Results
- Best value and candidate  
- Iteration where it was found  
- Improvement over initial performance  
- Convergence behaviour  

### Discussion
- What the results mean  
- How well the optimization worked  
- Any patterns or insights in parameters  
- Limitations (e.g. surrogate quality, few iterations)  

### Why it is important
- Real-world or research relevance  
- What this enables  
- What should be done next  
