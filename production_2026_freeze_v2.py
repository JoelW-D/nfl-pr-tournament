#!/usr/bin/env python3
"""Week 1 baseline bridge QA correction.

Executes the frozen 2026 production builder with two roster-bridge parsing fixes:
1) nflverse defensive groups are named like 'Base 3-4 D'/'Base 4-3 D', not 'defense'.
2) front-seven position abbreviations include LDE/RDE/LDT/RDT and scheme LB labels.
No F1 architecture, fit procedure, coefficient, state mechanic, or probability
calibration is changed by this wrapper.
"""
from pathlib import Path

src_path = Path(__file__).with_name("production_2026_freeze.py")
src = src_path.read_text()
old_def = 'defense = g[g.pos_grp_l.str.contains("def", na=False)]'
new_def = 'defense = g[g.pos_grp_l.str.contains(r"base .* d", regex=True, na=False)]'
old_front = 'front = defense[defense.pos_abb_u.isin(["DE","DT","NT","DL","EDGE","OLB","LB","LOLB","ROLB"])]'
new_front = 'front = defense[defense.pos_abb_u.isin(["LDE","RDE","DE","EDGE","LDT","RDT","DT","NT","DL","WLB","SLB","MLB","ILB","OLB","LOLB","ROLB","LB"])]'
if old_def not in src or old_front not in src:
    raise RuntimeError("Expected Week 1 bridge code not found; refuse silent drift")
src = src.replace(old_def, new_def).replace(old_front, new_front)
exec(compile(src, str(src_path), "exec"), {"__name__":"__main__", "__file__":str(src_path)})
