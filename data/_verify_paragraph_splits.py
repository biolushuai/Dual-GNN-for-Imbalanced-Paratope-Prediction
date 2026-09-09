#!/usr/bin/env python
"""精确对齐 paragraph 与 paratope128 的 PDB ID"""
import pickle

print("=" * 60)
print("Detailed PDB ID format inspection")
print("=" * 60)

# paratope128 test
with open("/mnt/d/ThesisCode/Datasets/paratope128/test.pkl", "rb") as f:
    d = pickle.load(f)
print(f"\nparatope128/test.pkl: d[0] first 10: {d[0][:10]}")

# paragraph test
with open("/mnt/d/ProjectsData/ParaLoRADG/pecan-paratope-test-all-paragraph.pkl", "rb") as f:
    d2 = pickle.load(f)
para_names = [s["protein_name"] for s in d2]
print(f"paragraph/test.pkl: protein_name first 10: {para_names[:10]}")

# Both use full protein_name format, compare directly
para_set = set(para_names)
para128_set = set(d[0])
common = para_set & para128_set
print(f"\nFull protein_name comparison: para={len(para_set)} para128={len(para128_set)} common={len(common)}")
print(f"only paragraph: {list(para_set - para128_set)[:5]}")
print(f"only paratope128: {list(para128_set - para_set)[:5]}")

# All splits check
print("\n" + "=" * 60)
print("All splits alignment")
print("=" * 60)
for split in ["train", "val", "test"]:
    with open(f"/mnt/d/ProjectsData/ParaLoRADG/pecan-paratope-{split}-all-paragraph.pkl", "rb") as f:
        dp = pickle.load(f)
    with open(f"/mnt/d/ThesisCode/Datasets/paratope128/{split}.pkl", "rb") as f:
        d128 = pickle.load(f)
    para_set = set([s["protein_name"] for s in dp])
    para128_set = set(d128[0])
    common = para_set & para128_set
    only_para = para_set - para128_set
    only_128 = para128_set - para_set
    print(f"\n{split}: paragraph={len(para_set)} paratope128={len(para128_set)} common={len(common)}")
    print(f"  only paragraph ({len(only_para)}): {sorted(only_para)[:5]}")
    print(f"  only paratope128 ({len(only_128)}): {sorted(only_128)[:5]}")