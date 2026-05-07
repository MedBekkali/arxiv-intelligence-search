"""Quick sanity check: what are paper #0's nearest neighbors?"""
import pandas as pd

meta = pd.read_parquet('models/v3_papers_meta.parquet')

print('Query paper #0:')
print(f'  Category: {meta.iloc[0]["first_cat"]}')
print(f'  Title:    {meta.iloc[0]["title"]}')
print()
print('Top neighbors found by FAISS:')
for i in [902018, 477507, 221833, 161829]:
    print(f'  [{i:>6}] {meta.iloc[i]["first_cat"]:8s}  {meta.iloc[i]["title"]}')