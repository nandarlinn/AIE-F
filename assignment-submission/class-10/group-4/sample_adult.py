#This creates a 400-row balanced subset of the original adult dataset for faster experimentation with SCRDR interactive learning.

import pandas as pd

df = pd.read_csv('./data/adult.csv')
print(f"Original: {len(df)} rows")
print(df['income'].value_counts())

# Take 200 of each class for a balanced 400-row sample
sample = pd.concat([
    df[df['income'] == '<=50K'].sample(n=200, random_state=42),
    df[df['income'] == '>50K'].sample(n=200, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

sample.to_csv('./data/adult_small.csv', index=False)
print(f"Sampled: {len(sample)} rows")
print(sample['income'].value_counts())