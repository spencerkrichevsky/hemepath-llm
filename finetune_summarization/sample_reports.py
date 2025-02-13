import pandas as pd

df = pd.read_csv('./data/annotations/annotations.csv')
df = df.replace(to_replace='msk', value='real', regex=True)
df = df.replace(to_replace='nguyen', value='real', regex=True)

# Sample N=10 rows per group of data
seed = 123
sampled_df = df.groupby('subset', group_keys=False).sample(n=10, random_state=seed)
remaining_df = df.drop(sampled_df.index)
sampled_df = sampled_df.reset_index(drop=True)
remaining_df = remaining_df.reset_index(drop=True)
sampled_df.to_csv('./data/annotations/sampled_annotations.csv')
remaining_df.to_csv('./data/annotations/heldout_annotations.csv')
