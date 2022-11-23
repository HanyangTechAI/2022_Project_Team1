import pandas as pd
from tqdm import tqdm
from glob import glob

df = pd.DataFrame(columns=["contents", "label"])

files = glob('./*.csv')

for file in tqdm(files):
    file_df = pd.read_csv(file)
    df = pd.concat([df,file_df], axis=0, ignore_index=True)

df.to_csv("./result.csv", index=False)
