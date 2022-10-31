import pandas as pd
import json
from tqdm import tqdm
from glob import glob

def proc_json(object):
    standards = []
    dialects = []
    utterance = object.get('utterance')
    for item in utterance:
        standard = item.get('standard_form')
        dialect = item.get('dialect_form')

        if standard == dialect:
            continue
        else:
            standards.append([standard, 0])
            dialects.append([dialect, 1])
    return standards, dialects

arr_df = []

files = glob('./json/*.json')

for file in tqdm(files):
    open_file = open(file, "r", encoding="utf-8")
    object = json.load(open_file)
    std, dial = proc_json(object)
    arr_df += std
    arr_df += dial
    open_file.close()

df = pd.DataFrame(arr_df, columns=["contents", "label"])
df.to_csv("./result/result.csv", index=False)
