import pandas as pd
import json
from tqdm import tqdm
from glob import glob
import re

def proc_json(object, pattern):
    standards = []
    dialects = []
    utterance = object.get('utterance')
    for item in utterance:
        form = item.get('form')
        form = re.sub('\*|\#|(\(\(\)\))', '', form)

        match_result = pattern.findall(form)

        if len(match_result) == 0:
            continue
        else:
            standard = form
            dialect = form

            flag = False

            for term in match_result:
                terms = term.split(')/(')
                term_std = terms[1].replace(')', '')
                term_dial = terms[0].replace('(', '')

                if not flag:
                    if term_std != term_dial:
                        flag = True

                standard = standard.replace(term, term_std)
                dialect = dialect.replace(term, term_dial)

            if flag:
                standards.append([standard, 0])
                dialects.append([dialect, 3])

                #print(standard)
                #print(dialect)
                #input()
    return standards, dialects

arr_df = []

files = glob('./json/*.json')

pattern = re.compile("\([가-힣 .!?]*\)/\([가-힣 .!?]*\)")

for file in tqdm(files):
    open_file = open(file, "r", encoding="utf-8-sig")
    object = json.load(open_file)
    std, dial = proc_json(object, pattern)
    arr_df += std
    arr_df += dial
    open_file.close()

df = pd.DataFrame(arr_df, columns=["contents", "label"])
df.to_csv("./result/result.csv", index=False)
