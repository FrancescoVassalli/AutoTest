import os
import openai
import json
import pandas as pd
import random
import numpy as np

system_prompt = '''You are given examples pairs of phrases from US patents with a score indicating how similar they are.
The user will then give new phrases and you will answer with what the similarity score should be. In the format 'Score: 0.25'. Only return 0,0.25,0.5,0.75, 1.0 as options,
Examples:
Anchor: prolog; Target: prolog signal; Score: 1.0
Anchor: smooth outer surface; Target: rectangular cylindrical surface; Score: 0.25
Anchor: morpholin; Target: base; Score: 0.25
Anchor: displacement mechanism; Target: displacement member; Score: 0.5
Anchor: seal members; Target: sealing member; Score: 1.0
Anchor: dual clutch; Target: clutch system; Score: 0.5
Anchor: hexahydro; Target: hexahydro s triazine; Score: 0.5
Anchor: summits; Target: support system for vehicle bodies; Score: 0.25
Anchor: sheet supply roller; Target: sheet roller; Score: 0.5
Anchor: metallic cartridges; Target: case file; Score: 0.0
Anchor: gas leak; Target: liquid leak; Score: 0.25
Anchor: component composite coating; Target: phase polymer; Score: 0.5
Anchor: vertical chute; Target: inclined plane through which objects are moved; Score: 0.75
Anchor: insulation sleeve; Target: insulating tube; Score: 0.75
Anchor: brush guard; Target: encapsulated electrical igniter; Score: 0.25
Anchor: imaging axis; Target: optical of direction; Score: 0.0'''

ideal_mapping = {0.0:['Score: 0.0','Score: 0.25'],
                 0.25:['Score: 0.0', 'Score: 0.5','Score: 0.25'],
                 0.5:['Score: 0.25','Score: 0.5','Score: 0.75'],
                 0.75:['Score: 0.5','Score: 0.75','Score: 1.0'],
                 1.0: ['Score: 0.75','Score: 1.0']}

def create_chat_prompt(one, two):
    return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Anchor: {one}\n Target: {two}"},
        ]

df = pd.read_csv("train.csv")
input = [0] *len(df)
ideal = [0] *len(df)

for i in range(len(df)):
    input[i] =create_chat_prompt(df.loc[i,'anchor'],df.loc[i,'target'])
    ideal[i] = ideal_mapping[df.loc[i,'score']]

df = pd.DataFrame({'input':input,'ideal':ideal})
df.to_json("samples.jsonl",orient='records',lines=True)
eval_yaml = '''patent_phrases:
    id: patent_phrases.dev.v0
    description: Eval for gpt performance on US Patent Phrase Matching 
    metrics: [accuracy]
patent_phrases.dev.v0:
    class: evals.elsuite.basic.match:Match
    args: 
        samples_jsonl: patent_phrases/samples.jsonl
'''
with open("PatentPhrases.yaml", "w") as f:
    f.write(eval_yaml)