import os
import openai
import json
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = '''You are given target phrases and need to determine if they are related to the anchor phrase. The phrases come from US Patents. You can answer 'similar', 'not similar', or 'unsure'.'''
phrase_map = {'similar':1,'not similar':0, 'unsure':.5}
df = pd.read_csv("train.csv")

def checkPhrase(one, two):
    return openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Anchor: {one}\n Target: {two}"},
        ]
    )

def scoreCompletion(completion):
        return phrase_map[completion.choices[0].message.content]


def compareScores(completion_score,real_score):
    if completion_score is None:
        return 1
    if abs(completion_score-real_score) <= 0.25:
        return 0
    else:
        return abs(completion_score-real_score)


count = 0
loss = 0

for i in range(len(df)):
    completion = checkPhrase(df.loc[i,'anchor'],df.loc[i,'target'])
    print(completion.choices[0].message.content)
    loss = loss + compareScores(scoreCompletion(completion),df.loc[i,'score'])
    count = count+1
    if count>5:
        break

print(loss)