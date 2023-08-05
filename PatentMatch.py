import os
import openai
import json
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = '''You are given target phrases and need to determine if they are related to the anchor phrase. The phrases come from US Patents. You can answer 'similar', 'not similar', or 'unsure'. Tune your responses so they are are equally likely'''
phrase_map = {'similar':1,'not similar':0, 'unsure':.5}
round_max_loss = {0:1,.25:.75,.5:.5,.75:.75,1:1}
df = pd.read_csv("train.csv")

def checkPhrase(one, two):
    return openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Anchor: {one}\n Target: {two}"},
        ],
      temperature=0
    )

def scoreCompletion(completion):
        return phrase_map[completion.choices[0].message.content.lower()]


def compareScores(completion_score,real_score):
    if completion_score is None:
        return 1
    if abs(completion_score-real_score) <= 0.25:
        return 0
    else:
        return abs(completion_score-real_score)


count = 0
loss = 0
max_loss = 0
print(df.head(6))
for i in range(len(df)):
    completion = checkPhrase(df.loc[i,'anchor'],df.loc[i,'target'])
    print(completion.choices[0].message.content)
    loss = loss + compareScores(scoreCompletion(completion),df.loc[i,'score'])
    max_loss = max_loss + round_max_loss[df.loc[i,'score']]
    count = count+1
    if count>50:
        break

print(1-loss/max_loss)
print(df.loc[:, 'score'].mean())