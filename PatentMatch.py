import os
import openai
import json
import pandas as pd
import random
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = '''The following is an example of pairs of phrases from US patents with a score indicating how similar they are. 
The user will then give new phrases and you will answer with what the similarity score should be. In the format 'Score: 0.25'. Only return 0,0.25,0.5,0.75, or 1.0 as options,
Examples:\n'''
phrase_map = {'similar':1,'not similar':0, 'unsure':.5}
round_max_loss = {0:1,.25:.75,.5:.5,.75:.75,1:1}
df = pd.read_csv("train.csv")
random_indicies = list(range((len(df))))
random.shuffle(random_indicies)
random_indicies = random_indicies[:64]
score_options = [0,.25,.5,.75,1.0]
count = 0
loss = 0
max_loss = 0
max_count = 50
random_scores = np.random.choice(score_options,size=len(df),replace=True)
random_loss = 0

def buildSystemPrompt(df):
    phrases = [f"Anchor: {df.loc[i,'anchor']}; Target: {df.loc[i,'target']}; Score: {df.loc[i,'score']}" for i in random_indicies]
    phrases ='\n'.join(phrases)
    return system_prompt+phrases

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
        return float(completion.choices[0].message.content.lower()[7:])


def compareScores(completion_score,real_score):
    if completion_score is None:
        return 1
    if abs(completion_score-real_score) <= 0.25:
        return 0
    else:
        return abs(completion_score-real_score)

print(df.head(6))
for i in range(len(df)):
    if i in random_indicies:
        continue
    if count>max_count:
        break
    completion = checkPhrase(df.loc[i,'anchor'],df.loc[i,'target'])
    print(completion.choices[0].message.content)
    loss = loss + compareScores(scoreCompletion(completion),df.loc[i,'score'])
    random_loss = random_loss + compareScores(random_scores[i],df.loc[i,'score'])
    max_loss = max_loss + round_max_loss[df.loc[i,'score']]
    count = count+1


print(1-loss/max_loss)
print(1-random_loss/max_loss)
print(df.loc[:, 'score'].mean())