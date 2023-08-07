import os
import openai
import json
import pandas as pd
import random
import numpy as np
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = '''You are to assign a similarity score to pairs of phrases from US patents. Answer in the format of 'Score: 1.0'. The similarity scores are defined as follows:
1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation and quantity.
0.75 - Close synonym. This also includes abbreviations and acronyms.
0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
0.0 - Unrelated
'''

system_prompt_alt = '''You are given pairs of phrases, each with a similarity score. Using the given phrase pairs and similarity score as context, 
 assign a similarity score to new pairs of phrases in the format 'Score: 1.0'. Only use 1, 2, 3, 4, 5 as score options. 
Examples:\n'''

phrase_map = {'similar':1,'not similar':0, 'unsure':.5}
round_max_loss = {0:1,.25:.75,.5:.5,.75:.75,1:1}
df = pd.read_csv("train.csv")
random_indicies = list(range((len(df))))
random.shuffle(random_indicies)
shuffle_indicies = random_indicies[0:]
random_indicies = random_indicies[:0]
#print(random_indicies)
#print(shuffle_indicies)
score_options = [0,.25,.5,.75,1.0]
bad_indicies = []
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

def buildSystemPrompt_alt(df):
    phrases = [f"Anchor: {df.loc[i,'anchor']}; Target: {df.loc[i,'target']}; Score: {(df.loc[i,'score'] * 4) + 1}" for i in random_indicies]
    phrases ='\n'.join(phrases)
    return system_prompt_alt+phrases

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


def scoreCompletion_alt(completion):
    score_alt = float(completion.choices[0].message.content.lower()[7:])
    return (score_alt - 1) / 4

def compareScores(completion_score,real_score):
    if completion_score is None:
        return 1
    if abs(completion_score-real_score) <= 0:
        return 0
    else:
        return abs(completion_score-real_score)

#system_prompt = buildSystemPrompt(df)

for i in shuffle_indicies:
    if i in random_indicies:
        continue
    if count>max_count:
        break
    completion = checkPhrase(df.loc[i,'anchor'],df.loc[i,'target'])
    print(completion.choices[0].message.content)
    loss = loss + compareScores(scoreCompletion(completion),df.loc[i,'score'])
    random_loss = random_loss + compareScores(random_scores[i],df.loc[i,'score'])
    max_loss = max_loss + round_max_loss[df.loc[i,'score']]
    if loss == max_loss:
        bad_indicies.append(i)
    count = count+1
    time.sleep(1)
print("loss")
print(1-loss/max_loss)
print("random loss")
print(1-random_loss/max_loss)
print(df.loc[:, 'score'].mean())