import os
import openai
import json
import pandas as pd
import random
import numpy as np

df = pd.read_csv("trainShort.csv")
print("read csv")
# from open ai example
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

df['ada_embedding_anchor'] = df.anchor.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
print("got anchor embeddings")
df['ada_embedding_target'] = df.target.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
print("got target embeddings")
df.to_csv('embeded.csv', index=False)
