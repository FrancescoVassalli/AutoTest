import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

phrases = ['The block is red', 'The color of the block is red.',
           'The block is big.', 'The fat orange cat fell asleep.',
           'The big orange cat fell asleep.',
           "The hairy ball theorem of algebraic topology says that 'one cannot comb the hair flat on a hairy ball without creating a cowlick.' This fact is immediately convincing to most people, even though they might not recognize the more formal statement of the theorem, that there is no nonvanishing continuous tangent vector field on the sphere.",
           "The furry ball theorem of algebraic topology says that 'you cannot comb the hair of a furry ball without creating a wick'. This fact is immediately convincing to most people, even if they don't recognize the more formal statement of the theorem, that there is no nonzero continuous tangent vector field on the sphere."
           ]

#equivalent phrases simple test case
equiv_test = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are an assistant focused on natural language processing. Respond with yes or no."},
        {"role": "user", "content": f"Do the following two phrases have the same meaning? Phrase 1: {phrases[0]} Phrase 2: {phrases[1]}"},
    ]
)

#unequivalent phrases simple test case
unequiv_test = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are an assistant focused on natural language processing. Respond with yes or no."},
        {"role": "user", "content": f"Do the following two phrases have the same meaning? Phrase 1: {phrases[0]} Phrase 2: {phrases[2]}"},
    ]
)

#simple phrase, one layer of google translate (english -> french -> english)
translate_test1 = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are an assistant focused on natural language processing."},
        {"role": "user", "content": f"Do the following two phrases have the same meaning? Phrase 1: {phrases[3]} Phrase 2: {phrases[4]}"},
    ]
)

#technical jargon phrase, one layer of google translate (english -> french -> english)
translate_test2 = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are an assistant focused on natural language processing."},
        {"role": "user", "content": f"Do the following two phrases have the same meaning? Phrase 1: {phrases[5]} Phrase 2: {phrases[6]}"},
    ]
)

#All cases seem to pass consistently, however cases 3 and 4 only work if "reply with yes or no" is removed from assitant behavior prompt
print(equiv_test["choices"][0]["message"],
      unequiv_test["choices"][0]["message"],
      translate_test1["choices"][0]["message"],
      translate_test2["choices"][0]["message"])