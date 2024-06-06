import numpy as np
from tokenizers import Tokenizer
from rwkv import load_model, RWKV, embed


MODEL_FILE = '../model/RWKV-4-Pile-430M-20220808-8066.pth'
N_LAYER = 24
N_EMBD = 1024

print(f'\nLoading {MODEL_FILE}')

weights = load_model(MODEL_FILE)
tokenizer = Tokenizer.from_file('../model/20B_tokenizer.json')

print(f'\nEmbedding text...')

texts = ['Apples are red.', 'Apples are blue.', 'I hate sharks.']
embeddings = []

for i, text in enumerate(texts):
  embeddings.append(embed(text, weights, tokenizer, N_LAYER, N_EMBD))
  print(f'\n"{text}":\n{np.around(embeddings[-1], 3)}\n')

print('Done.')
