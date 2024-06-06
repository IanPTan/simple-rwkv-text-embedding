import numpy as np
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from utils import load_model, embed, load_texts, pca

MODEL_FILE = '../model/RWKV-4-Pile-430M-20220808-8066.pth'
N_LAYER = 24
N_EMBD = 1024

print(f'\nLoading {MODEL_FILE}...')

weights = load_model(MODEL_FILE)
tokenizer = Tokenizer.from_file('../model/20B_tokenizer.json')

print('Loading texts...')

texts = load_texts('texts.txt')

try:

  embs = np.load('embeddings.npy')

except:

  print(f'\nEmbedding text...')

  embs = np.zeros((len(texts), N_EMBD), np.float32)

  for i, text in enumerate(texts):
    embs[i] = embed(text, weights, tokenizer, N_LAYER, N_EMBD)
    print(f'\n{i + 1}/{len(texts)} - "{text}":\n{np.around(embs[i], 3)}\n')

  print('Saving embeddings...')

  np.save('embeddings.npy', embs)

print('Creating visualization...')

pca_result = pca(embs, 3)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])

for i, text in enumerate(texts):
  ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], text)

ax.set_title('PCA visualization of embeddings')

print('Done.')

plt.show()
