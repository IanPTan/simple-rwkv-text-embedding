import numpy as np
from torch import load as torch_load
from rwkv import RWKV

def dumb_simi(a, b):

  return ((a - b) ** 2).sum()

def cos_simi(a, b):

  dot_prod = a @ b
  a_mag = (a ** 2).sum()
  b_mag = (b ** 2).sum()

  return dot_prod / (a_mag * b_mag)

def load_model(MODEL_FILE, device='cpu'):

  weights = torch_load(MODEL_FILE, map_location=device)

  for k in weights.keys():
    if '.time_' in k: weights[k] = weights[k].squeeze()
    weights[k] = weights[k].float().numpy()

  return weights

def init_state(N_LAYER, N_EMBD):

  return np.zeros((N_LAYER, 4, N_EMBD), dtype=np.float32)

def embed(text, weights, tokenizer, N_LAYER, N_EMBD):

  state = init_state(N_LAYER, N_EMBD)

  for token in tokenizer.encode(text).ids:
    probs, state = RWKV(weights, token, state, N_LAYER)

  embed = state[-1][1] / state[-1][2]

  return embed

def load_texts(file_name):

  with open(file_name, 'r') as f:
    raw_texts = f.read()

  texts = raw_texts.split('\n')[:-1]

  return texts

def pca(X, num_components):

  X_meaned = X - np.mean(X, axis=0)
  
  cov_mat = np.cov(X_meaned, rowvar=False)
  
  eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
  
  sorted_index = np.argsort(eigenvalues)[::-1]
  sorted_eigenvectors = eigenvectors[:, sorted_index]
  
  eigenvector_subset = sorted_eigenvectors[:, :num_components]
  
  X_reduced = np.dot(X_meaned, eigenvector_subset)

  return X_reduced
