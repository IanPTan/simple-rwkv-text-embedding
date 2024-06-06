import numpy as np

layer_norm = lambda x, w, b : (x - np.mean(x)) / np.std(x) * w + b
exp = np.exp
sigmoid = lambda x : 1/(1 + exp(-x))

def time_mixing(x, last_x, last_num, last_den, decay, bonus, mix_k, mix_v, mix_r, Wk, Wv, Wr, Wout):

    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    v = Wv @ ( x * mix_v + last_x * (1 - mix_v) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )

    wkv = (last_num + exp(bonus + k) * v) /      \
          (last_den + exp(bonus + k))
    rwkv = sigmoid(r) * wkv

    num = exp(-exp(decay)) * last_num + exp(k) * v
    den = exp(-exp(decay)) * last_den + exp(k)

    return Wout @ rwkv, (x,num,den)

def channel_mixing(x, last_x, mix_k, mix_r, Wk, Wr, Wv):

    k = Wk @ ( x * mix_k + last_x * (1 - mix_k) )
    r = Wr @ ( x * mix_r + last_x * (1 - mix_r) )
    vk = Wv @ np.maximum(k, 0)**2

    return sigmoid(r) * vk, x

def RWKV(model, token, state, N_LAYER):

    params = lambda prefix : [model[key] for key in model.keys() if key.startswith(prefix)]

    x = params('emb')[0][token]
    x = layer_norm(x, *params('blocks.0.ln0'))

    for i in range(N_LAYER):
        x_ = layer_norm(x, *params(f'blocks.{i}.ln1'))
        dx, state[i][:3] = time_mixing(x_, *state[i][:3], *params(f'blocks.{i}.att'))
        x = x + dx

        x_ = layer_norm(x, *params(f'blocks.{i}.ln2'))
        dx, state[i][3] = channel_mixing(x_, state[i][3], *params(f'blocks.{i}.ffn'))
        x = x + dx

    x = layer_norm(x, *params('ln_out'))
    x = params('head')[0] @ x

    e_x = exp(x-np.max(x))
    probs = e_x / e_x.sum() # Softmax of x

    return probs, state

