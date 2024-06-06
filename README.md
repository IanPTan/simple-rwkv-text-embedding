# Simple RWKV Text Embedding

## Introduction
This repo is just a small minimal test to see how well the RWKV model works for text embedding. The idea is simple. RWKV is fundamentally an RNN architecture and so it stands to reason that it has a statically sized hidden state which acts as the memory of the model and accumilates important information about the text. Hence all one needs to do in order to create a text embedding is to run text through the model and then use the hidden state as the embedding. To setup the repository read [this section](#setup)

## How it works

### The RWKV Hidden State
The hidden state is actually a tensor with the shape ``(N_LAYER, 4, N_EMBD)`` where ``N_LAYER`` is the number of layers and ``N_EMBD`` is the size of the embedding. Within each layer there are essentialy two main types of hidden states: previous neuron activations, and then the attention memory state. The attention memory state is stored as two seperate vectors (of size ``N_EMBD``): ``num`` and ``den``. The ``num`` vector is essentilaly the weighted sum of vectors encoding useful information to remember at each time step, and ``den`` is just the sum of those weights. Together, dividing ``num`` by ``den`` produces a weighted average of useful information.

### How to Turn the RWKV Hidden State into an Embedding
The entire hidden state is not required for the text embedding. Firstly, in order to get the richest information, using the deepest layer's memory state is the best decision. So that narrows it down to the matrix at ``state[-1]`` which has the shape ``(4, N_EMBD)``. Next the most important information is the information in the attention mechanism and so ``num`` and ``den`` are the vectors needed. They are stored in ``state[-1][1]`` and ``state[-1][2]`` respectively. Altogether, that means ``embedding = state[-1][1] / state[-1][1]``.

## How to Setup<a id="setup"></a>

### 1. Clone the Repository
```bash
git clone https://github.com/IanPatrickMTan/simple-rwkv-text-embedding/tree/main
```

### 2. Create a Virtual Environment (optional but recommended)
First enter inside the repository's directory and then create and activate a virutal environment, (this guide assumes linux is used).
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install the Required Packages
Using the package manager of your choice, (this guide uses pip), install the packages from the ``requirements.txt`` file.
```bash
pip install -r requirements.txt
```

### 4. Run the Script
```bash
cd src
python main.py
```
