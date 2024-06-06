# Simple RWKV Text Embedding

## Introduction
This is just a small test to see how well the RWKV model works for text embedding. The idea is simple. RWKV is fundamentally an RNN architecture with a statically sized memory state, and so it stands to reason that this state accumilates important information about the text. Hence all one needs to do in order to create a text embedding is to run text through the model and then use the hidden state as the embedding. To set the repository up read [this section](#setup).

## How it works

### The RWKV Memory State
The memory state is actually a tensor with the shape ``(N_LAYER, 4, N_EMBD)`` where ``N_LAYER`` is the number of layers and ``N_EMBD`` is the size of the embedding. Within each layer there are essentialy two main types of memory states: previous neuron activations, and the attention mechanism's state. The attention mechanism's state is stored as two seperate vectors (of size ``N_EMBD``): ``num`` and ``den``. The ``num`` vector is essentilaly the weighted sum of vectors encoding useful information to remember at each time step, and ``den`` is just the sum of those weights. Together, dividing ``num`` by ``den`` produces a weighted average of useful information through all of the time steps.

### How to Create an Embedding
The entire hidden state is not required for the text embedding. Firstly, in order to get the richest information, using the deepest layer's memory state is the best decision. So that narrows it down to the matrix located at ``state[-1]`` which has the shape ``(4, N_EMBD)``. Next the most important information is stord in the attention mechanism and so ``num`` and ``den`` are the vectors needed. They are stored in ``state[-1][1]`` and ``state[-1][2]`` respectively. Altogether, to produce the embedding for a piece of text one can use ``embedding = state[-1][1] / state[-1][2]``.

## How to Setup<a id="setup"></a>

### 1. Clone the Repository
```bash
git clone https://github.com/IanPatrickMTan/simple-rwkv-text-embedding/tree/main
```

### 2. Download the RWKV-4 430M Model
Download the RWKV-4 430M model and place it in the ``model`` directory. If you do not do this the repository does not come with its own model so the script will not work.

### 3. Create a Virtual Environment (optional but recommended)
First enter inside the repository's directory and then create and activate a virutal environment, (this guide assumes linux is used).
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install the Required Packages
Using the package manager of your choice, (this guide uses pip), install the packages from the ``requirements.txt`` file.
```bash
pip install -r requirements.txt
```

### 5. Run the Script
```bash
cd src
python main.py
```

## Credits
A lot of code used here was taken and modified from the official ChatRWKV repository and johanwind's RWKV guide. Much love to the RWKV team and I highly recommend reading the RWKV paper.
<br>
https://github.com/BlinkDL/ChatRWKV
<br>
https://johanwind.github.io/2023/03/23/rwkv_details.html
<br>
https://arxiv.org/abs/2305.13048
