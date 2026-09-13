# Neural Networks: Zero to Hero — worked through

A hands-on learning repository following Andrej Karpathy's [**Neural Networks: Zero to Hero**](https://karpathy.ai/zero-to-hero.html) series, with annotated notebooks and from-scratch PyTorch implementations that build from character-level language models to Transformers, tokenization, and GPT-2 checkpoint reproduction.

The goal here is not to hide the mechanics behind high-level APIs. The notebooks work through tensor shapes, loss functions, backpropagation, normalization, autoregressive generation, attention, tokenization, and checkpoint loading directly enough that the abstractions stop feeling magical.

> **Current status:** the repository covers the full **makemore Parts 1–5 → GPT → GPT Tokenizer** progression and also includes a GPT-2 reproduction script. The official course begins one lecture earlier with **micrograd**, which is not yet included, so this is not yet a literal 100% mirror of the official syllabus.

## Course coverage

| Topic | Repository artifact | Status |
| --- | --- | --- |
| Backpropagation / micrograd | — | **Not yet included** |
| Makemore Part 1 — bigram language model | [`makemore/build_makemore_yay_bigram.ipynb`](makemore/build_makemore_yay_bigram.ipynb) | Complete |
| Makemore Part 2 — MLP | [`makemore/build_makemore_yay_MLP.ipynb`](makemore/build_makemore_yay_MLP.ipynb) | Complete |
| Makemore Part 3 — activations, gradients & BatchNorm | [`makemore/build_makemore_yay_RNN.ipynb`](makemore/build_makemore_yay_RNN.ipynb) | Complete |
| Makemore Part 4 — manual backprop | [`makemore/build_makemore_manual_backprop.ipynb`](makemore/build_makemore_manual_backprop.ipynb) | Complete |
| Makemore Part 5 — WaveNet-style hierarchy | [`makemore/build_makemore_yay_Wavenet.ipynb`](makemore/build_makemore_yay_Wavenet.ipynb) | Complete |
| GPT from scratch | [`nanogpt-lecture/gpt_dev.ipynb`](nanogpt-lecture/gpt_dev.ipynb) + [`build_GPT.py`](nanogpt-lecture/build_GPT.py) | Notebook partial; finished script complete |
| GPT tokenizer / BPE | [`Tokenizer.ipynb`](Tokenizer.ipynb) | Complete |
| GPT-2 architecture + checkpoint loading | [`GPT2_from_scratch - Andrej Karpathy.py`](nanogpt-lecture/GPT2_from_scratch%20-%20Andrej%20Karpathy.py) | Extra follow-up implementation |

## Repository layout

```text
├── Tokenizer.ipynb
├── names.txt
├── makemore/
│   ├── build_makemore_yay_bigram.ipynb
│   ├── build_makemore_yay_MLP.ipynb
│   ├── build_makemore_yay_RNN.ipynb
│   ├── build_makemore_manual_backprop.ipynb
│   ├── build_makemore_yay_Wavenet.ipynb
│   ├── build_GPT.ipynb                     # currently an empty scaffold
│   ├── makemore.py                         # Karpathy reference implementation
│   └── names.txt
└── nanogpt-lecture/
    ├── gpt_dev.ipynb                       # incremental GPT development notebook
    ├── build_GPT.py                        # finished ~10.8M-parameter character GPT
    ├── GPT2_from_scratch - Andrej Karpathy.py
    └── input.txt                           # Tiny Shakespeare
```

---

## 1. The makemore progression

The makemore sequence is where the repository builds the fundamentals before attention appears. `names.txt` contains **32,032 names**, and the same character-level modeling problem is attacked with progressively richer neural-network machinery.

| Notebook | Main idea | Recorded result |
| --- | --- | --- |
| `build_makemore_yay_bigram.ipynb` | Count-based bigram model, then the same model re-derived as a trainable neural network | loss **2.4622** |
| `build_makemore_yay_MLP.ipynb` | Character embeddings, MLP context model, minibatching, learning-rate search, train/dev/test split | — |
| `build_makemore_yay_RNN.ipynb` | Activation/gradient statistics, initialization, hand-rolled `Linear`, `Tanh`, and `BatchNorm1d` | train **2.4342** / val **2.4390** |
| `build_makemore_manual_backprop.ipynb` | Manual differentiation through the entire MLP + BatchNorm stack and comparison with autograd | gradient checks |
| `build_makemore_yay_Wavenet.ipynb` | Hierarchical character grouping with `FlattenConsecutive` and a custom `Sequential` | train **2.4243** / val **2.4378** |

The important progression is not simply that each architecture pushes the loss lower. It is that each notebook exposes a different piece of neural-network mechanics: representations, optimization, initialization, normalization, manual gradient flow, and hierarchical context aggregation.

Part 1 is especially useful because the count table is re-derived as a neural network. Once the learned weights and probabilities can be connected back to something analytically understandable, gradient descent stops looking like a black box.

---

## 2. Building GPT from scratch

> Lecture: [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY)

There are two versions of the GPT work in this repository:

- [`gpt_dev.ipynb`](nanogpt-lecture/gpt_dev.ipynb) records the early step-by-step development path and currently stops at the bigram baseline.
- [`build_GPT.py`](nanogpt-lecture/build_GPT.py) contains the finished decoder-only Transformer from the lecture.

### Finished model configuration

`build_GPT.py` contains **10,788,929 parameters**:

| Hyperparameter | Value |
| --- | --- |
| Embedding width | `384` |
| Attention heads | `6` |
| Transformer blocks | `6` |
| Context length | `256` |
| Batch size | `64` |
| Dropout | `0.2` |
| Optimizer | AdamW |
| Learning rate | `3e-4` |
| Training iterations | `5,000` |

Parameter breakdown:

| Component | Parameters |
| --- | ---: |
| Token embedding | 24,960 |
| Position embedding | 98,304 |
| 6 Transformer blocks | 10,639,872 |
| Final LayerNorm | 768 |
| LM head | 25,025 |
| **Total** | **10,788,929** |

### What the implementation makes explicit

**Causal self-attention.** Queries and keys form an affinity matrix of shape `(B, T, T)`. Scaling by `1 / sqrt(head_size)` keeps the softmax logits from becoming excessively sharp as the head dimension grows, while the lower-triangular mask prevents each token from reading future positions.

**Multi-head attention.** Six independent 64-dimensional heads run in parallel and concatenate back into the 384-dimensional residual stream before the output projection mixes them.

**Pre-LayerNorm residual blocks.** The model uses the modern pre-norm pattern:

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

The residual pathway gives gradients an identity route through the network while attention and the MLP learn residual updates.

**Communication followed by computation.** Attention exchanges information across token positions; the `384 → 1536 → 384` feed-forward network then performs per-token computation on the result.

**Learned positional embeddings.** Self-attention itself is permutation-agnostic, so position embeddings inject ordering information and establish the model's finite context window.

### Running the character GPT

```bash
cd nanogpt-lecture
python build_GPT.py
```

The script uses CUDA automatically when available and otherwise falls back to CPU. The full 5,000-step configuration is intended for a GPU; for a CPU smoke test, reduce `n_layer`, `n_embd`, `block_size`, `batch_size`, and `max_iters`.

---

## 3. Reproducing GPT-2

> Follow-up lecture: [*Let's reproduce GPT-2 (124M)*](https://www.youtube.com/watch?v=l8pRSuU81PU)
>
> Script: [`nanogpt-lecture/GPT2_from_scratch - Andrej Karpathy.py`](nanogpt-lecture/GPT2_from_scratch%20-%20Andrej%20Karpathy.py)

This script defines a GPT-2-compatible architecture locally, downloads a Hugging Face `GPT2LMHeadModel`, copies the pretrained checkpoint into the local implementation, and generates text using the copied weights.

The current implementation supports the architecture configurations for:

- `gpt2`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`

The example at the bottom loads `gpt2`.

### What changed in the latest implementation

The current script fixes several rough edges from the earlier version:

- model hyperparameters now live in a `GPTConfig` dataclass instead of unused module-level variables;
- the `gpt2-large` configuration key is spelled correctly;
- generated tokens are moved to the detected `device`, so CPU, CUDA, and Apple MPS are handled consistently;
- dropout is now represented explicitly in attention and residual/MLP paths via `GPTConfig.dropout`.

### Mapping the lecture implementation to GPT-2

| Small GPT implementation | GPT-2 reproduction | Why |
| --- | --- | --- |
| separate K/Q/V linear layers | one `c_attn` projected to `3 * n_embd`, then split | one batched projection |
| Python `Head` objects | tensorized multi-head attention | heads become a tensor dimension |
| ReLU | `nn.GELU(approximate='tanh')` | matches GPT-2's activation |
| token/position tables | `transformer.wte` / `transformer.wpe` | matches checkpoint naming |
| independent block hyperparameters | `GPTConfig` | architecture is configured from one object |
| attention + residual dropout | explicit `nn.Dropout` modules | disabled automatically during `model.eval()` |

For GPT-2 small, the structural configuration is:

```text
vocab_size = 50,257
block_size = 1,024
n_layer    = 12
n_head     = 12
n_embd     = 768
```

### Checkpoint-loading surgery

Four matrices require transposition because Hugging Face's GPT-2 checkpoint inherits OpenAI's `Conv1D` weight orientation:

- `attn.c_attn.weight`
- `attn.c_proj.weight`
- `mlp.c_fc.weight`
- `mlp.c_proj.weight`

The implementation asserts the expected transposed shape before copying each tensor. Attention-mask buffers are excluded from the comparison because they are constants rather than learned parameters.

This makes checkpoint loading a useful structural correctness test: if names or tensor shapes do not line up, the copy fails immediately.

### Generation

The script:

1. tokenizes `"Hello, I'm a language model,"` with `tiktoken`'s GPT-2 encoding;
2. repeats the prompt across five sequences;
3. runs the local GPT implementation;
4. keeps the top 50 next-token candidates;
5. samples one candidate with `torch.multinomial`;
6. repeats until each sequence contains 30 tokens.

Device selection is automatic:

```python
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
```

### Why the local model reports 163M parameters instead of 124M

The local implementation currently does **not tie** the input token embedding and output LM-head weights.

| Component | Parameters |
| --- | ---: |
| Token embedding `wte` | 38,597,376 |
| LM head | 38,597,376 |
| **Total as currently instantiated** | **163,037,184** |
| **Total with tied embedding/head weights** | **124,439,808** |

The Hugging Face checkpoint provides identical values for the two tensors, so inference still reproduces the pretrained model's mapping after copying. They simply occupy separate parameter storage in this local class.

### Dropout note

The script is currently an **inference reproduction**, and `model.eval()` disables all dropout before generation, so the configured probability does not affect the generated outputs.

If this class is extended into a faithful GPT-2 training implementation, two details should be revisited:

- `GPTConfig.dropout` is currently `0.2`, whereas canonical GPT-2 uses `0.1` dropout probabilities;
- GPT-2 also applies embedding dropout after adding token and positional embeddings, which this script does not currently implement.

---

## 4. Building the GPT tokenizer

> Lecture: [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE)
>
> Notebook: [`Tokenizer.ipynb`](Tokenizer.ipynb)

The tokenizer notebook moves below the model layer and builds Byte Pair Encoding from raw UTF-8 bytes.

It covers:

- Unicode code points versus UTF-8 bytes;
- pair-frequency counting;
- iterative BPE merges;
- encode/decode logic over a learned merge table;
- GPT-2's regex-based text splitting;
- inspection of GPT-2's published `encoder.json` and `vocab.bpe` artifacts;
- why GPT-2 ends up with a 50,257-token vocabulary;
- a minimal tokenizer interface inspired by `minbpe`;
- SentencePiece as a contrasting tokenizer family.

The important conceptual separation is that **the tokenizer is trained independently from the language model**. It defines the discrete vocabulary and sequence representation the model receives; many seemingly strange LLM behaviors originate in this preprocessing layer rather than in the Transformer itself.

---

## Running the GPT-2 script

```bash
pip install torch transformers tiktoken
cd nanogpt-lecture
python "GPT2_from_scratch - Andrej Karpathy.py"
```

The first run downloads the selected Hugging Face GPT-2 checkpoint. `gpt2-medium`, `gpt2-large`, and especially `gpt2-xl` require substantially more RAM/VRAM than the default small model.

The notebooks are easiest to run in Jupyter or Google Colab. A pinned repository-level environment is not yet included; adding one is on the recommended cleanup list below.

---

## Known gaps and rough edges

This is intentionally a learning repository rather than a polished library, but these are the current limitations worth keeping explicit.

### `gpt_dev.ipynb`

- The notebook currently stops at the **bigram baseline**, while the finished Transformer lives in `build_GPT.py`.
- Its final generation cell is malformed: `torch.zeros(idx, ...)` is used where the seed tensor should be passed to `generate`, and the expression has mismatched parentheses. The saved output comes from an earlier working state.
- Its Colab badge still points to the older `pop123-ux/makemore-project` path instead of this repository.

### `GPT2_from_scratch - Andrej Karpathy.py`

- It is currently **inference-only**: `forward` returns logits, there is no target/loss path, optimizer, scheduler, or training loop.
- Sampling is written inline at module level rather than exposed as a reusable `generate()` method.
- There is no `if __name__ == "__main__":` guard, so importing the file also downloads the checkpoint and starts generation.
- Input embeddings and the LM head are not weight-tied, so the local class allocates about 163M parameters rather than GPT-2 small's canonical ~124M.
- Dropout is sufficient for inference reproduction because it is disabled in evaluation mode, but the class is not yet an exact GPT-2 training implementation.

### Repository structure

- `makemore/build_GPT.ipynb` currently contains only empty cells and should either be populated or removed.
- `names.txt` is duplicated at the repository root and inside `makemore/`.
- There is no top-level `requirements.txt` / `pyproject.toml` describing the full environment.
- There is no top-level automated test suite or GitHub Actions workflow.
- The official course's opening **micrograd** lecture is not yet represented.

---

## Recommended validation targets

For turning this from a strong learning archive into a stronger portfolio repository, the highest-value tests are small and deterministic — full training does **not** belong in CI.

1. **GPT forward-shape test** — verify `(B, T)` token IDs produce the expected logits shape.
2. **Causal-mask test** — changing a future token must not change logits at an earlier position.
3. **Parameter-count test** — lock the small GPT at `10,788,929` parameters unless the architecture intentionally changes.
4. **Checkpoint-equivalence test** — after loading GPT-2 weights, compare local logits against Hugging Face GPT-2 for the same short token sequence with a tight tolerance.
5. **Tokenizer round-trip tests** — verify `decode(encode(text)) == text` across ASCII, Unicode, whitespace, and emoji examples.
6. **BPE parity tests** — compare the finished GPT-2 tokenizer path against `tiktoken` on a small fixed corpus.
7. **Notebook smoke checks** — at minimum validate notebook JSON and syntax; execute only lightweight notebooks/cells in CI.

A compact CPU-only `pytest` suite covering those properties would add more portfolio value than committing trained checkpoints or running multi-hour training jobs in Actions.

---

## Suggested next steps

In priority order:

1. **Add the micrograd lecture** so the repository genuinely covers the official Zero to Hero syllabus from the beginning.
2. **Finish or repair `gpt_dev.ipynb`**, especially the attention derivation and broken final cell.
3. **Delete or populate `makemore/build_GPT.ipynb`** — an empty notebook weakens the repo more than its absence would.
4. **Add `pyproject.toml` or `requirements.txt` + a CPU test suite + GitHub Actions.**
5. **Add GPT-2 weight tying and a reusable `generate()` method.**
6. **Add a numerical Hugging Face parity test** for the copied GPT-2 checkpoint.
7. **Rename the GPT-2 script** to something shell-friendly such as `gpt2_from_scratch.py` and place the executable demo behind a `__main__` guard.
8. **Deduplicate `names.txt`** and make all makemore notebooks use the same canonical dataset path.
9. **Add a top-level license/attribution file** covering the repository as a whole, while retaining credit to Karpathy and upstream datasets/code.

---

## Credits

The course material and original educational implementations are by [Andrej Karpathy](https://github.com/karpathy):

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [makemore](https://github.com/karpathy/makemore)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

Tiny Shakespeare comes from the [char-rnn](https://github.com/karpathy/char-rnn) dataset.

This repository is my worked-through learning record: notebooks, annotations, experiments, and implementation notes built while following the material.

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium: [medium.com/@Pop123](https://medium.com/@Pop123)
