import math
import tiktoken
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print (f"using device: {device}")

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.2
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, more of a mask, but following the OpenAI/HF naming
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nr.of heads, T, head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nr.of heads, T, head size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nr.of heads, T, head size)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.c_proj(y))
        return y
        
class MLP(nn.Module):
    """Multi-Layer Perceptron that expands the representation and applies GELU.
       Essentially performs a large nonlinear transformation on each token independently."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
"""
x
↓
Linear | in_features → hidden_features (in_features*4)
↓
GELU
↓
Linear | (in_features*4) hidden_features → out_features
↓
Dropout
"""

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x): # The gradients from the top flow straight to the inputs, through the residual pathways (dy/dx = I + dF/dx because of the addition) | In addition to that the gradient flows through the blocks and the blocks kick in and change the optimization over time 
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # meaning of token + location of token
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word/token embedding (B,T) -> (B,T,num_dim)
            wpe = nn.Embedding(config.block_size, config.n_embd), # word position embedding (location of token)
            # the transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layernorm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # (B,T,num_dim) -> (B,T,50257)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
        
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600) # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, ignore
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def predict(self, sequence, max_length, num_return_seq, verbose=True):
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(sequence)
            tokens = torch.tensor(tokens, dtype=torch.long) # (len(sequence),)
            tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1) # (num_return_seq, len(sequence)) | generates num_return_seq continuations simultaneously
            
            x = tokens.to(device)
            
            model = self # instantiate GPT model
            model.eval()
            model.to(device)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            # autoregressive generation = one token at a time
            while x.size(1) < max_length:
                previous_shape = tuple(x.shape)

                # all_logits: (B, T, vocab_size)
                all_logits = model(x)

                # We only need the prediction made from the final position.
                # next_token_logits: (B, vocab_size)
                next_token_logits = all_logits[:, -1, :]

                # Keep the 50 highest-logit candidates for EACH sequence.
                # Both tensors have shape (B, 50).
                topk_logits, topk_indices = torch.topk(
                    next_token_logits,
                    k=50,
                    dim=-1
                )

                # Convert only those 50 candidate logits into probabilities.
                topk_probs = F.softmax(topk_logits, dim=-1)

                # Choose ONE candidate position independently for EACH row.
                # Shape: (B, 1). These are positions 0..49, NOT vocabulary IDs.
                ix = torch.multinomial(topk_probs, num_samples=1)

                # Translate each sampled top-k position into the actual GPT-2 token ID.
                # Shape: (B, 1).
                xcol = torch.gather(topk_indices, dim=-1, index=ix)

                # Append one token to every existing sequence.
                # Because dim=1 is the sequence-length dimension:
                # (B, T) + (B, 1) -> (B, T+1)
                # The number of sequences B does NOT increase.
                x = torch.cat((x, xcol), dim=1)

                if verbose:
                    sampled_text = [
                        enc.decode([token_id])
                        for token_id in xcol.squeeze(1).tolist()
                    ]

                    print(
                        f"batch shape: {previous_shape} -> {tuple(x.shape)}\n"
                        f"next-token logits shape: {next_token_logits.shape}\n"
                        f"sampled top-k positions (one per sequence):\n{ix}\n"
                        f"sampled token IDs (one per sequence):\n{xcol}\n"
                        f"sampled token text: {sampled_text}\n"
                        f"sequence 0 so far: {enc.decode(x[0].tolist())}\n"
                    )

            # Decode the B completed sequences.
            for i in range(num_return_seq):
                token_ids = x[i].tolist()  # token IDs, not probabilities
                decoded = enc.decode(token_ids)
                print("-->", decoded)