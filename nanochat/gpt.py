"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768

#https://blog.csdn.net/shizheng_Li/article/details/145830637
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

"""
This function implements Rotary Positional Embeddings (RoPE). Instead of adding a position vector to the embeddings (like in the original Transformer), RoPE rotates the query and key vectors based on their position in the sequence.

Here is the step-by-step breakdown of the math happening in the code:

d = x.shape[3] // 2: It calculates the halfway point of the head dimension. RoPE works by rotating pairs of numbers. This implementation treats the first half of the vector and the second half of the vector as the pairs to rotate.
x1, x2 = ...: It splits the input vector x into two halves:
x1: The first half of the features.
x2: The second half of the features.
The Rotation (y1, y2): It applies a rotation matrix to these pairs. The formula being applied is effectively: $$ \begin{pmatrix} y_1 \ y_2 \end{pmatrix} = \begin{pmatrix} \cos \theta & \sin \theta \ -\sin \theta & \cos \theta \end{pmatrix} \begin{pmatrix} x_1 \ x_2 \end{pmatrix} $$
y1 = x1 * cos + x2 * sin
y2 = x1 * (-sin) + x2 * cos
This rotates the vector in high-dimensional space. The angle of rotation ($\theta$) depends on the token's position in the sequence, which allows the model to understand relative distances between tokens (e.g., "word A is 5 words before word B").
torch.cat([y1, y2], 3): It stitches the two rotated halves back together to form the final query or key vector.
Why do this? This allows the attention mechanism to naturally understand relative positions (how far apart tokens are) rather than just absolute positions (index 5 vs index 100), which generally leads to better performance on long sequences.
"""
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    #x(B, T, self.n_head, self.head_dim)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble 在第4维上拼接，拼接回head_dim
    out = out.to(x.dtype) # ensure input/output dtypes match
    """
    (Pdb) x.dtype
torch.float32
(Pdb) out.dtype
torch.float32
    """
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # n_kv_head是n_head的因数
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size() # batch, sequence 长度, channel/或者说embedding 维度

        # Project the input to get queries, keys, and values
        # x是(batch, sequence 长度, self.n_embd)
        # c_q是（self.n_embd, self.n_head * self.head_dim）
        # x * c_q的输出是(batch, sequence 长度, self.n_head * self.head_dim)
        # 也就是c_q(x)的输出是(B, sequence 长度, self.n_head * self.head_dim)
        """
        The .view() function in PyTorch is used to reshape a tensor without changing its underlying data.

In this specific line:

python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
It is transforming the flat output of the linear layer into separate attention heads. Here is the step-by-step breakdown:

Linear Projection (self.c_q(x)):
Input x has shape 
(Batch, Sequence_Length, n_embd)
.
The linear layer projects this to shape 
(Batch, Sequence_Length, n_head * head_dim)
.
At this point, all the data for all heads is flattened into one large last dimension.
Reshape (.view(...)):
It takes that flattened last dimension and splits it into two dimensions: n_head (number of heads) and head_dim (size of each head).
New Shape: 
(Batch, Sequence_Length, n_head, head_dim)
.
Why? This prepares the tensor for Multi-Head Attention. By separating the n_head dimension, the code can later process each attention head independently (e.g., in the subsequent transpose and scaled_dot_product_attention steps).


        """
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        # 转成每个head内，seq len * Head_dim
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        # size(2)是seq len
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            """
            if kv_cache is None or Tq == Tk:
This handles the common training case (no kv cache) and also the case where the current forward pass contains the full key sequence (Tq == Tk).
Calls F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
is_causal=True enforces causal masking internally (each query can only attend to the same-or-earlier time positions).
enable_gqa passes the GQA flag through for correct head handling.

            """
            # Transformer 特制函数：https://zhuanlan.zhihu.com/p/1885368445824123521
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
            # Tq == Tk表示当前batch的query和key的sequence长度相同
            # 那么就进行全量的attention计算，scaled_dot_product_attention函数内部会自动进行causal masking
            # 计算出来的y是(batch, n_head, Tq, head_dim)，包含当前batch里面每个q对自己和前面的k的attention结果，就是所有v按照q和k的相关性得分softmax后的加权平均（参照相关attention原理）
        elif Tq == 1: 
            # 比如，上一次生成后，用户输入了一个token的prompt，或者自回归时一个个生成token，这些情况下Tq == 1
            # Tq == 1表示只送了一个token来推理，q变量只有一行
            # 为什么Tq==1表示is_causal=False？
            # is_causal=False here because the cached keys already represent earlier tokens only, and there's no need for an extra
            #  causal mask — the single query can attend to all cached keys.


            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # 比如上一次生成后，prompt前缀被cache了，接着用户问了多个token的prompt

            """
            Lines 97–106 (third branch)

else:
This handles inference when we pass a chunk of multiple new queries (Tq > 1) while having a non-empty prefix in the cache (Tk > Tq). We must allow each query to attend to:
all cached prefix keys (full prefix), and
only the causal portion of the current chunk (so later queries in the chunk cannot see future queries in the same chunk).
attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
Creates a boolean mask of shape (queries in this chunk, total keys). The code comment states "True = keep, False = mask" (the mask is built accordingly).
prefix_len = Tk - Tq
The number of prefix tokens (cached keys) preceding this chunk.
if prefix_len > 0: attn_mask[:, :prefix_len] = True
Allow all queries in the chunk to attend to the entire cached prefix.
attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
For the portion corresponding to keys from the current chunk, set a lower-triangular (causal) pattern so each query i can attend to queries <= i within the chunk.
y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
Call attention with the explicit attn_mask constructed above (and GQA flag). Note the code uses attn_mask instead of is_causal because we need the mixed prefix+causal pattern.
            """
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            # 如 attn_mask = 
            # [[False, False, False, False],
            #  [False, False, False, False]]
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            # Tq=2, Tk=4, prefix_len=2， 那么
            prefix_len = Tk - Tq #如果有KV Cache，Tk>Tq。 Tk=prefix_len+Tq，prefix_len为已缓存的token数
            attn_mask[:, :prefix_len] = True #每一行的前prefix_len个元素设置为True，表示所有cache的key-value对都可以被当前query做attention
            #此时attn_mask=
            # [[True, True, False, False],
            #  [True, True, False, False]]
            # Then, causal attention within this chunk
            # 每一行prefix_len之后的元素生成下三角矩阵，也就是只跟当前chunk之前的token做attention
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            # 此时attn_mask=
            # [ [True, True, True, False],
            #   [True, True, True, True]]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
            """ 
            branch 1和Branch 3的区别是:
            branch 1是完整的attention mask: 
             [[True, False, False, False],
              [True, True, False, False]，
              [True, True, True, False],
              [True, True, True, True]，
              ]

            branch 3:
             #此时attn_mask=
            # [[True, True, False, False],
            #  [True, True, False, False]]
            
            """


        """
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
The attention output y from F.scaled_dot_product_attention has heads as a separate dimension, typically (B, H, T, D). This transpose swaps head and time dims back to (B, T, H, D), contiguous() ensures memory layout is contiguous, and view(B, T, -1) flattens the heads into the embedding dimension so the tensor becomes (B, T, H*D) == (B, T, C).
y = self.c_proj(y)
Applies the output linear projection (c_proj) to mix head outputs back into the model embedding dimension (residual stream) and produce the final attention contribution returned by the attention module.
Notes / shapes

Before the attention choices:
q: (B, H_q, Tq, D)
k, v: (B, H_kv, Tk, D)
H_q = self.n_head, H_kv = self.n_kv_head, D = head_dim
After attention:
y (from attention): (B, H_q, Tq, D) — then reassembled to (B, Tq, C) with C = H_q * D and projected to (B, Tq, C).
The branching ensures correct causal behavior both in training (full sequence) and in different inference modes (single-step vs chunked multi-step) while supporting GQA head-count mismatches.
        """
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
        """
        Here is the breakdown:

Check for GPU (if ... == "cuda"): It checks if the token embedding weights (wte.weight) are currently stored on a CUDA device (NVIDIA GPU).
Convert to bfloat16: If the model is on the GPU, it converts the token embedding layer to use the bfloat16 (Brain Floating Point) data type instead of the default float32.
Why do this?

Memory Savings: bfloat16 uses half the memory (2 bytes) compared to float32 (4 bytes). Since the embedding layer (vocab_size × n_embd) can be very large, this significantly reduces the model's memory footprint.
Performance: Modern GPUs (like Ampere or Hopper architectures) are highly optimized for bfloat16 math, making operations faster.
Stability: Unlike standard float16, bfloat16 preserves the dynamic range of float32, which is crucial for training stability in Large Language Models (LLMs).
        """

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, vocab_size) <- very big tensor, large amount of memory
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature # temperature < 1, 会放大logits元素之间的差异，从而使得概率大的元素更有可能被选中. Temperature > 1, 会缩小logits元素之间的差异，从而增加可能性
                probs = F.softmax(logits, dim=-1) 
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng) # 自回归一个个生成token
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)# 自回归一个个生成token
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
