"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    token_to_id_map = {}
    index = 0
    # Step 1 : Assign index to special tokens 
    for special_token in specials:
        if special_token not in token_to_id_map.keys():
            token_to_id_map[special_token] = index
            index += 1
    # Step 2 : Assign index to tokens from sentences
    for tokens in sentences:
        tokens = tokens.split(" ")
        for token in tokens:
            if token not in token_to_id_map.keys():
                token_to_id_map[token] = index 
                index += 1
    return token_to_id_map

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token_map = { }
    for token, index in token_to_id.items():
        id_to_token_map[index] = token
    return id_to_token_map

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    token_ids = [ ]
    if len(sentence) > 0:
        tokens = sentence.split()
        for token in tokens:
            if token not in token_to_id:
                token_ids.append(token_to_id.get(unk_token))
            else:
                token_ids.append(token_to_id.get(token))
    return token_ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    tokens = [ ]
    for index in ids:
        tokens.append(id_to_token[index])
    return tokens

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model_v1(embeddings, d_model):
    scale = torch.sqrt(
        torch.tensor(
            d_model,
            dtype=embeddings.dtype,
            device=embeddings.device
        )
    )
    return embeddings * scale

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    scaling_factor = math.sqrt(d_model)
    return scaling_factor * embeddings

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    return torch.exp(torch.arange(0, d_model,2,dtype=torch.float32) * (-math.log(10000.0) / d_model))

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len,dtype=torch.float32).reshape(max_len, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros((max_len, d_model),dtype=torch.float32)
    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)
    pe = fill_even_indices_with_sin(pe,position,div_term)
    pe = fill_odd_indices_with_cos(pe,position,div_term)
    # pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    L = embedded_batch.shape[1]
    return (embedded_batch + positional_encoding[:L])

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    # Step 1: Mark all non-padding positions.
    # Shape: (B, L)
    B, L = token_ids.shape
    mask = (token_ids != pad_id)
    # Step 2: Add two singleton dimensions.
    # (B, L) -> (B, 1, 1, L)
    # mask = mask.unsqueeze(1).unsqueeze(2)
    mask = mask.reshape(B, 1, 1, L)
    return mask

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    return mask.reshape(1, 1, seq_len,seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    # key.transpose(-2, -1) means swap last dimension of key with second-last dimension
    return query @ key.transpose(-2, -1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    scaling_factor = math.sqrt(d_k)
    return scores / scaling_factor

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    # Boolean indexing generally expects the boolean tensor to match the indexed dimensions directly.
    # Boolean indexing demands exact shapes, unlike Arithmetic indexing where broadcasting settles down the disputes
    # So, conceptually, scores[~mask] = -torch.inf is correct.
    # But, we will used masked_fill to ensure broadcasting happens, since it is boolean indexing
    scores.masked_fill_(~mask, float("-inf"))
    return scores

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights_v1(masked_scores):
    # Step 1 : Find rows that are entirely -inf
    all_masked = torch.all(torch.isneginf(masked_scores), dim=-1,keepdim=True)
    # Step 2 : Replace -inf rows temporarily with zeros
    safe_scores = torch.where(all_masked,torch.zeros_like(masked_scores), masked_scores)
    # Step 3 : Numerically stable softmax
    max_scores = torch.max(safe_scores, dim=-1, keepdim=True).values
    exp_scores = torch.exp(safe_scores - max_scores)
    sum_exp = torch.sum(exp_scores,dim=-1,keepdim=True)
    attention_weights = (exp_scores / sum_exp)
    # Step 4 : Restore all-masked rows to zeros
    attention_weights = torch.where(
        all_masked,
        torch.zeros_like(attention_weights),
        attention_weights
    )
    return attention_weights

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    return softmax_attention_weights_v1(masked_scores)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    # return attention_weights @ value
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import math
import torch

import math
import torch

def scaled_dot_product_attention(
    query,
    key,
    value,
    mask=None
):
    """
    Run scaled dot-product attention.

    Args:
        query : (..., Lq, d_k)
        key   : (..., Lk, d_k)
        value : (..., Lk, d_v)

        mask:
            (B, Lk)            padding mask
            (B, Lq, Lk)        attention mask
            (B, H, Lq, Lk)     fully expanded mask

            True  = keep
            False = block

    Returns:
        context           : (..., Lq, d_v)
        attention_weights : (..., Lq, Lk)
    """

    # --------------------------------------------------
    # Step 1: Raw attention scores
    # (..., Lq, d_k) @ (..., d_k, Lk)
    # -> (..., Lq, Lk)
    # --------------------------------------------------
    scores = query @ key.transpose(-2, -1)

    # --------------------------------------------------
    # Step 2: Scale by sqrt(d_k)
    # --------------------------------------------------
    d_k = query.shape[-1]
    scores = scores / math.sqrt(d_k)

    # --------------------------------------------------
    # Step 3: Normalize mask shape
    # --------------------------------------------------
    if mask is not None:

        # Case 1:
        # mask already matches scores
        if mask.shape == scores.shape:
            pass

        # Case 2:
        # (Lq, Lk)
        elif mask.dim() == 2 and mask.shape == scores.shape[-2:]:
            pass

        # Case 3:
        # (B, Lk)
        elif mask.dim() == 2:
            mask = mask[:, None, None, :]

        # Case 4:
        # (B, Lq, Lk)
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]

        # Case 5:
        # (B, H, Lq, Lk)
        elif mask.dim() == 4:
            pass

        else:
            raise ValueError(f"Unsupported mask shape {mask.shape}")
        # Mask out disallowed positions
        scores = scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # Detect rows that are completely masked
        all_masked = torch.all(
            ~mask,
            dim=-1,
            keepdim=True
        )

        # Replace fully masked rows temporarily
        safe_scores = torch.where(
            all_masked,
            torch.zeros_like(scores),
            scores
        )

    else:
        all_masked = None
        safe_scores = scores

    # --------------------------------------------------
    # Step 4: Softmax
    # --------------------------------------------------
    attention_weights = torch.softmax(
        safe_scores,
        dim=-1
    )

    # Restore fully-masked rows to zeros
    if all_masked is not None:
        attention_weights = torch.where(
            all_masked,
            torch.zeros_like(attention_weights),
            attention_weights
        )

    # --------------------------------------------------
    # Step 5: Weighted sum of values
    # (..., Lq, Lk) @ (..., Lk, d_v)
    # -> (..., Lq, d_v)
    # --------------------------------------------------
    context = attention_weights @ value

    return context, attention_weights


def scaled_dot_product_attention_v1(
    query,
    key,
    value,
    mask=None
):
    """
    Run scaled dot-product attention.

    Args:
        query : (..., Lq, d_k)
        key   : (..., Lk, d_k)
        value : (..., Lk, d_v)
        mask  : broadcastable to (..., Lq, Lk)

    Returns:
        context           : (..., Lq, d_v)
        attention_weights : (..., Lq, Lk)
    """

    # Step 1: Compute raw attention scores
    # scores[i, j] = dot(query_i, key_j)
    # Shape: (..., Lq, d_k) @ (..., d_k, Lk) = (..., Lq, Lk)
    scores = query @ key.transpose(-2, -1)
    # Step 2: Scale scores by sqrt(d_k)
    # Without scaling, large d_k causes dot products to grow in magnitude,
    # pushing softmax into saturated regions and producing tiny gradients.
    # Transformer paper: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    d_k = query.shape[-1]
    scores = scores / math.sqrt(d_k)
    # Step 3: Apply mask (optional)
    # Common masks:
    # 1. Padding mask : Prevent attention to PAD tokens.
    # 2. Causal mask : Prevent attention to future tokens.
    # Any blocked position receives -inf.
    # After softmax: exp(-inf) = 0
    # so its attention probability becomes 0.
    if mask is not None:
        scores = scores.masked_fill(~mask,float("-inf"))
        # Edge case:
        # A row may be completely masked.
        # Example: mask = [[False, False, False]]
        # Then: scores = [[-inf, -inf, -inf]]
        # Softmax would compute: [0, 0, 0] / 0 producing NaNs.
        # Detect such rows before softmax.
        all_masked = torch.all(~mask,dim=-1,keepdim=True)
        # Temporarily replace fully-masked rows with zeros.
        # Example: [-inf, -inf, -inf] becomes: [0, 0, 0] so softmax remains numerically stable.
        safe_scores = torch.where(all_masked,torch.zeros_like(scores),scores)
    else:
        all_masked = None
        safe_scores = scores
    # Step 4: Convert scores into probabilities
    # Softmax is applied over the key dimension.
    # Each row sums to 1.
    # Shape: (..., Lq, Lk)
    attention_weights = torch.softmax(safe_scores,dim=-1)
    # Restore fully-masked rows to all zeros.
    # Example: softmax([0,0,0]) would produce [1/3, 1/3, 1/3]
    # but the correct attention distribution for a fully masked row is [0, 0, 0]
    # because no key is allowed to receive attention.
    if all_masked is not None:
        attention_weights = torch.where(all_masked,torch.zeros_like(attention_weights),attention_weights)
    # Step 5: Weighted sum of value vectors
    # Each query receives a weighted combination of all value vectors.
    # Shape: (..., Lq, Lk) @ (..., Lk, d_v) = (..., Lq, d_v)
    context = attention_weights @ value
    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.shape 
    assert d_model % num_heads == 0 , "d_model is not divisible by num_heads"
    head_dim = d_model // num_heads
    return tensor.reshape(B,L,num_heads,head_dim)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    B, L, num_heads, d_k = split_tensor.shape 
    # Swap 1st dim L and 2nd dim num_heads so we get (B, num_heads, L, d_k)
    return split_tensor.transpose(1,2)
    # We can also use permute [from (0, 1, 2, 3) -> (0, 2, 1, 3)] as shown below
    # return split_tensor.permute(0, 2, 1, 3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B, num_heads, L, d_k = multi_head_tensor.shape
    # Step 1:Move sequence dimension before head dimension.
    # (B, num_heads, L, d_k) -> (B, L, num_heads, d_k)
    # tensor = multi_head_tensor.transpose(1,2)
    tensor = transpose_heads_before_sequence(multi_head_tensor)
    # Step 2: Merge all head features back into one model dimension.
    # (B, L, num_heads, d_k) * (B, L, num_heads * d_k)
    tensor = tensor.reshape(B,L, num_heads * d_k)
    return tensor

# Step 26 - apply_linear_projection
import torch.nn.functional as F 
# F.linear(x, weight, bias) is typically more optimized because it routes directly into PyTorch's low-level 
# C++/CUDA implementation, which can dispatch a single highly tuned GEMM (matrix multiplication) kernel from 
# libraries like cuBLAS, oneDNN, MKL, or OpenBLAS and fuse the bias addition into the same execution path 
# when possible. In contrast, writing x @ weight.T + bias in Python expresses the computation as separate 
# operations: a transpose view, a matrix multiplication, and then a bias add, leaving optimization decisions 
# to later stages. While modern PyTorch often optimizes this well, F.linear communicates intent explicitly 
# ("this is a dense layer"), allowing backend-specific kernels, memory-layout optimizations, graph compilers 
# (torch.compile), quantized inference paths, mixed-precision kernels, and vendor libraries to recognize and 
# optimize the operation more aggressively. In practice the performance difference may be small for toy workloads, 
# but F.linear is the primitive that PyTorch and nn.Linear are built around, so it benefits from the most 
# engineering effort and backend-specific optimizations.

def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    y = x @ weight.T 
    if bias is not None:
        y += bias
    return y

def apply_linear_projection_v1(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    # Computes: x @ weight^T + bias
    return F.linear(x, weight, bias)

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q = transpose_heads_before_sequence(split_last_dim_into_heads(q,num_heads))
    k = transpose_heads_before_sequence(split_last_dim_into_heads(k,num_heads))
    v = transpose_heads_before_sequence(split_last_dim_into_heads(v,num_heads))
    return q, k, v

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    context, attention_weights = scaled_dot_product_attention(q_h,k_h,v_h,mask)
    return context, attention_weights

# Step 30 - merge_heads_and_project_output
import torch

# We want : (B, num_heads, L, d_k) -> (B, L, d_model) -> Output Projection -> (B, L, d_model)
def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    # Step 1: Merge all attention heads back into a single model dimension.
    # (B, num_heads, L, d_k) -> (B, L, d_model)
    merged_context = (merge_heads_back_to_model_dim(context))
    # Step 2: Apply the output projection W_O.
    # (B, L, d_model) -> (B, L, d_model)
    output = apply_linear_projection(merged_context,w_o,b_o)
    return output

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    # Step 1: Project inputs into Q, K and V spaces.
    q = apply_linear_projection(query,w_q,None)
    k = apply_linear_projection(key,w_k,None)
    v = apply_linear_projection(value,w_v,None)
    # Step 2: Split the model dimension across multiple attention heads.
    # (B, L, d_model) -> (B, num_heads, L, d_k)
    q_h, k_h, v_h = split_qkv_into_heads(q,k,v,num_heads)
    # Step 3: Run scaled dot-product attention independently for each head.
    # context:
    #   (B, num_heads, L, d_k)
    # attention_weights:
    #   (B, num_heads, Lq, Lk)
    context, attention_weights = scaled_dot_product_attention(q_h,k_h,v_h,mask)
    # Step 4: Concatenate all heads back together
    # and apply the output projection W_o.
    output = merge_heads_and_project_output(context,w_o,None)
    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    # Step 1: Apply the first FFN linear projection.
    # Shape: (..., d_model) -> (..., d_ff)
    try:
        hidden = x @ w1
    except Exception as err:
        hidden = x @ w1.T 
    if b1 is not None:
        hidden = hidden + b1
    # Step 2: Apply ReLU activation.
    # Negative values become 0. Positive values remain unchanged.
    hidden = torch.relu(hidden)
    return hidden

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    # Step 1: Project the FFN hidden representation back to the model dimension.
    # Shape:(B, L, d_ff) @ (d_ff, d_model) = (B, L, d_model)
    try:
        output = hidden @ w2
    except Exception as err:
        output = hidden @ w2.T 
    # Step 2: Add bias.
    # b2 has shape: (d_model,)
    # PyTorch automatically broadcasts it across
    # batch and sequence dimensions.
    output = output + b2
    return output

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    # Step 1: First FFN linear layer + ReLU.
    # (B, T, d_model) -> (B, T, d_ff)
    hidden = apply_ffn_first_linear_and_relu(x,w1,b1)
    # Step 2: Second FFN linear layer.
    # (B, T, d_ff) -> (B, T, d_model)
    output = apply_ffn_second_linear(hidden,w2,b2)
    return output

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = torch.mean(x,dim=-1,keepdim=True)
    variance = torch.var(x,dim=-1,keepdim=True,unbiased=False)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    # Step 1: Compute mean and variance over the
    # feature dimension.
    mean, variance = (compute_layer_norm_mean_and_variance(x))
    # Step 2: Standardize features.
    # x_hat = (x - mean) / sqrt(variance + eps)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    # Step 3: Apply learnable affine transform.
    # gamma scales each feature.
    # beta shifts each feature.
    # Shapes:
    #   normalized : (..., d_model)
    #   gamma      : (d_model,)
    #   beta       : (d_model,)
    #
    # Broadcasting automatically applies gamma
    # and beta across batch and sequence axes.
    output = (gamma * normalized) + beta
    return output

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    # Step 1: Residual connection.
    # Add the original input to the sublayer output.
    # Shape: (B, T, d_model) + (B, T, d_model) = (B, T, d_model)
    combined = (residual_input + sublayer_output)
    # Step 2: Layer Normalization.
    # Normalize over the feature dimension and apply learnable gamma/beta parameters.
    output = normalize_and_scale_with_gamma_beta(combined,gamma,beta,eps)
    return output

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    # Step 1: Zero out dropped positions.
    # True  -> keep original value
    # False -> replace with 0
    masked_x = x * keep_mask
    # Step 2: Rescale surviving activations.
    # This preserves E[output] = x
    # Example: keep_prob = 0.8
    # surviving activations are multiplied by: 1 / 0.8 = 1.25
    output = masked_x / keep_prob
    return output

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    # Step 1 : Self-attention: x is used as query, key and value.
    attention_output = assemble_multi_head_attention_forward(
        query=x,
        key=x,
        value=x,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
        mask=src_mask
    )
    # Step 2 : Residual connection + LayerNorm.
    output = apply_residual_add_and_norm(
        residual_input=x,
        sublayer_output=attention_output,
        gamma=gamma,
        beta=beta
    )
    return output

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    # Step 1: Position-wise feed-forward network.
    ffn_output = position_wise_feed_forward_network(x,w1,b1,w2,b2)
    # Step 2: Residual connection + LayerNorm.
    output = apply_residual_add_and_norm(residual_input=x,sublayer_output=ffn_output,gamma=gamma,beta=beta)
    return output

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    # h = LayerNorm(x+MHA(x,x,x))
    # y = LayerNorm(h+FFN(h))
    # and then return y . 
    
    # Sublayer 1: Self-Attention + Add & Norm ----
    x = encoder_layer_self_attention_sublayer(
        x=x,
        w_q=layer_params["w_q"],
        w_k=layer_params["w_k"],
        w_v=layer_params["w_v"],
        w_o=layer_params["w_o"],
        gamma=layer_params["attn_gamma"],
        beta=layer_params["attn_beta"],
        num_heads=num_heads,
        src_mask=src_mask,
    )

    # Sublayer 2: Feed-Forward + Add & Norm ----
    x = encoder_layer_feed_forward_sublayer(
        x=x,
        w1=layer_params["w1"],
        b1=layer_params["b1"],
        w2=layer_params["w2"],
        b2=layer_params["b2"],
        gamma=layer_params["ffn_gamma"],
        beta=layer_params["ffn_beta"],
    )

    return x

# Step 42 - stack_encoder_layers
# For a 6-layer encoder, we have : 
# x
# │
# ├─ Encoder Layer 1 → h1
# │
# ├─ Encoder Layer 2 → h2
# │
# ├─ Encoder Layer 3 → h3
# │
# ├─ Encoder Layer 4 → h4
# │
# ├─ Encoder Layer 5 → h5
# │
# └─ Encoder Layer 6 → h6
# All tensors have the same shape: (B, T, d_model)
# Only the representations become progressively richer as they pass through the stack.

def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    # Running hidden state.
    hidden = x
    # Pass through each encoder layer in sequence.
    for layer_params in encoder_layer_params_list:
        hidden = assemble_encoder_layer(hidden,layer_params,num_heads,src_mask)
    return hidden

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

#       y                     : (B, T_tgt, d_model)
#       │
#       ▼
# Masked Self-Attention       : (B, T_tgt, d_model)
#       │
#       ▼
# y + attention_output        : (B, T_tgt, d_model)
#       │
#       ▼
# LayerNorm                   : (B, T_tgt, d_model)

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    # Step 1: Masked multi-head self-attention.
    attention_output = assemble_multi_head_attention_forward(
        query=y,
        key=y,
        value=y,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
        mask=tgt_mask
    )

    # Step 2: Residual connection + LayerNorm.
    output = apply_residual_add_and_norm(
        residual_input=y,
        sublayer_output=attention_output,
        gamma=gamma,
        beta=beta
    )

    return output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    # Cross-attention:
    # Q comes from decoder hidden states.
    # K and V come from encoder outputs.
    attention_output = assemble_multi_head_attention_forward(
        query=y,
        key=encoder_output,
        value=encoder_output,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        num_heads=num_heads,
        mask=src_mask
    )

    # Residual connection + LayerNorm.
    output = apply_residual_add_and_norm(
        residual_input=y,
        sublayer_output=attention_output,
        gamma=gamma,
        beta=beta
    )

    return output

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

#       y                : (B, T_tgt, d_model)
#       │
#       ▼
# Position-wise FFN      : (B, T_tgt, d_model)
#       │
#       ▼
# y + ffn_output         : (B, T_tgt, d_model)
#       │
#       ▼
# LayerNorm              : (B, T_tgt, d_model)
#       │
#       ▼
# Output                 : (B, T_tgt, d_model)
def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
     # Step 1: Position-wise feed-forward network
    ffn_output = position_wise_feed_forward_network(
        y,
        w1,
        b1,
        w2,
        b2
    )

    # Step 2: Residual connection + LayerNorm
    output = apply_residual_add_and_norm(
        residual_input=y,
        sublayer_output=ffn_output,
        gamma=gamma,
        beta=beta
    )

    return output

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(
    y,
    encoder_output,
    layer_params,
    num_heads,
    src_mask,
    tgt_mask
):
    """
    Run a full decoder layer:
      1. Masked self-attention
      2. Cross-attention
      3. Feed-forward
    """
    # Step 1. Masked self-attention
    y = decoder_layer_masked_self_attention_sublayer(
        y=y,
        w_q=layer_params["w_q_self"],
        w_k=layer_params["w_k_self"],
        w_v=layer_params["w_v_self"],
        w_o=layer_params["w_o_self"],
        gamma=layer_params["self_gamma"],
        beta=layer_params["self_beta"],
        num_heads=num_heads,
        tgt_mask=tgt_mask,
    )

    # Step 2 : Encoder-decoder cross attention
    y = decoder_layer_cross_attention_sublayer(
        y=y,
        encoder_output=encoder_output,
        w_q=layer_params["w_q_cross"],
        w_k=layer_params["w_k_cross"],
        w_v=layer_params["w_v_cross"],
        w_o=layer_params["w_o_cross"],
        gamma=layer_params["cross_gamma"],
        beta=layer_params["cross_beta"],
        num_heads=num_heads,
        src_mask=src_mask,
    )
    # # Step 3 : Position-wise feed-forward
    y = decoder_layer_feed_forward_sublayer(
        y=y,
        w1=layer_params["w1"],
        b1=layer_params["b1"],
        w2=layer_params["w2"],
        b2=layer_params["b2"],
        gamma=layer_params["ffn_gamma"],
        beta=layer_params["ffn_beta"],
    )

    return y

# Step 47 - stack_decoder_layers
# y0 -> Decoder Layer 1 -> y1 -> Decoder Layer 2 -> y2 -> .... -> Decoder Layer N -> yN
# All intermediate tensors keep the same shape: (B, T_tgt, d_model)

# Only the representation becomes progressively richer as each layer adds: Masked self-attention, Cross-attention to encoder outputs, and Feed-forward transformation

def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    hidden = y

    for layer_params in decoder_layer_params_list:
        hidden = assemble_decoder_layer(
            y=hidden,
            encoder_output=encoder_output,
            layer_params=layer_params,
            num_heads=num_heads,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

    return hidden

# Step 48 - apply_final_output_projection
# decoder_output          : (B, T, d_model)
# output_projection_weight: (vocab_size, d_model)
# Linear Projection -> logits whose shape is (B, T, vocab_size)

def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(
        decoder_output,
        output_projection_weight,
        output_projection_bias
    )

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab_v1(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return torch.log_softmax(logits, dim=-1)


def apply_log_softmax_over_vocab(logits):
    """
    Args:
        logits: (B, T, V)

    Returns:
        log_probs: (B, T, V)
    """
    # (B, T, 1)
    max_logits = torch.max(logits,dim=-1,keepdim=True).values
    # Prevent overflow in exp()
    shifted_logits = logits - max_logits
    # (B, T, 1)
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits),dim=-1,keepdim=True))
    # log softmax
    log_probs = shifted_logits - log_sum_exp
    return log_probs

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    # Backward-compatible embedding lookup
    try:
        if "token_embedding" in model_params:
            src_embedding = model_params["token_embedding"]
            tgt_embedding = model_params["token_embedding"]
        else:
            src_embedding = model_params["src_embedding"]
            tgt_embedding = model_params["tgt_embedding"]
    except Exception as err:
        print(f"Exception met : {err}")
    # Step 1 : Get Source embeddings and PE
    src = scale_embeddings_by_sqrt_d_model(
        src_embedding[src_ids],
        src_embedding.shape[1]
    )
    src_pe = build_sinusoidal_positional_encoding(src.shape[1],src.shape[2])
    src = add_positional_encoding_to_embeddings(src,src_pe)
    # Step 2 : Get Target embeddings and PE
    tgt = scale_embeddings_by_sqrt_d_model(
        tgt_embedding[tgt_ids],
        tgt_embedding.shape[1]
    )
    tgt_pe = build_sinusoidal_positional_encoding(tgt.shape[1],tgt.shape[2])
    tgt = add_positional_encoding_to_embeddings(tgt,tgt_pe)
    # Step 3 : Get Masks
    src_mask = build_padding_mask(src_ids,pad_id)
    tgt_padding_mask = build_padding_mask(tgt_ids,pad_id)
    tgt_causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask,tgt_causal_mask)
    # Step 4 : Encoder
    encoder_output = stack_encoder_layers(
        src,
        model_params["encoder_layers"],
        num_heads,
        src_mask
    )
    # Step 5 : Decoder
    decoder_output = stack_decoder_layers(
        tgt,
        encoder_output,
        model_params["decoder_layers"],
        num_heads,
        src_mask,
        tgt_mask
    )
    # Step 6 : Do Vocabulary projection
    logits = apply_final_output_projection(
        decoder_output,
        model_params["output_projection"]
    )
    # Step 7 : Get Log probabilities from logits
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

import math
import torch

def init_encoder_layer_parameters(d_model, num_heads, d_ff):

    def xavier(shape):
        fan_in, fan_out = shape
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return (torch.randn(*shape, dtype=torch.float32) * std).requires_grad_()

    return {
        "w_q": xavier((d_model, d_model)),
        "w_k": xavier((d_model, d_model)),
        "w_v": xavier((d_model, d_model)),
        "w_o": xavier((d_model, d_model)),

        "w1": xavier((d_model, d_ff)),
        "b1": torch.zeros(
            d_ff,
            dtype=torch.float32,
            requires_grad=True
        ),

        "w2": xavier((d_ff, d_model)),
        "b2": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),

        "attn_gamma": torch.ones(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        "attn_beta": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),

        "ffn_gamma": torch.ones(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        "ffn_beta": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
    }

# Step 53 - init_decoder_layer_parameters
import torch

import math
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    """
    Return a dict containing all learnable parameters
    for one decoder layer.
    """

    def xavier(shape):
        fan_in, fan_out = shape
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return (
            torch.randn(*shape, dtype=torch.float32) * std
        ).requires_grad_()

    return {
        # Masked self-attention
        "w_q_self": xavier((d_model, d_model)),
        "w_k_self": xavier((d_model, d_model)),
        "w_v_self": xavier((d_model, d_model)),
        "w_o_self": xavier((d_model, d_model)),
        # Cross-attention
        "w_q_cross": xavier((d_model, d_model)),
        "w_k_cross": xavier((d_model, d_model)),
        "w_v_cross": xavier((d_model, d_model)),
        "w_o_cross": xavier((d_model, d_model)),
        # Feed-forward network
        "w1": xavier((d_model, d_ff)),
        "b1": torch.zeros(
            d_ff,
            dtype=torch.float32,
            requires_grad=True
        ),
        "w2": xavier((d_ff, d_model)),
        "b2": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        # Self-attention LayerNorm
        "self_gamma": torch.ones(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        "self_beta": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        # Cross-attention LayerNorm
        "cross_gamma": torch.ones(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        "cross_beta": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        # FFN LayerNorm
        "ffn_gamma": torch.ones(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
        "ffn_beta": torch.zeros(
            d_model,
            dtype=torch.float32,
            requires_grad=True
        ),
    }

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    src_embedding = torch.randn(
        vocab_size,
        d_model,
        dtype=torch.float32,
        requires_grad=True
    )
    tgt_embedding = torch.randn(
        vocab_size,
        d_model,
        dtype=torch.float32,
        requires_grad=True
    )
    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = torch.randn(
            vocab_size,
            d_model,
            dtype=torch.float32,
            requires_grad=True
        )

    return {
        "src_embedding": src_embedding,
        "tgt_embedding": tgt_embedding,
        "output_projection": output_projection,
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    params = []
    seen = set()

    def add_tensor(t):
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            params.append(t)

    # Encoder layers
    for layer in encoder_layer_params:
        for tensor in layer.values():
            add_tensor(tensor)

    # Decoder layers
    for layer in decoder_layer_params:
        for tensor in layer.values():
            add_tensor(tensor)

    # Embeddings / projection
    for tensor in embedding_params.values():
        add_tensor(tensor)

    return params

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    B, T = target_ids.shape
    shifted = torch.empty_like(target_ids)
    shifted[:, 0] = start_token_id
    shifted[:, 1:] = target_ids[:, :-1]
    return shifted
    
def shift_targets_right_with_start_token_v1(target_ids, start_token_id):
    B, T = target_ids.shape
    shifted_target_ids = []
    for batch_id in range(B):
        new_batch = [0] * T
        new_batch[0] = start_token_id
        # Copy everything except the last token
        new_batch[1:] = target_ids[batch_id][:-1].tolist()
        shifted_target_ids.append(new_batch)

    return torch.tensor(
        shifted_target_ids,
        dtype=target_ids.dtype,
    )

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    step = max(step, 1)
    return (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    return torch.full(
        shape,
        epsilon / (vocab_size - 2),
        dtype=torch.float32
    )

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    result = smoothed_distribution.clone()
    B, T = gold_token_ids.shape
    for b in range(B):
        for t in range(T):
            result[b, t, gold_token_ids[b, t]] = confidence

    return result

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    result = smoothed_distribution.clone()
    # Zero the PAD vocabulary column.
    result[:, :, pad_id] = 0
    # Find positions whose target token is PAD.
    pad_rows = (gold_token_ids == pad_id)
    # Zero the entire distribution for those positions.
    result[pad_rows] = 0

    return result

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    loss = -torch.sum(smoothed_distribution * log_probabilities)
    return torch.abs(loss)

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    non_pad_count = (gold_token_ids != pad_id).sum()
    if non_pad_count.item() == 0:
        return total_loss
    return total_loss / non_pad_count

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    # (B, T)
    predictions = torch.argmax(
        log_probabilities,
        dim=-1
    )

    # (B, T)
    non_pad_mask = (gold_token_ids != pad_id)
    non_pad_count = non_pad_mask.sum()

    if non_pad_count.item() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    # (B, T)
    correct = (predictions == gold_token_ids) & non_pad_mask
    accuracy = (correct.float().sum() / non_pad_count.float())
    return accuracy

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    return {
        "m": [torch.zeros_like(p) for p in parameter_list],
        "v": [torch.zeros_like(p) for p in parameter_list],
        "t": 0,
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return beta1 * m_prev + (1.0 - beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1.0 - beta2) * grad * grad

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_hat = m_t / (1.0 - beta1 ** step)
    v_hat = v_t / (1.0 - beta2 ** step)

    return m_hat, v_hat

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    # Step counter
    optimizer_state["t"] += 1
    step = optimizer_state["t"]

    for i, param in enumerate(parameter_list):

        if param.grad is None:
            continue

        grad = param.grad

        # First moment
        m_t = (beta1 * optimizer_state["m"][i] + (1.0 - beta1) * grad)

        # Second moment
        v_t = (beta2 * optimizer_state["v"][i] + (1.0 - beta2) * (grad * grad))

        optimizer_state["m"][i] = m_t
        optimizer_state["v"][i] = v_t

        # Bias correction
        m_hat = m_t / (1.0 - beta1 ** step)
        v_hat = v_t / (1.0 - beta2 ** step)

        # Parameter update (in-place)
        with torch.no_grad():
            param -= (learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon))

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    pad_id = config["pad_id"]
    start_id = config["start_id"]
    vocab_size = config["vocab_size"]
    smoothing = config["smoothing"]
    num_heads = config["num_heads"]

    # run_transformer_forward expects a single (tied) token embedding under
    # "token_embedding". Expose it on model_params in place so the embedding
    # participates in the autograd graph and callers can read its .grad.
    if "token_embedding" not in model_params:
        model_params["token_embedding"] = model_params["src_embedding"]

    # Teacher forcing: shift targets right to form the decoder input.
    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)

    # Forward pass -> log probabilities (B, T, V).
    log_probs = run_transformer_forward(
        src_batch, decoder_input, model_params, num_heads, pad_id
    )

    # Build the label-smoothed target distribution.
    confidence = 1.0 - smoothing
    smoothed = build_uniform_smoothing_distribution(log_probs.shape, vocab_size, smoothing)
    smoothed = set_confidence_on_gold_tokens(smoothed, tgt_batch, confidence)
    smoothed = zero_pad_column_and_pad_token_rows(smoothed, tgt_batch, pad_id)

    # Summed KL loss, averaged over non-pad tokens.
    total_loss = compute_label_smoothed_kl_loss(log_probs, smoothed)
    return average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # TODO: zero grads, compute loss, backward, look up Noam LR, apply Adam step
    # 1. Clear stale gradients
    zero_all_parameter_gradients(parameter_list)

    # 2. Forward pass
    loss = compute_batch_training_loss(
        src_batch,
        tgt_batch,
        model_params,
        config
    )

    # 3. Backpropagation
    loss.backward()

    # 4. Noam learning rate
    learning_rate = compute_noam_learning_rate(
        step=step_number,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"]
    )

    # 5. Adam update
    apply_adam_step_to_all_parameters(
        parameter_list=parameter_list,
        optimizer_state=optimizer_state,
        learning_rate=learning_rate,
        beta1=config.get("beta1", 0.9),
        beta2=config.get("beta2", 0.98),
        epsilon=config.get("epsilon", 1e-9)
    )

    # 6. Return scalar Python float for logging
    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    # TODO: iterate for num_steps steps, calling run_training_step_with_backprop each time
    losses = []

    for step in range(1, num_steps + 1):
        # Cycle through batches if num_steps > len(batches)
        src_batch, tgt_batch = batches[(step - 1) % len(batches)]

        loss = run_training_step_with_backprop(
            src_batch,
            tgt_batch,
            parameter_list,
            model_params,
            optimizer_state,
            step,
            config
        )

        losses.append(loss)

    return losses

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    return torch.argmax(final_step_logits, dim=-1)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    return ((5.0 + sequence_length) / 6.0) ** alpha

# Step 76 - compute_candidate_scores
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    # TODO: add each beam's running log-prob to its row of next-token log probs.
    return beam_scores.unsqueeze(-1) + next_token_log_probs

# Step 77 - select_top_k_candidates
import torch

def select_top_k_candidates(candidate_scores, k):
    # TODO: pick the top k (beam_index, token_id, score) triples from candidate_scores
    num_beams, vocab_size = candidate_scores.shape

    # Flatten all (beam, token) candidates
    flat_scores = candidate_scores.reshape(-1)

    # Top-k scores and flattened indices
    scores, flat_indices = torch.topk(flat_scores, k)

    # Recover original coordinates
    beam_indices = flat_indices // vocab_size
    token_ids = flat_indices % vocab_size

    return {
        "beam_indices": beam_indices,
        "token_ids": token_ids,
        "scores": scores,
    }

# Step 78 - append_tokens_to_beam_sequences
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    # TODO: gather parent beam rows and append the new token ids as the last column
    parent_sequences = beam_sequences[beam_indices]

    return torch.cat(
        [
            parent_sequences,
            token_ids.unsqueeze(1)
        ],
        dim=1
    )

# Step 79 - mark_finished_beams
import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    # TODO: return updated boolean finished flags for each beam given the new token ids
    return finished_flags | (token_ids == end_token_id)

# Step 80 - select_best_finished_beam
def select_best_finished_beam(finished_sequences, finished_scores, alpha):
    # TODO: return the finished beam with the highest length-penalized score
    best_idx = 0    
    best_score = float("-inf")

    for i, sequence in enumerate(finished_sequences):
        raw_score = float(finished_scores[i])

        normalized_score = (
            raw_score
            / compute_length_penalty(len(sequence), alpha)
        )

        if normalized_score > best_score:
            best_score = normalized_score
            best_idx = i

    return {
        "sequence": finished_sequences[best_idx],
        "score": best_score,
    }

