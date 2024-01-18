import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float32):
        super().__init__()
        self.max_seq_length = max_seq_length
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class ModifiedAttentionLayer(nn.Module):
    def __init__(self, attn_layer, n_heads, seq_len, qkv_names, kv_cache):
        super().__init__()
        self.attn_layer = attn_layer
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.qkv_names = qkv_names
        self.kv_cache = kv_cache

    def forward(self, x: torch.Tensor, attn_mask:torch.Tensor = None, start_pos:int = 0):

        # x dimensions: (batch_size, seq_len, hidden_dim)
        if len(self.qkv_names) == 1:
            # If qkv_names has length 1, split it into q, k, v layers
            qkv_combined = getattr(self.attn_layer, self.qkv_names[0])
            q, k, v = torch.chunk(qkv_combined, 3, dim=-1)
        else:
            q, k, v = [getattr(self.attn_layer, name) for name in self.qkv_names]

        # Update kv_cache with the latest key and value information
        input_pos = torch.arange(start_pos, start_pos + x.shape[-2])

        # Ensure that k(x) and v(x) have the correct dimensions
        k_val, v_val = k(x), v(x)

        k_val = k_val.view(-1, self.n_heads, x.shape[-2], k_val.shape[-1] // self.n_heads)
        v_val = v_val.view(-1, self.n_heads, x.shape[-2], v_val.shape[-1] // self.n_heads)

        assert input_pos.shape[0] == k_val.shape[2], f"Assertion failed. Shapes: {input_pos.shape}, {k_val.shape}"

        k_cache, v_cache = self.kv_cache.update(input_pos, k_val, v_val)

        q_val = q(x)
        q_val = q_val.view(-1, self.n_heads, x.shape[-2], q_val.shape[-1] // self.n_heads)
        # Original self-attention forward pass with updated kv_cache
        attn_scores = torch.matmul(q_val, k_cache.transpose(-2, -1))

        attn_scores = attn_scores / (k_cache.shape[-1] ** 0.5)

        # Apply attention mask if provided or create one
        if attn_mask is None:
            attn_mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
            attn_mask = attn_mask.to(device=x.device, dtype=x.dtype)

        attn_scores = attn_scores - 1e9 * (1 - attn_mask)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_cache)

        # Concatenate multiheaded attention results
        attn_output = attn_output.view(attn_output.shape[0], -1, x.shape[-1])

        assert attn_output.shape == x.shape, f"Assertion failed. Shapes: {x.shape}, {attn_output.shape}"

        return attn_output

def modify_transformer_attention_blocks(model, block_name:str, attn_layer_name:str, qkv_names:list[str], max_batch_size:int, max_seq_len:int, n_heads:int, head_dim:int):
    blocks = getattr(model, block_name)
    for block in blocks:
        attn_layer = getattr(block, attn_layer_name)  # Assuming the attribute is named 'ModifiedAttentionLayer'
        kv_cache = KVCache(max_batch_size, max_seq_len, n_heads, head_dim)
        modified_attention = ModifiedAttentionLayer(attn_layer, n_heads, max_seq_len, qkv_names, kv_cache)
        setattr(block, attn_layer_name, modified_attention)
    
    return model