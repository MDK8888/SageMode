import torch
import torch.nn.functional as F
import unittest
from ..KVCacheExp import *

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.W_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_v = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, **kwargs):
        batch_size, seq_len, _ = x.size()

        # Linear transformations for query, key, and value
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose dimensions for matrix multiplication
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and concatenate attention outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return attn_output

class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(TransformerBlock, self).__init__()

        # Self-Attention Layer
        self.self_attention = SelfAttentionLayer(hidden_dim, n_heads)

        # Fully connected feedforward layer
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim)
        )

        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, **kwargs):
        # Self-Attention
        attn_output = self.self_attention(x, **kwargs)
        x = x + attn_output
        x = self.norm1(x)

        # Fully connected feedforward layer
        ff_output = self.feedforward(x)
        x = x + ff_output
        x = self.norm2(x)

        return x

class TransformerModel(torch.nn.Module):
    def __init__(self, num_blocks, hidden_dim, n_heads):
        super(TransformerModel, self).__init__()

        self.blocks = torch.nn.ModuleList([TransformerBlock(hidden_dim, n_heads) for _ in range(num_blocks)])

    def forward(self, x, **kwargs):
        for block in self.blocks:
            x = block(x, **kwargs)
        return x

class TestAttentionModule(unittest.TestCase):

    def test_forward_pass(self):
        num_blocks = 3
        hidden_dim = 2048
        n_heads = 8
        batch_size = 64
        seq_len = 8


        model = TransformerModel(num_blocks, hidden_dim, n_heads)
        input = torch.rand((batch_size, seq_len, hidden_dim))
        output = model(input)
        self.assertEqual(output.shape, input.shape)
    
    def test_kv_cache(self):
        hidden_dim = 2048
        n_heads = 8
        batch_size = 64
        seq_len = 8
        dtype=torch.float32

        cache = KVCache(max_batch_size=batch_size, max_seq_length=seq_len, n_heads=n_heads, head_dim=hidden_dim, dtype=dtype)
        input_pos = torch.tensor([0, 1, 2, 3])

        k_val = torch.rand((batch_size, n_heads, len(input_pos), hidden_dim), dtype=dtype)
        v_val = torch.rand((batch_size, n_heads, len(input_pos), hidden_dim), dtype=dtype)

        k_cache, v_cache = cache.update(input_pos, k_val, v_val)

        self.assertTrue(torch.equal(k_cache[:, :, input_pos], k_val))
        self.assertTrue(torch.equal(v_cache[:, :, input_pos], v_val))
    
    def test_model_update(self):
        num_blocks = 3
        hidden_dim = 2048
        n_heads = 1
        head_dim = hidden_dim // n_heads 
        batch_size = 1
        seq_len = 8

        model = TransformerModel(num_blocks, hidden_dim, n_heads)
        model = modify_transformer_attention_blocks(model, "blocks", "self_attention", ["W_q", "W_k", "W_v"], max_batch_size=batch_size, max_seq_len=seq_len, n_heads=n_heads, head_dim=head_dim)
        input = torch.rand((batch_size, 5, hidden_dim))
        output = model(input, start_pos=1)
        self.assertEqual(output.shape, input.shape)

if __name__ == "__main__":
    unittest.main()


