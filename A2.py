import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

class A2ModelConfig(PretrainedConfig):
    """
    Configuration object that stores all hyperparameters defining the Transformer language model.
    Inheriting from HuggingFace's PretrainedConfig allows the model to easily save/load 
    configurations using standard HF methods (e.g., from_pretrained).
    """
    def __init__(self, vocab_size=10000, hidden_size=256, intermediate_size=512, num_attention_heads=8, 
                 num_hidden_layers=4,
                 rope_theta=10000.0, hidden_act='silu', max_position_embeddings=2048, rms_norm_eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps # Small value to prevent division by zero in RMSNorm
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta # Base frequency for Rotary Position Embeddings (RoPE)
        self.hidden_act = hidden_act # Activation function (SiLU is used for SwiGLU)
        self.intermediate_size = intermediate_size # Hidden dimension size inside the MLP
        self.num_hidden_layers = num_hidden_layers # Number of Transformer blocks


class A2MLP(nn.Module):
    """
    The MLP (Feed-Forward) layer of the Transformer. 
    Uses the SwiGLU architecture (widely used in LLaMA, OLMo, etc.) instead of the standard ReLU MLP.
    """
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # OLMo 2 / Llama style SwiGLU components: 
        # Modern LLMs typically remove bias terms in linear layers for better training stability.
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU() # Sigmoid Linear Unit (SiLU), also known as Swish

    def forward(self, hidden_states):
        # SwiGLU formulation: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        # The 'gate' controls the information flow from the 'up' projection before projecting back down.
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class A2RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    A computationally cheaper alternative to standard LayerNorm. It only scales the variance
    and does not recenter the mean, which has been shown to perform equally well in LLMs.
    """
    def __init__(self, config):
        super().__init__()
        self.eps = config.rms_norm_eps
        # Learnable scaling parameter (gamma), initialized to ones. Size equals hidden_size.
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        # Calculate Variance. 
        # Cast to float32 before squaring to avoid numerical overflow/underflow issues in half-precision (fp16).
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # Normalize the hidden states (x / sqrt(Var + eps))
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # Apply learnable parameter and cast back to original dtype
        # Apply the learnable weight parameter and cast back to the original data type (e.g., float16 or bfloat16)
        return self.weight * hidden_states.to(self.weight.dtype)


class A2Attention(nn.Module):
    """
    The Multi-Head Attention (MHA) layer of the Transformer. 
    Uses scaled dot-product attention with causal masking and Rotary Position Embeddings (RoPE).
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Dimension of each individual attention head
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Q, K, V, and Output projections without bias
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Specific to OLMo 2: Layer normalization on Query and Key BEFORE reshaping
        self.q_norm = A2RMSNorm(config)
        self.k_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # b: batch_size, m: sequence_length, d: hidden_size
        b, m, d = hidden_states.shape
        
        # 1. Linear Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. Apply Q/K normalizers (OLMo 2 feature)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 3. Reshape for Multi-Head Attention: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = q.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. Apply Rotary Position Embeddings (RoPE)
        # This injects relative positional information directly into the attention mechanism.
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        
        # 5. Scaled Dot-Product Attention with Causal Mask
        # PyTorch built-in implements this highly efficiently
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, is_causal=True
        )
        
        # 6. Re-assemble heads and Output Projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, m, d)
        return self.o_proj(attn_out)


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.self_attn = A2Attention(config)
        self.mlp = A2MLP(config)
        self.input_layernorm = A2RMSNorm(config)
        self.post_attention_layernorm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # Block 1: Attention with Pre-Norm and Residual Connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, rope_rotations)
        hidden_states = residual + hidden_states
        
        # Block 2: MLP with Pre-Norm and Residual Connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        # 1. Embedding Layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 2. Positional RoPE
        self.rotary_emb = A2RotaryEmbedding(config)
        
        # 3. Stack of Decoder Layers
        self.layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 4. Final Normalization and Output Projection (No bias)
        self.norm = A2RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # PyTorch loss function handling Next-Token shift mapping automatically
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

        # Initialize weights (HuggingFace standard method)
        self.post_init()

    def forward(self, input_ids, labels=None):
        # Forward process
        hidden_states = self.embed_tokens(input_ids)
        
        # RoPE uses sequence length (x.shape[1] internally), input_ids matches this requirement
        rope_rotations = self.rotary_emb(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Same Shift-Loss computation as Assignment 1
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_func(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])   
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

import torch

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=5, device='cuda'):
    """
    Top-K sampling algorithm for text generation.
    """
    model.eval()
    
    encoded = tokenizer([prompt], return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    
    generated_text = []

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            
            next_token_logits = outputs.logits[0, -1, :] / temperature
            
            topk_logits, topk_indices = torch.topk(next_token_logits, topk)
            
            probs = torch.nn.functional.softmax(topk_logits, dim=-1)
            
            dist = torch.distributions.Categorical(probs)
            next_token_idx_in_topk = dist.sample()
            
            next_token_id = topk_indices[next_token_idx_in_topk].unsqueeze(0).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            generated_token = next_token_id.item()
            if generated_token == tokenizer.eos_token_id:
                break
                
            generated_text.append(generated_token)
            
    decoded_words = [tokenizer.int_to_str[i] for i in generated_text]
    return prompt + " " + " ".join(decoded_words)
