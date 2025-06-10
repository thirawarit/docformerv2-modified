import math
from typing import Optional, Tuple

import torch
from torch import nn


class SimpleVisualFeature(nn.Module):

    def __init__(
            self, 
            patch_size: int, 
            temporal_patch_size: int, 
            channel: int, 
            hidden_size: int,
            merge_size: int = 2,
            max_position_embeddings: int = 512,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.channel = channel
        self.merge_size = merge_size
        self.in_channels = channel
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        kernel_size = [patch_size, patch_size]
        self.conv2d = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        self.relu = nn.ReLU()
        self.linear_layer = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size, 
            bias=True
        )

    def forward(self, hidden_states: torch.Tensor, image_grid_hw: torch.Tensor):
        bs, _ = image_grid_hw.shape
        grid_h, grid_w = image_grid_hw[0]
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.view(-1, self.in_channels * self.patch_size * self.patch_size)
        # hidden_states = hidden_states[:self.max_position_embeddings]
        hidden_states = hidden_states.reshape(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.conv2d(hidden_states)
        hidden_states = hidden_states.view(bs, grid_h * grid_w, self.hidden_size)
        hidden_states = hidden_states[:, :self.max_position_embeddings, :]
        # print(hidden_states.size())
        hidden_states = self.relu(hidden_states)
        hidden_states = self.linear_layer(hidden_states)
        return hidden_states
    

class DocformerV2VLEncoder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.vision_config = config.vision_config
        self.simple_visual_feat = SimpleVisualFeature(
            patch_size=self.vision_config.patch_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            channel=self.vision_config.in_channels,
            hidden_size=self.vision_config.hidden_size,
            max_position_embeddings=self.vision_config.max_position_embeddings,
        )

        self.rot_pos_emb = SpatialRotaryPositionalEmbedding(config)

    def forward(self, x, image_grid_hw):
        x = self.simple_visual_feat(x, image_grid_hw)
        batch, seq_patches, _ = x.size()
        cache_position = torch.arange(0, seq_patches)
        position_ids = cache_position.view(1, -1).expand(batch, -1).to(dtype=torch.float32)
        cos, sin = self.rot_pos_emb(x, position_ids)
        x = apply_rope_spatial(x, cos, sin)
        # print(x.shape)
        return x
    

class SpatialRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.rope_init_fn = self._compute_default_rope_parameters
        inv_freq, self.attn_scaling = self.rope_init_fn(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _compute_default_rope_parameters(self, config):
        base = config.rope_theta
        partial_rotary_factor = 1.0
        dim = int(config.hidden_size * partial_rotary_factor)

        attn_factor = 1.0
        # Compute the inverse frequencies.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        return inv_freq, attn_factor
    
    @torch.no_grad()
    def forward(self, inputs, position_ids):
        inv_freq_expanded = self.inv_freq[torch.newaxis,:,torch.newaxis].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:,torch.newaxis,:]
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = inputs.device.type
        device_type = device_type if isinstance(device_type, str) and device_type == "mps" else "cpu"
        inv_freq_expanded = inv_freq_expanded.to(device_type)
        position_ids_expanded = position_ids_expanded.to(device_type)
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attn_scaling
        sin = sin * self.attn_scaling

        return cos, sin
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_spatial(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_dtype = x.dtype
    x = x.float()
    cos, sin = cos.float(), sin.float()
    # | cos_ -sin_ | | x_1 |
    # | sin_  cos_ | | x_2 |
    # x_1*cos_ + x_2*(-sin_)
    # x_1*sin_ + x_2*cos_
    x_emb = (x * cos) + (rotate_half(x) * sin)
    x_emb = x_emb.to(x_dtype)
    return x_emb


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_size = config.hidden_size
        self.coordinate_size = coordinate_size = config.coordinate_size
        self.coordinate_dim = coordinate_dim = config.coordinate_dim
    
        # RoPE
        self.rot_pos_emb = SpatialRotaryPositionalEmbedding(config)

        # t: top, b: bottom, l: left, r: right, ct: centroid
        # pos_emb: embedding of position, v: vision, t: text
        self.x_tl_pos_emb_v = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.x_br_pos_emb_v = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.w_pos_emb_v = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.x_tl_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_bl_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_tr_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_br_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_ct_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)

        self.y_tl_pos_emb_v = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.y_br_pos_emb_v = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.h_pos_emb_v = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.y_tl_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_bl_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_tr_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_br_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_ct_distance_to_prev_emb_v = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)

        self.x_tl_pos_emb_t = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.x_br_pos_emb_t = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.w_pos_emb_t = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.x_tl_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_bl_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_tr_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_br_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.x_ct_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)

        self.y_tl_pos_emb_t = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.y_br_pos_emb_t = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.h_pos_emb_t = nn.Embedding(num_embeddings=coordinate_size, embedding_dim=coordinate_dim)
        self.y_tl_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_bl_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_tr_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_br_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)
        self.y_ct_distance_to_prev_emb_t = nn.Embedding(num_embeddings=2*coordinate_size + 1, embedding_dim=coordinate_dim)


    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor):
        
        batch, seq_len, num_feat = x_features.shape
        cache_position = torch.arange(seq_len)
        position_ids = cache_position.view(1, -1).expand(batch, -1).to(dtype=torch.float32)
        clone_x_features = x_features.clone().to(dtype=torch.float32)
        cos, sin = self.rot_pos_emb(clone_x_features, position_ids)

        # Clamping and adding a bias for handling negative values
        x_features[:,:,3:] = torch.clamp(x_features[:,:,3:], -self.coordinate_size, self.coordinate_size)
        x_features[:,:,3:] += self.coordinate_size

        y_features[:,:,3:] = torch.clamp(y_features[:,:,3:], -self.coordinate_size, self.coordinate_size)
        y_features[:,:,3:] += self.coordinate_size

        x_tl_pos_emb_v = self.x_tl_pos_emb_v(x_features[:,:,0])
        x_br_pos_emb_v = self.x_br_pos_emb_v(x_features[:,:,1])
        w_pos_emb_v = self.w_pos_emb_v(x_features[:,:,2])
        x_tl_distance_to_prev_emb_v = self.x_tl_distance_to_prev_emb_v(x_features[:,:,3])
        x_bl_distance_to_prev_emb_v = self.x_tl_distance_to_prev_emb_v(x_features[:,:,4])
        x_tr_distance_to_prev_emb_v = self.x_tl_distance_to_prev_emb_v(x_features[:,:,5])
        x_br_distance_to_prev_emb_v = self.x_tl_distance_to_prev_emb_v(x_features[:,:,6])
        x_ct_distance_to_prev_emb_v = self.x_tl_distance_to_prev_emb_v(x_features[:,:,7])

        y_tl_pos_emb_v = self.y_tl_pos_emb_v(y_features[:,:,0])
        y_br_pos_emb_v = self.y_br_pos_emb_v(y_features[:,:,1])
        h_pos_emb_v = self.h_pos_emb_v(y_features[:,:,2])
        y_tl_distance_to_prev_emb_v = self.y_tl_distance_to_prev_emb_v(y_features[:,:,3])
        y_bl_distance_to_prev_emb_v = self.y_tl_distance_to_prev_emb_v(y_features[:,:,4])
        y_tr_distance_to_prev_emb_v = self.y_tl_distance_to_prev_emb_v(y_features[:,:,5])
        y_br_distance_to_prev_emb_v = self.y_tl_distance_to_prev_emb_v(y_features[:,:,6])
        y_ct_distance_to_prev_emb_v = self.y_tl_distance_to_prev_emb_v(y_features[:,:,7])

        x_emb_v = torch.cat(
            [
                x_tl_pos_emb_v,
                x_br_pos_emb_v,
                w_pos_emb_v,
                x_tl_distance_to_prev_emb_v,
                x_bl_distance_to_prev_emb_v,
                x_tr_distance_to_prev_emb_v,
                x_br_distance_to_prev_emb_v,
                x_ct_distance_to_prev_emb_v,
            ],
            dim=-1
        )

        y_emb_v = torch.cat(
            [
                y_tl_pos_emb_v,
                y_br_pos_emb_v,
                h_pos_emb_v,
                y_tl_distance_to_prev_emb_v,
                y_bl_distance_to_prev_emb_v,
                y_tr_distance_to_prev_emb_v,
                y_br_distance_to_prev_emb_v,
                y_ct_distance_to_prev_emb_v,
            ],
            dim=-1
        )

        v_bar_s = apply_rope_spatial(x_emb_v + y_emb_v, cos, sin)


        x_tl_pos_emb_t = self.x_tl_pos_emb_t(x_features[:,:,0])
        x_br_pos_emb_t = self.x_br_pos_emb_t(x_features[:,:,1])
        w_pos_emb_t = self.w_pos_emb_t(x_features[:,:,2])
        x_tl_distance_to_prev_emb_t = self.x_tl_distance_to_prev_emb_t(x_features[:,:,3])
        x_bl_distance_to_prev_emb_t = self.x_tl_distance_to_prev_emb_t(x_features[:,:,4])
        x_tr_distance_to_prev_emb_t = self.x_tl_distance_to_prev_emb_t(x_features[:,:,5])
        x_br_distance_to_prev_emb_t = self.x_tl_distance_to_prev_emb_t(x_features[:,:,6])
        x_ct_distance_to_prev_emb_t = self.x_tl_distance_to_prev_emb_t(x_features[:,:,7])

        y_tl_pos_emb_t = self.y_tl_pos_emb_t(y_features[:,:,0])
        y_br_pos_emb_t = self.y_br_pos_emb_t(y_features[:,:,1])
        h_pos_emb_t = self.h_pos_emb_t(y_features[:,:,2])
        y_tl_distance_to_prev_emb_t = self.y_tl_distance_to_prev_emb_t(y_features[:,:,3])
        y_bl_distance_to_prev_emb_t = self.y_tl_distance_to_prev_emb_t(y_features[:,:,4])
        y_tr_distance_to_prev_emb_t = self.y_tl_distance_to_prev_emb_t(y_features[:,:,5])
        y_br_distance_to_prev_emb_t = self.y_tl_distance_to_prev_emb_t(y_features[:,:,6])
        y_ct_distance_to_prev_emb_t = self.y_tl_distance_to_prev_emb_t(y_features[:,:,7])

        x_emb_t = torch.cat(
            [
                x_tl_pos_emb_t,
                x_br_pos_emb_t,
                w_pos_emb_t,
                x_tl_distance_to_prev_emb_t,
                x_bl_distance_to_prev_emb_t,
                x_tr_distance_to_prev_emb_t,
                x_br_distance_to_prev_emb_t,
                x_ct_distance_to_prev_emb_t,
            ],
            dim=-1
        )

        y_emb_t = torch.cat(
            [
                y_tl_pos_emb_t,
                y_br_pos_emb_t,
                h_pos_emb_t,
                y_tl_distance_to_prev_emb_t,
                y_bl_distance_to_prev_emb_t,
                y_tr_distance_to_prev_emb_t,
                y_br_distance_to_prev_emb_t,
                y_ct_distance_to_prev_emb_t,
            ],
            dim=-1
        )

        t_bar_s = apply_rope_spatial(x_emb_t + y_emb_t, cos, sin)

        return v_bar_s, t_bar_s
    

class LanguageFeatureExtractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_vector = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=config.pad_token_id)

    def forward(self, x):
        return self.embedding_vector(x)
    

class ExtractFeatures(nn.Module):

    '''
    Inputs: dictionary
    Output: v_bar, t_bar, v_bar_s, t_bar_s
    '''

    def __init__(self, config):
        super().__init__()
        self.visual_feature = DocformerV2VLEncoder(config)
        self.language_feature = LanguageFeatureExtractor(config)
        self.spatial_feature = SpatialFeatureExtractor(config)

    def forward(self, encoding):
      
        image = encoding['pixel_values']
        image_grid_hw = encoding['image_grid_hw']
            
        language = encoding['input_ids']
        x_feature = encoding['x_features']
        y_feature = encoding['y_features']

        v_bar = self.visual_feature(image, image_grid_hw)
        t_bar = self.language_feature(language)

        v_bar_s, t_bar_s = self.spatial_feature(x_feature, y_feature)
        
        return v_bar, t_bar, v_bar_s, t_bar_s
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos, sin = cos.float(), sin.float()

    # | cos_ -sin_ | | x_1 |
    # | sin_  cos_ | | x_2 |

    # x_1*cos_ + x_2*(-sin_)
    # x_1*sin_ + x_2*cos_

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class AttentionRotaryPositionalEmbedding(SpatialRotaryPositionalEmbedding):
    def __init__(self, config):
        super().__init__(config)
    
    def _compute_default_rope_parameters(self, config):
        base = config.rope_theta
        partial_rotary_factor = 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        return inv_freq, attention_factor
    

class RelativePosition(nn.Module):
    def __init__(self, head_dim: int, max_relative_position: int, max_seq_length: int):
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty(2*max_relative_position + 1, head_dim))
        nn.init.xavier_uniform_(self.embedding_table)

        max_q_len = torch.arange(max_seq_length)
        max_k_len = torch.arange(max_seq_length)
        distance_mat = max_q_len[torch.newaxis, :] - max_k_len[:, torch.newaxis]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        self.final_mat = torch.LongTensor(final_mat)

    def forward(self, len_q, len_k):
        embeddings = self.embedding_table[self.final_mat[:len_q, :len_k]]
        return embeddings


class AttentionWithSpatialFeature(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.hidden_dropout_prob
        self.rope_scaling = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.relative_positions_k = RelativePosition(config.head_dim, config.max_relative_positions, config.max_seq_length)
        self.relative_positions_q = RelativePosition(config.head_dim, config.max_relative_positions, config.max_seq_length)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_spatial_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_spatial_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.rotary_emb = AttentionRotaryPositionalEmbedding(config=config)
        self.spatial_rotary_emb = AttentionRotaryPositionalEmbedding(config=config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            spatial_hidden_states: torch.Tensor,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            spatial_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if hidden_states.size() != spatial_hidden_states.size():
            raise ValueError(
                "Size of hidden_states and spatial_hidden_states are not equal "
                f"hidden_states.size() != spatial_hidden_states.size() "
                f"{hidden_states.size()} != {spatial_hidden_states.size()} "
            )
        
        bs, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bs, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bs, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bs, q_len, -1, self.head_dim).transpose(1, 2)

        # if position_embeddings is None:
        #     cache_position = torch.arange(0, q_len)
        #     position_ids = cache_position.view(1, -1).expand(bs, -1)
        #     position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # cos, sin = position_embeddings
        # query_states, key_states = apply_multimodal_rope(query_states, value_states, cos, sin)

        rel_pos_embed = self.relative_positions_k(q_len, q_len)

        spatial_query_states = self.q_spatial_proj(spatial_hidden_states)
        spatial_key_states = self.k_spatial_proj(spatial_hidden_states)

        spatial_query_states = spatial_query_states.view(bs, q_len, -1, self.head_dim).transpose(1, 2)
        spatial_key_states = spatial_key_states.view(bs, q_len, -1, self.head_dim).transpose(1, 2)

        # if spatial_position_embeddings is None:
        #     cache_position = torch.arange(0, q_len)
        #     position_ids = cache_position.view(1, -1).expand(bs, -1)
        #     spatial_position_embeddings = self.spatial_rotary_emb(hidden_states, position_ids)
        # cos, sin = spatial_position_embeddings
        # spatial_query_states, spatial_key_states = apply_multimodal_rope(spatial_query_states, spatial_key_states, cos, sin)

        if query_states.device.type == "mps":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            spatial_query_states = spatial_query_states.contiguous()
            spatial_key_states = spatial_key_states.contiguous()

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        query_states_reshape = query_states.reshape(bs * self.num_heads, q_len, self.head_dim)
        relative_attn_weights = torch.matmul(query_states_reshape.unsqueeze(dim=1), rel_pos_embed.transpose(-2, -1).unsqueeze(dim=0)) # (bs*nh sq_rp sq_x sk_rp)
        relative_attn_weights = torch.diagonal(relative_attn_weights, dim1=-2, dim2=-1).reshape(bs, self.num_heads, q_len, q_len)

        spatial_attn_weights = torch.matmul(spatial_query_states, spatial_key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_weights = attn_weights + relative_attn_weights + spatial_attn_weights

        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = nn.functional.softmax(input=attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(input=attn_weights, p=self.attention_dropout, training=self.training)
        # atten_weight -> (bs, nh, sq, sk), where sk == sv
        # value_states -> (bs, nh, sv, hd), where sv == sk
        # attn_output -> (bs, nh, sq, hd)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bs, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bs, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # # attn_output -> (bs, sq, nh, hd)
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # # attn_output -> (bs, sq, nh * hd)
        # attn_output = attn_output.reshape(bs, q_len, -1)

        # # attn_output -> (bs, sq, hidden_size)
        # attn_output = self.o_proj(attn_output)

        # if not output_attentions:
        #     attn_weights = None

        return attn_output, attn_weights
    
class MultiModalAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.text_attn_layer = AttentionWithSpatialFeature(config)
        self.img_attn_layer = AttentionWithSpatialFeature(config)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self, 
            text_feat, 
            img_feat, 
            text_spatial_feat, 
            img_spatial_feat,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            spatial_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        bs, q_len, _ = text_feat.size()
        text_attn_output, text_attn_weights = self.text_attn_layer(
            text_feat, 
            text_spatial_feat,
            # output_attentions=output_attentions,
            # cache_position=cache_position,
            # position_embeddings=position_embeddings,
            # spatial_position_embeddings=spatial_position_embeddings,
        )
        img_attn_output, img_attn_weights = self.img_attn_layer(
            img_feat, 
            img_spatial_feat,
            # output_attentions=output_attentions,
            # cache_position=cache_position,
            # position_embeddings=position_embeddings,
            # spatial_position_embeddings=spatial_position_embeddings,
        )
        attn_output = text_attn_output + img_attn_output
        # attn_output -> (bs, sq, nh, hd)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output -> (bs, sq, nh * hd)
        attn_output = attn_output.reshape(bs, q_len, -1)

        # attn_output -> (bs, sq, hidden_size)
        attn_output = self.o_proj(attn_output)

        # if not output_attentions:
        #     text_attn_weights = None
        #     img_attn_weights = None

        return attn_output, text_attn_weights, img_attn_weights
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class DocFormerV2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiModalAttentionLayer(config)
        self.ffn = FeedForward(
            config.hidden_size, 
            config.hidden_size * config.intermediate_ff_size_factor,
            config.hidden_dropout_prob,
        )

    def forward(
            self, 
            text_feat, 
            img_feat, 
            text_spatial_feat, 
            img_spatial_feat,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            spatial_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # residual = text_feat + img_feat + text_spatial_feat + img_spatial_feat

        attn_output, text_attn_weights, img_attn_weights = self.self_attn(
            text_feat, 
            img_feat, 
            text_spatial_feat, 
            img_spatial_feat,
            # output_attentions,
            # cache_position,
            # position_embeddings,
            # spatial_position_embeddings
        )

        # attn_output = attn_output + residual

        output = attn_output
        output = self.ffn(output)

        return output

class DocformerV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.extract_feature = ExtractFeatures(config)
        self.encoder = nn.ModuleList(
            [DocFormerV2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
            self,
            encodings,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            spatial_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        v_bar, t_bar, v_bar_s, t_bar_s = self.extract_feature(encodings)
        for head_layer in self.encoder:
            output = head_layer(
                t_bar, 
                v_bar, 
                t_bar_s,
                v_bar_s, 
                # output_attentions,
                # cache_position,
                # position_embeddings,
                # spatial_position_embeddings
            )
            t_bar = output
        output = self.dropout(output)
        return output