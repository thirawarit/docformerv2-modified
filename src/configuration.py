class DocformerV2VisionConfig(object):
    patch_size = 14
    temporal_patch_size = 2
    hidden_size = 768
    in_channels = 3
    max_position_embeddings = 512

class DocformerV2Config(object):
    vision_config = DocformerV2VisionConfig
    hidden_size = 768
    vocab_size = 250100
    pad_token_id = 0
    coordinate_size = 1024
    max_seq_length = 512
    num_spatial_features = 8
    num_attention_heads = 8
    num_hidden_layers = 12
    coordinate_dim = hidden_size // num_spatial_features
    head_dim = hidden_size // num_attention_heads
    max_relative_positions = 8
    rope_theta = 1.
    hidden_dropout_prob = 0.1
    intermediate_ff_size_factor = 4