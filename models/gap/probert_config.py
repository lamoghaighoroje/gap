from pytorch_pretrained_bert.modeling import BertConfig
from transformers import PretrainedConfig

class ProBertConfig(PretrainedConfig,BertConfig):
    model_type = "probert"
    def __init__(
        self,
        vocab_size=30522,
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=512,
        num_attention_heads=16,
        num_hidden_layers=24,
        type_vocab_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
