from dataclasses import dataclass, field


@dataclass
class Tensor:
    shape: list[int]
    dtype: str
    sparse: bool = False
    is_activation: bool = True
    nbits: int = field(init=False)

    def __post_init__(self) -> None:
        bitwidths = {"bit": 1, "int8": 8, "fp16": 16}
        if self.dtype not in bitwidths:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        self.sparse = self.sparse or self.dtype == "bit"
        self.nbits = bitwidths[self.dtype]
        self.tensor = None


@dataclass
class FC:
    name: str
    output_dim: int
    input_dim: int
    sequence_length: int
    num_bits: int = 4
    algorithm: str = "aqlm"
    batch_size: int = 1
    is_expert: bool = False
    num_experts: int = 1
    top_k: int = 1
    is_shared_expert: bool = False

    def __post_init__(self) -> None:
        self.bitwidth = self.num_bits
        self.activation_tensor = Tensor([self.sequence_length, self.input_dim], "fp16", sparse=False)
        self.weight = Tensor([self.input_dim, self.output_dim], "fp16", sparse=False)
        self.output_tensor = Tensor([self.sequence_length, self.output_dim], "fp16", sparse=False)
        if self.algorithm == "aqlm":
            self.num_codebooks = self.num_bits
            self.vector_size = 8
            self.num_entries = 256
            self.codebooks = Tensor([self.num_codebooks, self.vector_size, self.num_entries], "fp16", sparse=False)
            self.index = Tensor([self.input_dim // self.vector_size, self.output_dim, self.num_codebooks], "int8", sparse=False)
        elif self.algorithm == "gptvq":
            self.num_codebooks = 1
            self.vector_size = 4
            self.num_entries = 256
            self.codebooks = Tensor([self.num_codebooks, self.vector_size, self.num_entries], "fp16", sparse=False)
            self.index = Tensor([self.input_dim // self.vector_size, self.output_dim], "int8", sparse=False)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


@dataclass
class Attention:
    name: str
    dim: int
    sequence_length: int
    bitwidth: int
    num_kv_heads: int | None = None
    num_attention_heads: int | None = None
    head_dim: int | None = None

    def __post_init__(self) -> None:
        self.head_dim = self.head_dim or 128
        self.num_attention_heads = self.num_attention_heads or self.dim // self.head_dim
        self.num_kv_heads = self.num_kv_heads or self.num_attention_heads
        self.act_q_tensor = Tensor([self.sequence_length, self.dim], "int8", sparse=True)
        self.act_k_tensor = Tensor([self.sequence_length * self.bitwidth, self.dim], "bit", sparse=True)
        self.act_v_tensor = Tensor([self.dim * self.bitwidth, self.sequence_length], "bit", sparse=True)
        self.attn_tensor = Tensor([self.sequence_length, self.sequence_length], "int8", sparse=True)
        self.output_tensor = Tensor([self.sequence_length, self.dim], "fp16", sparse=False)


@dataclass
class SFU:
    name: str
    sequence_length: int
    dim: int
    op_type: str
    nbits: int = 16
    is_expert: bool = False
    num_experts: int = 1
    top_k: int = 1
