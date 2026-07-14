from dataclasses import dataclass
from pathlib import Path


# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GPTConfig:
    seed = 42
    num_workers = 4
    pin_memory = True
    prefetch_factor = 2
    persistent_workers = True

    save_every_epoch = True
    save_best_model = True

    log_interval = 100

    resume = True

    max_steps = None


    # ======================
    # Data
    # ======================

    vocab_size: int = 32000
    max_seq_len: int = 256

    # ======================
    # Model
    # ======================

    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1

    # ======================
    # Training
    # ======================

    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10

    # ======================
    # Paths
    # ======================

    tokenizer_path: Path = (
        PROJECT_ROOT / "data" / "tokenizer" / "persian.model"
    )

    train_file: Path = (
        PROJECT_ROOT / "data" / "processed" / "clean_corpus.txt"
    )

    checkpoint_dir: Path = (
        PROJECT_ROOT / "checkpoints"
    )

    # ======================
    # Device
    # ======================

    device: str = "cuda"


# Default config object
config = GPTConfig()