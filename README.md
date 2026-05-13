# Dependency Parsing LXQ

This repository contains a course project on Chinese dependency parsing built around the Deep Biaffine Parser of Dozat and Manning. We start from a PyTorch implementation of graph-based biaffine dependency parsing, clean up the training harness, add logging and checkpointing, and run a controlled empirical study over word vectors, optimizers and encoders.

The accompanying paper is in `Paper/`. It reports fourteen training runs on an 8.3K-sentence Chinese dependency treebank. The main finding is that, at this data scale, the original BiLSTM encoder remains stronger than Transformer encoders trained from scratch. The best run is BiLSTM + Muon with warmup and cosine decay, reaching 86.93 dev UAS and 68.37 dev LAS. The largest SDPA encoder we trained reaches 83.92 UAS, suggesting that model capacity alone does not compensate for limited labeled data.

## What We Did

- Rebuilt the project into a more reproducible training workflow with checkpoints, JSON history logs, plots and run metadata.
- Compared pretrained word vector dimensionality and domain.
- Compared Adam, SGD and Muon optimizers.
- Added SDPA Transformer encoder variants while keeping the biaffine scorer and MST decoder fixed.
- Diagnosed two practical failure modes: post-norm Transformer without warmup, and an overly large Muon learning rate on small Transformer encoders.
- Released large training artifacts separately from GitHub.

## Repository Layout

```text
.
├── Paper/                                      # LaTeX paper and figures
├── PyTorch_Biaffine_Dependency_Parsing/        # Main codebase
│   ├── main.py                                 # CLI entry point for train/test mode
│   ├── trainer.py                              # Training, evaluation, checkpoint flow
│   ├── Config/config.cfg                       # Default experiment configuration
│   ├── Dataloader/                             # CoNLL parsing and instance construction
│   ├── DataUtils/                              # Alphabet, batching, embeddings, logging, checkpoints
│   ├── Model/Biaffine_Parsing/                 # Parser model, biaffine layers, MST decoding
│   ├── scripts/                                # Server launch scripts for Stage 2 runs
│   └── tests/                                  # Unit tests for added components
├── PROCESS.md                                  # Project progress log
├── PAPER.md                                    # Paper writing plan
├── requirements.txt                            # Python dependencies
└── LICENSE                                     # MIT license
```

## Key Entry Points

The main training entry is:

```bash
cd PyTorch_Biaffine_Dependency_Parsing
python main.py --config ./Config/config.cfg --device cuda:0 --train -p
```

Useful directions in the code:

- `main.py` parses CLI arguments, loads data, builds the parser and starts training.
- `trainer.py` owns the epoch loop, optimizer step, evaluation, early stopping and checkpoint saving.
- `Model/Biaffine_Parsing/Model.py` defines the encoder + MLP + biaffine scorer forward pass.
- `Model/Biaffine_Parsing/Parser.py` defines loss computation and parsing.
- `Model/Biaffine_Parsing/MST.py` decodes arc scores into valid dependency trees.
- `DataUtils/Logger.py` and `DataUtils/Checkpoint.py` provide experiment history and model artifact management.
- `scripts/launch_stage2_*.sh` show the server-side commands used for the encoder and optimizer ablations.

Compute-heavy experiments were run on server 4, an A800 machine with 8 x 80GB GPUs, CUDA 12.9, 1TB RAM and an `/essfs100` shared data volume.

## Released Artifacts

Large model artifacts are not stored in this GitHub repository. The full `Output/` directory, including checkpoints, logs, plots and run summaries, is available on Hugging Face:

[https://huggingface.co/kyksj/Dependency-Parsing](https://huggingface.co/kyksj/Dependency-Parsing)

The code repository is:

[https://github.com/kyksj-1/Dependency-Parsing-LXQ](https://github.com/kyksj-1/Dependency-Parsing-LXQ)

## Results Summary

| Configuration | Dev UAS | Dev LAS | Note |
|---|---:|---:|---|
| BiLSTM + Adam, 50 epochs | 86.43 | 67.65 | Stage-1 baseline |
| BiLSTM + Muon, 100 epochs | 86.93 | 68.37 | Best run |
| SDPA-9M + Adam, pre-norm + warmup | 82.00 | 63.55 | Recovered Transformer small |
| SDPA-Large + Muon | 83.92 | 65.60 | Largest encoder trained |
| BiLSTM + SGD | 17.66 | 1.58 | Negative control |

The headline interpretation is simple: on this small supervised treebank, BiLSTM's inductive bias is useful. Pretrained lexical coverage matters more than small differences among embedding domains. Muon can outperform Adam, but only with a schedule long enough for its orthogonalized updates to refine the model.

## Next Steps

Natural continuations include:

- Replace the custom `LSTMCell` loop with `nn.LSTM` for faster training while preserving comparable dropout behavior.
- Add a BERT-base-Chinese or MacBERT encoder to test whether pretrained contextual representations close the Transformer gap.
- Tune Muon learning rates separately for recurrent and Transformer parameters.
- Add a clean prediction CLI for parsing raw or CoNLL-formatted sentences with trained checkpoints.
- Run multiple seeds for the top configurations to estimate variance more rigorously.
- Test on a larger labeled treebank or a Universal Dependencies Chinese split.

## License

This project is released under the MIT License. See `LICENSE`.
