# Tokenization and Language Identity (DAT09-06)

## Question

Holding data, architecture and training budget fixed, which tokenization method (byte, BPE, unigram, character n-gram) best preserves language identity? Focus on short sentences and closely related language clusters.

## Approach

One architecture (embedding table → mean pooling → linear head), trained separately per tokenizer. Tokenizer configuration (subword vocab size, n-gram window style) and sample size per language are set via cheap empirical screens before any model is trained. Each method is run with one shared hyperparameter config and with its own tuned config, to separate the tokenization effect from tuning effort.

Full rationale: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Data

Tatoeba sentences, 20 languages; sentences/language set by the screening step below.

- **Romance**: Spanish, Portuguese, Italian, French, Catalan, Galician
- **East Slavic**: Russian, Ukrainian, Belarusian
- **Distinct**: English, German, Japanese, Mandarin, Arabic, Finnish, Turkish, Serbian, Greek, Hindi, Kabyle

Source: [https://tatoeba.org/en/downloads](https://tatoeba.org/en/downloads). Each sentence has a unique id and an ISO 639-3 code. Downloaded 16/11/2026, 15:55 UTC+1.

## Baselines

- Character n-gram Naive Bayes (overlapping trigrams)
- PCA at the same dimension as the learned embeddings

## Pipeline

1. `raw/` — Tatoeba as downloaded, never modified
2. `screening/sample_size/` — learning-curve check to set sentences/language
3. `splits/` — train/val/test (70/15/15, stratified per language, fixed seed)
4. `screening/tokenizer_config/` — vocab size and n-gram window style, checked on the training split
5. `tokenizers/` — one per method, trained on train split only
6. `tokenized_data/<method>/`
7. `runs/<method>/<arm>/seed_<n>/` — weights, curves, predictions (`<arm>` = shared or tuned)
8. `results/` — comparison tables and plots

## Model and training protocol

- One architecture per method; only the embedding table differs
- Two hyperparameter arms per method: shared (dim=128, lr=1e-3) and tuned (small grid, searched on validation)
- Early stopping on validation loss, same patience and max epochs
- Per-tokenizer max length covering ~99% of train sentences
- At least 3 seeds per method per arm; report mean and std
- Test set used once

## Evaluation

- Accuracy and macro-F1, overall and by language cluster and sentence length
- Paired bootstrap or McNemar on shared test predictions
- Tokenizer descriptors: fertility, sequence length, parameters, time
- Gap between shared and tuned arms, reported as its own result

## Scope note

Conclusions apply to this architecture; a second architecture is future work.
