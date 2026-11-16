# Methodology notes

Design decisions and reasoning, by topic.

## Language set

- Romance: `spa` (444k), `por` (446k), `ita` (993k), `fra` (731k), `cat` (11k), `glg` (8.2k)
- East Slavic: `rus` (1.24M), `ukr` (189k), `bel` (14.4k)
- Distinct: `eng` (2.04M), `deu` (787k), `jpn` (249k), `cmn` (89k), `ara` (69k), `fin` (155k), `tur` (750k), `srp` (52k), `ell` (43k), `hin` (16k), `kab` (794k)

(Counts are total sentences available in the raw download, not sample size.)

20 languages: two clusters at different genetic distances (Romance, East
Slavic) plus a distinct set spanning multiple families, branches, and
scripts, so the project can compare tokenization effects within vs. across
relatedness.

`srp` (South Slavic) and `kab` (Afro-Asiatic, Berber branch) sit in the
distinct set but share a top-level family with a cluster member
(`rus`/`ukr`/`bel`) or another distinct-set language (`ara`), respectively,
at a different branch. Tracked during analysis: if either patterns closer
to its relative than to the rest of the distinct set, that's a family-level
signal worth reporting.

`glg` (8.2k total sentences) is the binding ceiling on sample size across
the set.

## Sample size per language

Chosen via a learning-curve screen, run on throwaway samples before the
real split exists.

- Grid: 1,000 → 2,000 → 4,000 (each a doubling), with 6,000 as a
  conditional check only if 4,000 hasn't stabilized.
- At each N: draw a stratified sample, split internally into a throwaway
  train/held-out pair, compute NB validation accuracy and mutual
  information between token identity and language label; bootstrap within
  each N for a variance estimate.
- Stopping rule: smallest N where accuracy changes less than 1 point when N
  doubles, and bootstrap std stays within ~1 point.
- Stays well under `glg`'s 8.2k ceiling at every grid point.

## Train/val/test split

70/15/15, stratified per language, fixed seed, computed once at the chosen
N and shared by every tokenization method — the main control for isolating
the tokenization effect.

## Tokenization methods

Byte, BPE, unigram, character n-gram — four methods, one architecture; only
the embedding table differs between them.

### Subword vocabulary size (BPE / unigram)

One shared multilingual vocabulary across all 20 languages (a per-language
vocab would mean a separate embedding table per language per method,
breaking the fixed-architecture design).

Size (4,000 vs. 8,000) is screened on the real training split before
committing:

1. Train BPE/unigram at both sizes.
2. Encode train and held-out sentences with each.
3. Compare train vs. held-out: fertility gap, rare-piece usage rate,
   byte-fallback rate.
4. Pick the smaller vocab unless the larger shows no meaningful
   degradation.

### Character n-gram tokenization

Single order, n=3, overlapping (stride 1), vocabulary capped at the top
4,000 most frequent trigrams.

Overlapping, single-order n-grams match standard n-gram
language-identification practice (Cavnar & Trenkle, 1994; `langid.py`;
fastText subword features) and create an exact parallel to the NB baseline:
mean-pooling embeddings over overlapping n-grams is, up to normalization, a
weighted sum over n-gram counts — the same features as the NB baseline,
with learned weights instead of fixed counts.

Overlap vs. non-overlap is screened empirically, using the same two proxies
as the vocab-size check: mutual information (token identity vs. language
label, from counts) and NB validation accuracy on each candidate token set.
This proxy is well matched to the real architecture, since embedding
lookup + mean pooling is structurally close to NB/logistic regression over
the same counts. It screens out weaker configurations but doesn't guarantee
the learned-embedding architecture ranks them identically — reported as an
explicit scope limitation.

## Hyperparameters: shared vs. tuned arms

Every method runs under two arms, to measure how much per-method tuning
changes the comparison:

- **Shared**: one pre-declared config, `dim=128`, `lr=1e-3`, used
  unmodified for every method.
- **Tuned**: same grid, same size for every method, selected on
  validation.

`dim` scales every method's embedding table by the same factor, so it
doesn't affect the *ratio* of parameter counts between methods (vocab size
does) — it's a capacity/difficulty knob, not a fairness knob. 128 chosen to
avoid a floor effect that could mask tokenization differences.

## Open items

- Run the sample-size screen; lock in N.
- Build the real split at that N.
- Run the vocab-size and n-gram-style screens on that split; lock in both.
- Finalize the tuning grid for the tuned arm.
