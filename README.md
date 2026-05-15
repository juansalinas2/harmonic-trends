# harmonic-trends

This project studies harmonic language in popular music: which chord patterns
are common, how they change over time, how they differ by genre, and whether
those patterns can be used to predict what comes next.

The core representation is the harmonic `n`-gram. This borrows from the
language-modeling idea of word `n`-grams: a song is treated as a sequence, and
short windows of that sequence become the units we can count, compare, embed,
and model. Exact chord `n`-grams capture the surface progression. Harmonic
`n`-gram classes collapse exact progressions into transposition-normalized
patterns, so the analysis is about harmonic behavior rather than just chord
spelling or key.

The main question is not whether music is simply becoming more or less
"complex." That word is too broad. The notebooks make it measurable by looking
at several related quantities:

- **Vocabulary size:** how many harmonic `n`-gram classes are active in a decade
  or genre.
- **Concentration:** whether usage is spread across many patterns or dominated
  by a small set of common patterns.
- **Specificity:** which harmonic patterns are unusually characteristic of a
  genre, decade, or artist.
- **Change over time:** which harmonic families rise, fall, or shift between
  decades.
- **Predictability:** how well the next harmonic state can be estimated from the
  current state and metadata.

The result is a reproducible pipeline for building a harmonic vocabulary of
popular music. The project stores exact and harmonic `n`-gram counts in DuckDB,
exports document-term tables for decade and genre analysis, builds vector
representations of harmonic classes from co-occurrence behavior, and trains an
interpretable conditional model over transitions between harmonic states.

Mathematically, the pipeline keeps two views connected. `V_n` is the vocabulary
of exact chord `n`-grams, and `H_n` is the harmonic vocabulary after collapsing
exact progressions into normalized harmonic classes. The map `V_n -> H_n` lets
us study the fibers over a harmonic pattern: all exact progressions that realize
the same harmonic class. Across sequence length, truncation maps such as
`H_n -> H_{n-1}` let us ask how shorter harmonic contexts extend into longer
phrases. Those fibers are the basis for prediction and for comparing harmonic
organization across genres and eras.

## Notebook Pipeline

Run the notebooks in order. Notebooks `00` and `01` prepare the source data; the
remaining notebooks build and analyze the harmonic vocabulary.

0. `notebooks/00_download_chordonomicon_dataset.ipynb`
   Downloads the Chordonomicon dataset into `data/raw/`.

1. `notebooks/01_build_canonical_dataset.ipynb`
   Normalizes the source data into a canonical song table for downstream analysis.

2. `notebooks/02_build_ngram_dataset.ipynb`
   Builds global exact and harmonic n-gram vocabularies, then writes `V_n`, `H_n`, `O_n`, and `bar O_n` directly into DuckDB.

3. `notebooks/03_build_frequency_objects.ipynb`
   Validates and inspects the global DuckDB store.

4. `notebooks/04_stratified_ngram_trends.ipynb`
   Tracks a small fixed set of globally common targets across year, decade, and genre.

5. `notebooks/05_analyze_stratified_trends.ipynb`
   Analyzes the target-limited trend tables from notebook 4.

6. `notebooks/06_interpret_harmonic_trends.ipynb`
   Turns target-limited trends into interpretable candidate findings.

7. `notebooks/07_build_harmonic_document_terms.ipynb`
   Builds the broader support-thresholded document-term table needed for corpus-linguistic statistics.

8. `notebooks/08_corpus_linguistic_occurrence_analysis.ipynb`
   Computes document frequency, stop-gram candidates, TF-IDF, entropy, and one-vs-rest enrichment.

9. `notebooks/09_ultimate_harmonic_eda.ipynb`
   Pulls the distilled outputs together into a report-style EDA: rising/falling harmonic families, diversity and concentration through time, genre vocabulary breadth, signature terms, regime shifts, and bounded artist-level vocabulary profiling.

10. `notebooks/10_harmonic_distributional_embeddings.ipynb`
   Builds distributional embeddings of harmonic `n`-gram classes from co-occurrence graphs: song/local context counts, PPMI matrices, SVD embeddings, nearest-neighbor inspection, structural-vs-distributional comparison, and genre centroids.

11. `notebooks/11_conditional_harmonic_language_modeling.ipynb`
    Builds an interpretable conditional language model over harmonic states: adjacent `H_n(t) -> H_n(t+1)` transitions, global/decade/genre/artist scoped counts, support-aware backoff interpolation, holdout evaluation, and style-conditioned continuation examples.

## Data

The `data/` directory is intentionally ignored by Git. The raw dataset,
intermediate DuckDB database, and generated CSV/NPZ outputs are produced by the
notebooks and can be rebuilt locally.

## Notebook Troubleshooting

- If a notebook imports the wrong `utils` module, restart the kernel and rerun from the top. Notebooks 2-8 now assert that they loaded utilities from this repository.
- If `duckdb` is missing, install the notebook dependencies with `python3 -m pip install -r requirements.txt` in the same interpreter used by the notebook kernel.
- If DuckDB reports a lock on `data/processed/harmonic_trends.duckdb`, stop/restart old Jupyter kernels that still have the database open, then rerun the notebook.
