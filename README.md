# harmonic-trends

We study harmonic trends in popular music.

## Notebook Pipeline

The current analysis has two branches after the global vocabulary is built:

1. `notebooks/02_build_ngram_dataset.ipynb`
   Builds global exact and harmonic n-gram vocabularies, then writes `V_n`, `H_n`, `O_n`, and `bar O_n` directly into DuckDB.

2. `notebooks/03_build_frequency_objects.ipynb`
   Validates and inspects the global DuckDB store.

3. `notebooks/04_stratified_ngram_trends.ipynb`
   Tracks a small fixed set of globally common targets across year, decade, and genre.

4. `notebooks/05_analyze_stratified_trends.ipynb`
   Analyzes the target-limited trend tables from notebook 4.

5. `notebooks/06_interpret_harmonic_trends.ipynb`
   Turns target-limited trends into interpretable candidate findings.

6. `notebooks/07_build_harmonic_document_terms.ipynb`
   Builds the broader support-thresholded document-term table needed for corpus-linguistic statistics.

7. `notebooks/08_corpus_linguistic_occurrence_analysis.ipynb`
   Computes document frequency, stop-gram candidates, TF-IDF, entropy, and one-vs-rest enrichment.

8. `notebooks/09_ultimate_harmonic_eda.ipynb`
   Pulls the distilled outputs together into a report-style EDA: rising/falling harmonic families, diversity and concentration through time, genre vocabulary breadth, signature terms, regime shifts, and bounded artist-level vocabulary profiling.

9. `notebooks/10_harmonic_distributional_embeddings.ipynb`
   Builds distributional embeddings of harmonic `n`-gram classes from co-occurrence graphs: song/local context counts, PPMI matrices, SVD embeddings, nearest-neighbor inspection, structural-vs-distributional comparison, and genre centroids.

10. `notebooks/11_conditional_harmonic_language_modeling.ipynb`
    Builds an interpretable conditional language model over harmonic states: adjacent `H_n(t) -> H_n(t+1)` transitions, global/decade/genre/artist scoped counts, support-aware backoff interpolation, holdout evaluation, and style-conditioned continuation examples.

## Notebook Troubleshooting

- If a notebook imports the wrong `utils` module, restart the kernel and rerun from the top. Notebooks 2-8 now assert that they loaded utilities from this repository.
- If `duckdb` is missing, install the notebook dependencies with `python3 -m pip install -r requirements.txt` in the same interpreter used by the notebook kernel.
- If DuckDB reports a lock on `data/processed/harmonic_trends.duckdb`, stop/restart old Jupyter kernels that still have the database open, then rerun the notebook.
