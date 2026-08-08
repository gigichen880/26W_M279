# Frozen PCA-8 whitening

`similarity_forecast/redesign/embeddings.py` class `PCAOnlyEmbedder`:
- default `whiten: bool = True`
- `PCA(..., whiten=self.whiten)` in `fit`

Frozen D0 uses `build_embedder("pca_only", ..., pca_k=8)` with defaults → **whitened PCA is correct**.
Old pipeline used `whiten=False`; redesign changed this intentionally.
