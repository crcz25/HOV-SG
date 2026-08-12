# Object-level uncertainty signals

Each `Object` stores the five confidence signals from `uncertainty_signals.tex`
and their complementary uncertainties:

- `p_det`, `u_det`: SAM's pooled `predicted_iou` and `1 - p_det`.
- `p_sem`, `u_sem`: `sigmoid(alpha * m)` for the best non-synonym text-label
  margin and its complement.
- `p_coh`, `u_coh`: the corresponding leave-one-out visual-prototype margin.
- `p_mem`, `u_mem`: the size-normalized vocabulary-vs-negative likelihood
  posterior and its complement.
- `p_view`, `u_view`: the rescaled mean pairwise cosine of the accumulated
  unit view embeddings and its complement.

The label-error estimators are kept separately and combined only for the
object-level probability as `p_sem_bar = min(p_sem, p_coh)`; an undefined
coherence signal reduces this to `p_sem`. Undefined evidence is stored as
`None`, never as a fabricated endpoint probability.

`Object.save()` and `Object.load()` persist each signal and the raw detection
and cross-view accumulators needed to reproduce the derived values. Signal
calculation is refreshed by `Graph.recompute_cross_view_consistency()` and
`Graph.recompute_semantic_uncertainty()` before the graph is serialized.

The shared `alpha` and synonym threshold `tau` are configured by
`semantic_uncertainty_logit_scale` and
`semantic_uncertainty_synonym_threshold` in `config/create_graph.yaml`.
