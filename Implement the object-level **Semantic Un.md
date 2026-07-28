# Object semantic confidence and uncertainty

The scene-graph builder stores semantic confidence and semantic uncertainty on
each `Object`. The signal is based on the margin between the assigned label and
the strongest semantically distinct competitor in the configured vocabulary.

For an object embedding `e`, `compute_semantic_margin_uncertainty()` computes
cosine similarities to the text vocabulary, selects the assigned label's
similarity, and compares it with the best eligible competitor. The configured
logit scale maps that margin through a sigmoid:

```text
c_sem = sigmoid(logit_scale * (assigned_similarity - competitor_similarity))
u_sem = 1 - c_sem
```

Zero-norm embeddings produce `c_sem = 0` and `u_sem = 1`. If no distinct
competitor is eligible, the assigned label is treated as fully confident.

`Object.save()` and `Object.load()` persist the semantic fields used by the
implementation: `label_idx`, `label_cos_sim`, `runner_up_idx`,
`runner_up_cos_sim`, `semantic_margin`, `c_sem`, and `u_sem`. There is no
separate vocabulary-wide uncertainty value or compatibility distribution.

Object embeddings are normalized after feature aggregation and after object
merges. Merging clears the semantic fields that depend on the embedding;
`Graph.recompute_semantic_uncertainty()` repopulates them after the final merge
pass without changing the assigned label.

The relevant configuration values are
`semantic_uncertainty_logit_scale` and
`semantic_uncertainty_synonym_threshold` under `pipeline` in
`config/create_graph.yaml`.

Tests cover the margin helper, zero-norm behavior, metadata round trips,
post-merge recomputation, and assignment stability.
