# Implement semantic confidence and uncertainty

Implement the object-level semantic signal in the HOV-SG scene-graph builder.
The implementation must expose semantic confidence `c_sem` and semantic
uncertainty `u_sem`, where `u_sem = 1 - c_sem`.

For each object, use the normalized aggregated CLIP embedding and the same
vocabulary similarities used by `Graph.identify_object()`. Compare the
assigned label with the strongest eligible semantically distinct competitor.
Pass the similarity margin through the configured sigmoid scale. A zero-norm
embedding returns `c_sem = 0` and `u_sem = 1`; an empty competitor set returns
`c_sem = 1` and `u_sem = 0`.

Persist the assigned label index, assigned-label similarity, runner-up details,
semantic margin, `c_sem`, and `u_sem` in `Object.save()`/`Object.load()`.
Do not add a vocabulary-wide score or any additional uncertainty
representation.

`Object.__add__()` must normalize the merged embedding and clear the semantic
fields. `Graph.recompute_semantic_uncertainty()` must recompute those fields
after the optional room object-merge phase without changing the assigned label.

Use the existing pure helpers in `hovsg/utils/uncertainty.py`, and keep
`semantic_uncertainty_logit_scale` and
`semantic_uncertainty_synonym_threshold` configurable under `pipeline`.
Update tests for helper behavior, zero-norm handling, metadata round trips,
merge invalidation, and final recomputation.
