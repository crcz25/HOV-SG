# Room-level uncertainty propagation

Room containment beliefs use the two object-level signals already computed by
the scene-graph builder:

- semantic confidence `c_sem` and semantic uncertainty `u_sem`;
- detection confidence `c_det` and detection uncertainty `u_det`.

`Graph.propagate_semantic_uncertainty_to_rooms()` keeps the assigned
`label_idx` fixed and stores `c_sem`, `c_det`, and their product in the room's
semantic, detection, and combined object-belief maps. `Room` fuses each map
with a noisy-OR operation by class index.

Every object must have `label_idx`, `c_sem`, and `c_det` before propagation.
The method does not infer missing labels, inspect legacy confidence aliases, or
reconstruct a vocabulary-wide score. The object fields remain the authoritative
uncertainty values and are persisted by `Object.save()`.

The implementation is intentionally independent of graph construction and
navigation-graph code. Tests cover empty rooms, independent signal changes,
combined confidence, persistence, and monotonic noisy-OR fusion.
