Implement the Kill #1 "Semantic Uncertainty" signal from 3dsg-uncertainty-plan.md
inside the HOV-SG scene-graph builder. This is object-level instrumentation only —
do NOT implement Bernoulli propagation, room/floor aggregation, or Kill #2/#3. Scope
is strictly: compute and persist a cross-modal disagreement signal on each Object node.

## What to compute, per Object, at the point its label is assigned

For an object with aggregated CLIP embedding `e` (this is `object.embedding`,
already computed by `Graph.segment_objects()` from `self.mask_feats[mask_idx]`)
and an assigned label `object.name` drawn from a fixed vocabulary with text
embeddings `text_feats` (rows) / `classes` (names), aligned 1:1:

1. Renormalize `e` to unit norm before any similarity math — do NOT assume it's
   already unit norm. `feats_denoise_dbscan` (cluster mean) and `Object.__add__`
   (`np.mean([e1, e2])`) both produce non-unit-norm vectors today, and this is a
   real bug for cosine similarity, not a hypothetical.
2. `label_cos_sim = cosine_similarity(e, text_feats[idx_of(object.name)])` where
   `idx_of` is the row index used by the original argmax in `identify_object` —
   do NOT re-derive this via `classes.index(object.name)` (fragile if names
   aren't unique strings); thread the index through instead.
3. Full similarity vector `s = cosine_similarity(e, text_feats)` (all classes,
   not just the winner) — reuse the `np.dot` already computed inside
   `identify_object`, do not recompute it a second time.
4. Temperature-scaled softmax: `p = softmax(100.0 * s)`. Use `100.0` as the
   default scale (standard CLIP zero-shot logit scale — raw CLIP cosine sims
   live in a ~0.2–0.35 band and produce a near-uniform softmax without scaling,
   which would make the entropy signal useless). Expose this as
   `cfg.pipeline.semantic_uncertainty_temperature` in `config/create_graph.yaml`
   (default `100.0`) rather than hardcoding it, matching how every other
   pipeline knob in this repo is config-driven.
5. `semantic_uncertainty = entropy(p) / log(len(classes))` — normalize by max
   entropy so the score is in `[0, 1]` and comparable across different
   `obj_labels` vocabularies (COCO_STUFF has 183 classes, HM3DSEM_LABELS has
   1625 — unnormalized entropy is not comparable between them). `1.0` = maximally
   uncertain (uniform distribution over labels), `0.0` = fully confident.
6. Zero-norm guard: if `norm(e) < 1e-8` (this happens — `create_feature_map`
   fills empty-feature masks with an all-zero vector, see the
   `if feats.shape[0] == 0` branch), skip the cosine/softmax math and set
   `label_cos_sim = 0.0`, `semantic_uncertainty = 1.0` directly.

## Files to touch

### `hovsg/graph/object.py`
- Add `self.label_cos_sim = None` and `self.semantic_uncertainty = None` to
  `Object.__init__`.
- In `save()`, add both to the `metadata` dict (use `float(x) if x is not None
  else None` — don't let a bare `np.float32` hit `json.dump` unguarded like the
  existing `embedding` field already correctly handles).
- In `load()`, read them back with `metadata.get("label_cos_sim")` and
  `metadata.get("semantic_uncertainty")` (plain `.get`, not `metadata["..."]`) —
  there is already an untracked `hovsg/data/hm3dsem/outputs/` directory in this
  repo from a prior run; any graph saved before this change must still load
  without a `KeyError`.
- In `__add__` (the merge operator), after computing
  `self.embedding = np.mean([self.embedding, other.embedding], axis=0)`,
  renormalize it in place: `self.embedding = self.embedding /
  np.linalg.norm(self.embedding)`. Do NOT try to average `label_cos_sim` /
  `semantic_uncertainty` across the two merged objects — set both to `None`
  here. They get correctly recomputed in the post-merge pass (see below), and
  averaging two entropy scores is not equivalent to recomputing entropy from
  the merged embedding, so a stale averaged value would be actively wrong.

### `hovsg/utils/clip_utils.py` (or a new `hovsg/utils/uncertainty.py` — your call,
but if new, import it from `graph.py` the same way the other `hovsg.utils.*`
helpers are imported)
- Add one small pure function:
  `compute_label_uncertainty(embedding, text_feats, temperature=100.0)` →
  returns `(sim_vector, entropy_normalized)`. Handles the zero-norm guard
  internally. Keep it side-effect-free and unit-testable in isolation —
  no Graph/Object dependency.

### `hovsg/graph/graph.py`
- `identify_object()` (line ~579): currently returns only
  `classes[np.argmax(similarity)]`, discarding `similarity`. Change its return
  to `(classes[argmax], argmax, similarity)` (name, index, full sim vector) —
  do not silently change the return arity without updating the one call site.
- `segment_objects()` (line ~592 onward): after
  `object.embedding = self.mask_feats[mask_idx]` (line ~701), renormalize it
  the same way as in `Object.__add__` (defensive — `mask_feats` comes out of
  `feats_denoise_dbscan`'s cluster mean, same non-unit-norm issue), then call
  `compute_label_uncertainty` and set `object.label_cos_sim` /
  `object.semantic_uncertainty`. Store `text_feats` and `classes` as
  `self.label_text_feats` / `self.label_classes` on the `Graph` instance right
  after the `get_label_feats(...)` call at the top of `segment_objects` — the
  post-merge recompute step below needs them and must not recompute/reload the
  vocabulary a second time.
- Add a new method `Graph.recompute_semantic_uncertainty()`: iterates
  `self.objects`, and for any object whose embedding changed via a merge
  (i.e., anything touched by `Object.__add__`), recomputes
  `label_cos_sim`/`semantic_uncertainty` against `self.label_text_feats` using
  the *already-assigned* `object.name` — do not re-run `identify_object`'s
  argmax here; merging does not change the object's label, only its embedding,
  per the existing `merge_objects()` contract (only same-`name` objects are
  ever merged).
- In `build_graph()` (line ~808), call `self.recompute_semantic_uncertainty()`
  immediately after the `if self.cfg.pipeline.merge_objects_graph:` block
  (after line ~830, `room.merge_objects()`) and before `self.create_graph()`
  (line ~832). Ordering matters: object identities/embeddings must be final
  before this runs, and the networkx graph nodes are the same instances, so
  running it after graph construction would also work, but before is clearer
  and keeps "finalize object state" as one contiguous phase.
- If `cfg.pipeline.merge_objects_graph` is `False`, the values computed inline
  during `segment_objects()` are already final — `recompute_semantic_uncertainty()`
  should still run unconditionally (cheap, idempotent) rather than being
  wrapped in the same `if`, so behavior doesn't silently diverge based on that
  flag.

### `config/create_graph.yaml`
- Add `semantic_uncertainty_temperature: 100.0` under `pipeline:`.

## Explicit non-goals (do not implement these, even if they seem like natural next steps)

- No Bernoulli/room-level propagation (`P(room contains X) = 1 - ∏(1-pᵢ)`) —
  that's Step 2 of the plan, out of scope here.
- No changes to `Room` or `Floor` classes.
- No changes to `create_nav_graph()` / navigation graph — unrelated system.
- No new CLI flags beyond the one config key above.
- Don't touch `query_object`/`query_graph`/`query_hierarchy` — they read
  `obj.embedding` via raw `np.dot` assuming unit norm; the renormalization
  you're adding to `__add__`/`segment_objects` makes their assumption *more*
  correct than it is today, so leave their code as-is.

## Verification before calling this done

- Unit-check `compute_label_uncertainty` directly: a near-one-hot similarity
  vector → `semantic_uncertainty` near 0; a uniform similarity vector →
  `semantic_uncertainty` near 1; an all-zero embedding → `(0.0, 1.0)` without
  raising.
- Confirm `Object.save()`/`load()` round-trips both new fields (`None` and
  float cases), and that loading a pre-existing `.json` from
  `hovsg/data/hm3dsem/outputs/` (missing the keys entirely) does not raise.
- Run the actual pipeline (or `load_graph` on existing output) on one scene,
  then print a quick distribution of `label_cos_sim` and `semantic_uncertainty`
  across `self.objects` — sanity-check it's not degenerate (e.g. everything
  exactly 0 or 1, which would indicate the temperature or normalization is
  wrong) and spot-check a few objects with labels from the known near-synonym
  clusters in `HM3D_CountsOfObjectTypes.csv` (`desk`/`workstation`/`desk chair`,
  `sofa`/`couch`/`sofa chair`) to see whether they land at similarly
  low-confidence scores, as the plan's own kill-experiment methodology expects.
