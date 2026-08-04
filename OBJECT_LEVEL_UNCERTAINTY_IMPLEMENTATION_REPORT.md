# Object-Level Uncertainty in HOV-SG — Implementation Report

Audit, correction and validation of the five object-level uncertainty signals:
Detection Confidence, Semantic Uncertainty, Vocabulary Membership, Cross-view
Consistency and Label Coherence.

Convention used throughout the codebase after this work: **confidence `p_<signal>`
(= `P^x_i`)**, **uncertainty `u_<signal>` (= `U^x_i = 1 - P^x_i`)**. No compatibility
aliases were retained.

---

## 1. Executive summary

### How HOV-SG constructs object-level nodes

HOV-SG never runs a closed-vocabulary object detector. Objects emerge from a
three-stage process in `hovsg/graph/graph.py`:

1. **Per-frame** — SAM proposes class-agnostic masks; each mask is encoded with CLIP
   (ConceptFusion-style blend of a masked crop, an unmasked crop and the global image
   embedding) and back-projected into 3D.
2. **Fusion** — every mask's per-pixel feature is scattered onto the nearest point of
   the global point cloud and averaged, giving a per-point CLIP feature. Per-frame 3D
   masks are then merged across frames (`seq_merge` / `hierarchical_merge`) into
   persistent 3D masks. Each mask's points are DBSCAN-filtered in feature space and
   averaged into one fused embedding `v_i`.
3. **Graph** — each mask becomes an `Object`, assigned to the room with the highest
   2D overlap, labelled by nearest-text-embedding over the configured vocabulary, and
   serialized as `<object_id>.ply` + `<object_id>.json`.

### Where the signals enter

| Stage | Signals produced |
|---|---|
| `create_feature_map` (per frame) | Detection confidence accumulator, cross-view unit-embedding accumulator |
| `create_feature_map` (per mask) | Both accumulators rolled up per mask |
| `segment_objects` (per object) | `p_det`, `p_view`, `p_sem`, `p_mem` |
| `recompute_semantic_uncertainty` (after merges) | `p_sem`, `p_mem` refreshed; then `p_coh` |
| `recompute_label_coherence` (whole-graph) | `p_coh` — needs every object's embedding |

### Status of each signal on arrival, and what changed

| Signal | On arrival | Most important correction |
|---|---|---|
| Detection Confidence | Partially correct | Merges averaged already-averaged means, so a chain of N merges weighted objects 1/2, 1/4, 1/8 … and depended on merge order. Now stores `(sum, count)` and pools exactly. |
| Semantic Uncertainty | Correct formula, wrong edge cases | Degenerate inputs returned `margin = 0.0` **with** `P_sem = 0.0` — mutually inconsistent, since σ(α·0) = 0.5. Now undefined (`None`). |
| Vocabulary Membership | Formula correct, bank poor | 73.1% of the mined negative bank were proper nouns, dates and titles of works (`'s Gravenhage`, `15 August 1945`, `1st Baron Beaverbrook`). Mining now keeps common nouns only. |
| Cross-view Consistency | **Incorrect** | The accumulator counted every pixel that projected onto a point, while the fusion it is supposed to mirror keeps only one write per point per frame. `m_i` was the mean of a *different, larger* observation set than `v_i`. |
| Label Coherence | Correct, untested | No tests existed at all. Leave-one-out, τ-exclusion, prototype freshness and undefined handling are now covered. |

### Most important corrections

1. **Cross-view accumulation double-counted observations** (`accumulate_unit_embeddings`).
   HOV-SG's fusion uses buffered fancy indexing (`sum_features[idx] += F_2D`), which
   applies only the **last** write when `idx` repeats; the cross-view accumulator used
   `np.add.at`, which applies **all** of them. Since several pixels of one frame
   routinely project onto the same voxel-downsampled point, `cross_view_count` exceeded
   the fusion's own counter and `m_i` was the mean of observations `v_i` never saw.
   This directly contradicts the definition's premise — *"we recover the discarded
   disagreement **from the fusion**"*. Fixed by keeping one (last) valid observation per
   point per view.

2. **`cross_view_sufficient` measured the wrong quantity.** The count is a total over
   point-view pairs, so `count >= 2` was true for virtually any object seen once, making
   the low-support flag meaningless. It now thresholds **views per point**.

3. **Fabricated probabilities for undefined signals.** Semantic Uncertainty returned
   `P_sem = 0.0` for a zero embedding and `P_sem = 1.0` when every competitor was a
   synonym; Vocabulary Membership returned `P_mem = 0.0` for a zero embedding. All three
   are now `None`, matching how Label Coherence already reported the same situations.
   `identify_object` no longer assigns class 0 to an all-zero similarity vector.

4. **Detection merge semantics** — pooled mean via stored `(sum, count)`, order-independent.

5. **Negative-label mining quality** — common-noun filter, normalized storage, count
   limit that keeps the labels *furthest* from the vocabulary rather than the
   alphabetically first, and cache metadata versioning so stale banks are rebuilt.

### Remaining limitations

- **`m_i` is a point-weighted mean, not a per-object per-view mean.** `seq_merge`
  discards the frame→object correspondence, so there is no per-view object embedding to
  average. The accumulator is keyed by point and the object-level mean weights each view
  by how many of the object's points it saw. Where all pixels of a mask share one
  feature (the normal case in HOV-SG) this is a *weighted* form of the definition, not a
  different quantity — but it is not literally eq. (crossview). Recovering the exact form
  requires threading provenance through `seq_merge`; see §7.
- **`|N| / |C| ≈ 3:1`** after the mining fix (down from 12:1). The equation compares
  partition *sums*, so this ratio shifts `P_mem` for every object. It is a documented
  configuration knob, not a calibrated value.
- **SAM `predicted_iou` is a mask-quality score, not a calibrated object-existence
  probability.** Treating it as `P_det` is the definition's premise, but the calibration
  claim is untested without ground truth.
- No calibration evaluation was run (requires HM3D ground-truth annotations; see §7).

---

## 2. HOV-SG object-generation pipeline

### Data flow

```
dataset frame (RGB, depth, pose)
  └─ HM3DSemDataset.__getitem__            hovsg/dataloader/hm3dsem.py
     └─ extract_feats_per_pixel            hovsg/models/sam_clip_feats_extractor.py:70
        ├─ mask_generator.generate(image)  SAM → masks[] with predicted_iou
        ├─ CLIP on masked + unmasked crops → F_l, on full image → F_g
        ├─ F_p = w·F_g + (1-w)·F_l, L2-normalized       ← per-mask, unit norm
        └─ outfeat: per-pixel feature (sum of overlapping F_p, renormalized;
                    zero for pixels outside every mask)
     └─ dataset.create_pcd / create_3d_masks → per-frame 3D masks
     └─ scatter onto the global cloud via cKDTree.query
        ├─ sum_features[idx] += F_2D ; counter[idx] += 1      (fusion)
        ├─ accumulate_confidence(...)   → detection evidence
        └─ accumulate_unit_embeddings(...) → cross-view evidence
  └─ seq_merge / hierarchical_merge        hovsg/utils/graph_utils.py:595
     → self.mask_pcds (persistent 3D masks)
  └─ per-mask roll-up ("Fusing features" loop)
     ├─ feats_denoise_dbscan_with_indices → v_i + the surviving point indices
     ├─ mask_confs[i] = (conf sum, point count)
     └─ mask_cross_view_{sums,counts,point_counts}[i]
  └─ Graph.segment_objects                 → Object nodes, p_det/p_view/p_sem/p_mem
  └─ Room.merge_objects (optional)         → Object.__add__
  └─ Graph.recompute_cross_view_consistency
  └─ Graph.recompute_semantic_uncertainty  → refreshes p_sem/p_mem, then p_coh
  └─ Graph.save_graph                      → objects/<id>.ply + <id>.json
```

### Key symbols

| Symbol | Location | Role |
|---|---|---|
| `Graph.create_feature_map` | `hovsg/graph/graph.py:195` | Frame loop; builds all three point-level accumulators |
| `Graph.identify_object` | `hovsg/graph/graph.py:696` | Label assignment; returns `(None, None, sim)` for a degenerate embedding |
| `Graph.segment_objects` | `hovsg/graph/graph.py:721` | Creates `Object` nodes and writes four of the five signals |
| `Graph._assign_object_semantic_signals` | `hovsg/graph/graph.py:947` | Shared semantic + membership writer (single cosine vector reused) |
| `Graph._normalized_negative_text_feats` | `hovsg/graph/graph.py:936` | Lazily normalizes and caches the negative bank |
| `Graph.recompute_cross_view_consistency` | `hovsg/graph/graph.py:992` | Re-derives `p_view` from persisted raw evidence |
| `Graph.recompute_semantic_uncertainty` | `hovsg/graph/graph.py:1015` | Post-merge refresh; delegates to coherence |
| `Graph.recompute_label_coherence` | `hovsg/graph/graph.py:1047` | Rebuilds visual prototypes and scores every object |
| `Graph.propagate_semantic_uncertainty_to_rooms` | `hovsg/graph/graph.py:1166` | Room-level duplicate-aware noisy-OR over fused `p_obj`; skips undefined probabilities |
| `Object.__add__` | `hovsg/graph/object.py:239` | Merge: adds raw accumulators, clears embedding-derived signals |
| `Object.save` / `Object.load` | `hovsg/graph/object.py:79` / `:182` | JSON round-trip of raw evidence **and** derived values |

### Intermediate representations

| Name | Shape | Meaning |
|---|---|---|
| `full_feats_array` | `(n_points, D)` | Mean CLIP feature per point |
| `full_conf_array` | `(n_points, 1)` | Mean SAM `predicted_iou` per point |
| `full_cross_view_sum` | `(n_points, D)` | Σ unit per-view embeddings per point |
| `full_cross_view_count` | `(n_points,)` | Number of **views** per point (post-fix) |
| `mask_feats[i]` | `(D,)` | Fused object embedding `v_i` |
| `mask_confs[i]` | `(float, int)` | Detection `(sum, count)` |
| `mask_cross_view_sums/counts/point_counts[i]` | `(D,)`, `int`, `int` | Object-level cross-view evidence |
| `class_embedding_sum` / `class_count` / `class_prototype_full` | dicts keyed by `label_idx` | Visual prototypes for Label Coherence |

---

## 3. Per-signal implementation

### 3.1 Detection Confidence

**Definition.** The detector confidence `s_i` is itself the signal: `P^det_i = s_i`,
`U^det_i = 1 - P^det_i`. **Error event:** false positive — a detection corresponding to
no real object.

| Aspect | Detail |
|---|---|
| Source input | SAM `mask["predicted_iou"]` |
| Implementation | `hovsg/utils/detection_uncertainty.py` — `mask_predicted_iou`, `accumulate_confidence`, `finalize_confidence_array`, `object_confidence_sum_from_points`, `confidence_from_sum`, `uncertainty_from_confidence` |
| Computed at | Per frame (accumulate) → per mask (roll-up) → `segment_objects` (`graph.py:873`) |
| Update across observations | Point-level running mean over every mask observation covering the point; object-level mean over the object's points |
| Merge | `Object.__add__` adds `detection_conf_sum` and `detection_point_count` → exact pooled mean, order-independent |
| Stored | `p_det`, `u_det`, `detection_conf_sum`, `detection_point_count` |
| Undefined | `p_det = u_det = None` when `detection_point_count == 0` or absent |
| Config | none (SAM thresholds under `models.sam`) |
| Numerical stability | NaN `predicted_iou` raises; out-of-range values clamp with a `RuntimeWarning`; `finalize_confidence_array` guards division by zero with ε = 1e-5 |
| Cost | O(points per mask) per frame; 2 floats per object |
| Tests | `tests/test_detection_uncertainty.py` (12 tests) |

**Changed.** Previously `Object.__add__` did `mean([self.p_det, other.p_det])`. Merging
three objects gave weights (1/4, 1/4, 1/2) depending on order. Now:
`P_det = (Σ_a + Σ_b) / (n_a + n_b)`. `object_confidence_from_points` was removed as an
unused duplicate of `object_confidence_sum_from_points`.

**Aggregation justification.** The stored value is the mean SAM `predicted_iou` over all
mask observations covering the object's points. Averaging is the defensible choice here —
`max` would report the single most optimistic mask and `latest` would discard all prior
evidence — but see §7 on the calibration gap.

---

### 3.2 Semantic Uncertainty

**Definition.**

```
m_i   = cos(v_i, t_{l_i}) − max_{c : cos(t_c, t_{l_i}) < τ} cos(v_i, t_c)
P^sem = σ(α · m_i),   U^sem = 1 − P^sem
```

**Confidence:** the model's belief that `l_i` is the correct class rather than the nearest
*distinct* one. **Error event:** labelling error.

| Aspect | Detail |
|---|---|
| Source inputs | `v_i` (fused object embedding), vocabulary text bank `t_c`, τ, α |
| Implementation | `hovsg/utils/uncertainty.py:116` `compute_semantic_margin_uncertainty`; τ-mask from `build_synonym_eligibility_mask` (`:94`) |
| Computed at | `segment_objects` → `_assign_object_semantic_signals` (`graph.py:947`); refreshed by `recompute_semantic_uncertainty` |
| Update | Recomputed from scratch whenever `v_i` or `label_idx` changes; `__add__` clears it |
| Stored | `label_cos_sim`, `runner_up_idx`, `runner_up_cos_sim`, `semantic_margin`, `p_sem`, `u_sem` |
| Undefined | All `None` for a zero-norm/non-finite embedding, or when every competitor is a synonym (`label_cos_sim` still reported in the latter case) |
| Config | `semantic_uncertainty_logit_scale` (100.0), `semantic_uncertainty_synonym_threshold` (0.75) |
| Numerical stability | `scipy.special.expit` (no manual `exp`); rows normalized with a 1e-8 guard |
| Cost | One `(|C| × D)` matvec per object, reused by Vocabulary Membership; τ-mask is `O(|C|²)` built **once** |
| Tests | `tests/test_semantic_uncertainty.py` (26 tests) |

**Verification against the checklist.**

- Image and text embeddings normalized before cosine — yes (`normalize_rows`, plus the
  embedding is normalized inside every entry point).
- Assigned label is the highest-scoring class — yes (`identify_object`, `argmax`).
- Competing class excludes near-synonyms by `cos(t_c, t_l) < τ` — yes, and the mask's
  diagonal is cleared so `c = l_i` can never compete.
- Competing class is the highest-scoring *valid* one — yes (`argmax` restricted to the
  eligible set).
- σ applied to `α·m_i`, α = CLIP's learned scale — yes; verified over α ∈ {1, 10, 100}.
- Complement — yes, `u_sem = 1 − p_sem` exactly.
- No entropy anywhere — enforced by a regression test that greps the six source files.

**Changed.** Zero/non-finite embedding previously returned
`{margin: 0.0, p_sem: 0.0, u_sem: 1.0}` — internally contradictory, since σ(α·0) = 0.5.
Empty eligible set previously returned `p_sem = 1.0`. Both now report `None`.
`assume_normalized` was added so the 1624×1024 vocabulary is normalized once per run
rather than once per object.

---

### 3.3 Vocabulary Membership

**Definition.**

```
P^mem_i = Σ_{c∈C} e^{α cos(v_i,t_c)} / ( Σ_{c∈C} e^{α cos(v_i,t_c)} + Σ_{t∈N} e^{α cos(v_i,t)} )
```

**Confidence:** share of similarity mass falling on the vocabulary. **Error event:**
out-of-vocabulary content.

| Aspect | Detail |
|---|---|
| Source inputs | `v_i`, vocabulary bank `C`, negative bank `N`, α |
| Implementation | `hovsg/utils/uncertainty.py:310` `compute_vocabulary_membership`; bank from `hovsg/utils/negative_labels.py:107` `load_or_build_negative_label_feats` |
| Computed at | Same call site as Semantic Uncertainty, sharing the vocabulary cosine vector |
| Update | Recomputed whenever `v_i` changes; `__add__` clears it |
| Stored | `vocab_log_partition`, `negative_log_partition`, `p_mem`, `u_mem` |
| Undefined | All `None` for a zero-norm/non-finite embedding |
| Config | `vocab_membership_max_class_similarity` (0.5; `null` disables the signal), `vocab_membership_negative_label_count` (`null` = whole bank) |
| Numerical stability | **Required**, not optional: at α = 100, `exp(α·cos)` overflows float64 for cos ≳ 7.1e-3. Implemented as `σ(log Z_C − log Z_N)` with `scipy.special.logsumexp`, algebraically identical to the ratio |
| Cost | One `(|N| × D)` matvec per object (~5k × 1024). Bank is mined once, cached to disk, normalized once per run |
| Tests | `tests/test_semantic_uncertainty.py` (membership group), `tests/test_negative_labels.py` (7 tests) |

**Verification against the checklist.**

- Numerator is the **total** mass over `C`, not the max — verified by a test showing that
  duplicating a class raises `P_mem`.
- Denominator covers `C ∪ N`, same α for both — yes.
- Negative embeddings precomputed **and normalized** — now stored normalized.
- Mining reproducible and documented — sorted, deduplicated common-noun lemmas of
  Open English Wordnet (`oewn:2025+`), filtered by `is_common_noun_lemma`, kept when
  `max cos to any class < max_class_similarity`. Recorded in
  `negative_label_metadata.json` with a `mining_version`.
- Scene independence — the API takes only a CLIP model and the vocabulary; a test asserts
  the signature exposes no scene/object parameter.
- Membership ≠ semantic confidence — a test constructs an object with high `P_mem` and
  `P_sem = 0.5` simultaneously.
- Empty vocabulary / empty negative set / invalid embeddings — explicit `ValueError` for
  the first two, `None` for the third; zero-norm candidate rows are dropped at mining time.

**Changed.**

| | Before | After |
|---|---|---|
| Bank composition | 19,852 labels, **73.1% proper nouns/dates/titles** | Common nouns only |
| Stored norms | Unnormalized (0.85–0.98), renormalized per object | Unit rows, normalized once |
| Count limit | First N alphabetically (biased to "a…") | N furthest from the vocabulary |
| Missing metadata | Cache trusted | Cache rebuilt |

> The on-disk bank under `hovsg/labels/` predates this change. Its metadata no longer
> matches (`mining_version`), so it is rebuilt automatically on the next run — no manual
> cleanup required.

---

### 3.4 Cross-view Consistency

**Definition.** `P^view_i = ‖m_i‖`, with `m_i` the running mean of the per-view **unit**
embeddings of object `o_i`. One when every view agreed, → 0 as views cancel.
**Error event:** an inconsistent entry.

| Aspect | Detail |
|---|---|
| Source inputs | Per-pixel CLIP features `F_2D` (unit norm; zero outside every SAM mask) |
| Implementation | `hovsg/utils/cross_view_consistency.py` — `accumulate_unit_embeddings` (`:35`), `cross_view_values` (`:92`) |
| Computed at | Accumulated per frame in `create_feature_map`; rolled up per mask; written in `segment_objects` (`graph.py:889`) |
| Update | Sums and counts are additive — merges and later views update the mean incrementally |
| Stored | `cross_view_resultant_sum` (**unnormalized**), `cross_view_count`, `cross_view_point_count`, `p_view`, `u_view`, `cross_view_sufficient` |
| Undefined | `(None, None, False)` when `count == 0` or the sum is non-finite |
| Config | `cross_view_consistency_min_observations` (2) |
| Numerical stability | Zero-norm and non-finite embeddings skipped before normalization; `p_view` clipped to [0,1] to absorb drift at the perfect-agreement boundary |
| Cost | One `(n_points × D)` float64 accumulator + one int64 counter over the scene; D+2 values per object |
| Tests | `tests/test_cross_view_consistency.py` (10 tests), `tests/test_object_uncertainty_end_to_end.py` |

**Verification against the checklist.**

- Each per-view embedding unit-normalized before accumulation — yes.
- The running mean is **not** normalized before its norm is measured — verified by a test
  that shows normalizing first would always yield exactly 1.0.
- The fused embedding `v_i` never replaces the accumulator — they are separate fields,
  both serialized.
- Enough state retained to recompute — `recompute_cross_view_consistency` re-derives
  `p_view` from persisted raw evidence alone, including after a load.
- Rejected/invalid/duplicate observations excluded — zero features skipped; **duplicates
  now deduplicated per view (this was the bug)**.
- A single valid observation gives 1.0 — tested, including with 5 pixels in one view.
- Values stay in [0,1] — property test over 50 random view sets.

**Changed (the substantive correction).**

`accumulate_unit_embeddings` used `np.add.at`, adding **every** pixel that projected onto
a point. The fusion it mirrors uses buffered fancy indexing, which applies only the last
write per repeated index:

```python
a = np.zeros((3,2)); idx = [0,0,1]; b = [[1,1],[2,2],[5,5]]
a[idx] += b        # → [[2,2],[5,5],[0,0]]   (first row discarded)
np.add.at(a,idx,b) # → [[3,3],[5,5],[0,0]]   (both counted)
```

So `m_i` averaged observations that never entered `v_i`, and `cross_view_count` exceeded
the fusion's own counter. Fixed by keeping the last valid observation per point per view.

`cross_view_sufficient` previously thresholded the total point-view count, which is
`n_points × n_views` — always ≥ 2 for any real object, making the flag vacuous. It now
thresholds `count / point_count`, the mean number of views per point.

**Known approximation.** `seq_merge` discards the frame→object correspondence, so there
is no per-view *object* embedding to average; the accumulator is keyed by point and the
object-level mean weights each view by how many of the object's points it saw. Because
all pixels of a SAM mask share one `F_p`, this reduces to a point-count-weighted form of
eq. (crossview) rather than a different quantity — but it is not literally the unweighted
mean. See §5 and §7.

---

### 3.5 Label Coherence

**Definition.**

```
m'_i   = cos(v_i, μ_{l_i}) − max_{c : cos(t_c, t_{l_i}) < τ} cos(v_i, μ_c)
P^coh  = σ(α · m'_i),   U^coh = 1 − P^coh
```

with `μ_c` the normalized mean of the visual embeddings of objects labelled `c`, and `v_i`
**excluded** from the mean of its own class. Undefined when `l_i` has no other instance.
**Error event:** labelling error — *the same event as Semantic Uncertainty*.

| Aspect | Detail |
|---|---|
| Source inputs | All object embeddings and labels in the map; text bank only for the τ-gate |
| Implementation | `hovsg/utils/uncertainty.py:204` `compute_label_coherence_uncertainty`; prototypes built in `Graph.recompute_label_coherence` (`graph.py:1047`) |
| Computed at | Whole-graph pass **after** all merges — deliberately not in `segment_objects` |
| Update | Prototypes rebuilt from scratch on every call, so relabels, merges and deletions are all picked up |
| Stored | `coherence_prototype_cos_sim`, `coherence_runner_up_class`, `coherence_runner_up_cos_sim`, `label_coherence_margin`, `p_coh`, `u_coh` |
| Undefined | All `None` when: the class has < 2 members; the leave-one-out sum has zero norm; no eligible distinct prototype exists; the embedding or label is invalid |
| Config | `label_coherence_logit_scale` (100.0), `label_coherence_synonym_threshold` (0.9) — intentionally independent of the semantic values |
| Numerical stability | Leave-one-out norm and every competitor similarity checked with `np.isfinite` before use |
| Cost | O(#objects × D) to build prototypes + O(#classes × D) per object; prototypes held only for instantiated classes |
| Tests | `tests/test_label_coherence.py` (11 tests, new) |

**Verification against the checklist.**

- Prototypes from **visual** embeddings, not text — verified by a test whose text
  features are orthogonal to the visual ones, leaving the result unchanged.
- Object excluded from its own prototype — verified against the self-inclusive
  alternative, which scores strictly higher.
- Assigned-class prototype needs ≥ 1 other valid object — enforced by `count < 2`.
- Competing prototypes exclude synonyms via τ — verified with a synonym class that would
  otherwise win on visual similarity.
- Only instantiated classes with valid prototypes participate — `class_prototype_full`
  contains only classes whose mean has non-zero norm.
- Same α and σ conversion — yes, via the shared `_sigmoid_margin_confidence`.
- Undefined represented explicitly — `None`, never 0, 1 or a substitute.
- Caches invalidated on relabel/merge/delete — full rebuild per call; tested for all three.

**Changed.** The computation was already correct. Added: shared `COHERENCE_FIELDS`
constant (replacing three duplicated inline tuples), prototypes built from the
pre-normalized vocabulary, and the first tests for this signal.

**Dependence note.** Label Coherence and Semantic Uncertainty estimate the *same* event
via different references (visual prototypes vs. text embeddings). They are alternative
estimators, not independent factors; multiplying them would double-count. The code keeps
them in separate fields and never combines them — asserted by a test.

---

## 4. Integration matrix

| Signal | Input data | Computation location | Update frequency | Stored graph field(s) | P range | U range | Undefined when | Config | Tests | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Detection | SAM `predicted_iou` | `detection_uncertainty.py`; `graph.py:873` | Per frame → per mask → per object; pooled on merge | `p_det`, `u_det`, `detection_conf_sum`, `detection_point_count` | [0,1] | [0,1] | no point evidence | — | `test_detection_uncertainty.py` | Compliant after correction |
| Semantic | `v_i`, text bank, τ, α | `uncertainty.py:116`; `graph.py:947` | On object creation and after every merge | `p_sem`, `u_sem`, `semantic_margin`, `label_cos_sim`, `runner_up_idx`, `runner_up_cos_sim` | (0,1) | (0,1) | degenerate `v_i`; all competitors synonyms | `semantic_uncertainty_logit_scale`, `semantic_uncertainty_synonym_threshold` | `test_semantic_uncertainty.py` | Compliant after correction |
| Membership | `v_i`, text bank, negative bank, α | `uncertainty.py:310`; `negative_labels.py:107` | Same as Semantic (shares the cosine vector) | `p_mem`, `u_mem`, `vocab_log_partition`, `negative_log_partition` | (0,1) | (0,1) | degenerate `v_i`; signal disabled | `vocab_membership_max_class_similarity`, `vocab_membership_negative_label_count` | `test_semantic_uncertainty.py`, `test_negative_labels.py` | Compliant after correction |
| Cross-view | Per-pixel CLIP features | `cross_view_consistency.py`; `graph.py:889` | Per frame (incremental); additive on merge | `p_view`, `u_view`, `cross_view_resultant_sum`, `cross_view_count`, `cross_view_point_count`, `cross_view_sufficient` | [0,1] | [0,1] | zero observations; non-finite sum | `cross_view_consistency_min_observations` | `test_cross_view_consistency.py` | Partially compliant (see §5) |
| Coherence | All object embeddings + labels; text bank for τ | `uncertainty.py:204`; `graph.py:1047` | Whole-graph pass after all merges | `p_coh`, `u_coh`, `label_coherence_margin`, `coherence_prototype_cos_sim`, `coherence_runner_up_class`, `coherence_runner_up_cos_sim` | (0,1) | (0,1) | singleton class; no eligible prototype; invalid input | `label_coherence_logit_scale`, `label_coherence_synonym_threshold` | `test_label_coherence.py` | Fully compliant |

---

## 5. Mathematical compliance audit

### Detection Confidence — **Compliant after correction**

`P^det_i = s_i` with `s_i` the mean SAM `predicted_iou` over the object's points. The
definition does not prescribe an aggregation across multiple observations; the pooled
mean is now exact and order-independent. The previous mean-of-means was neither.

### Eq. (semantic): `m_i`, `P^sem_i = σ(α m_i)` — **Compliant after correction**

Every term matches: normalization, argmax label, τ-gated competitor set, `argmax` over
eligible classes only, `σ(α·m)` with α = 100, and `U = 1 − P`. Verified numerically
against an independent re-derivation of the equation. The corrections were confined to
the undefined branches, which previously reported inconsistent numbers.

### Eq. (membership): `P^mem_i` — **Compliant after correction**

`σ(log Z_C − log Z_N)` is algebraically identical to `Z_C / (Z_C + Z_N)` and is required
for stability at α = 100. Verified against a direct evaluation of the equation at α = 5,
where the naive form does not overflow. Both partitions use the same α; the numerator is
the full mass over `C`.

The equation is agnostic to how `N` is chosen, so a poor bank is not an equation
violation — but the definition requires labels "whose text embeddings are distant from
every class in `C`", and 73.1% of the previous bank were proper nouns and dates. That
selection is now corrected and documented.

### Eq. (crossview): `P^view_i = ‖m_i‖` — **Partially compliant**

The norm-of-the-unnormalized-running-mean is implemented exactly, and the accumulator now
holds one unit observation per view per point, matching the observation set the fusion
averaged. What remains inexact is the *granularity*: `m_i` is the mean over the object's
point-view observations, so a view that saw more of the object contributes proportionally
more. Eq. (crossview) as written is an unweighted mean over per-view object embeddings.

The two coincide when each view contributes equally many points; otherwise the
implemented value is a point-count-weighted variant. Recovering the literal form requires
`seq_merge`/`hierarchical_merge` to return the frame→object correspondence so the
per-frame mask embeddings `F_masks` (already unit-normalized) can be averaged directly.
That is a change to HOV-SG's core merge machinery and was left out of scope; it is the
top-ranked recommendation in §7.

### Eq. (coherence): `m'_i`, `P^coh_i = σ(α m'_i)` — **Fully compliant**

Visual prototypes, leave-one-out exclusion, τ-gated competitors over instantiated classes
only, same α and σ, and explicit undefined handling — all verified numerically.

### Signal combination — **Implemented**

`Graph.recompute_object_probability` computes the fused object probability `p_obj` from
the current detection, cross-view, vocabulary-membership, semantic, and coherence
signals. Room propagation reads only this authoritative fused value. It groups objects
by their assigned label, merges connected spatial near-duplicate components, uses the
maximum `p_obj` per component, and applies noisy-OR plus its fully-correlated limit.

---

## 6. Testing and validation

### Command

```bash
python -m pytest tests/ -q
```

### Result

```
101 passed in 6.74s
```

All tests pass. **No tests were skipped, and none could not be run.**

### Breakdown

| File | Tests | Scope |
|---|---|---|
| `tests/test_semantic_uncertainty.py` | 26 | Semantic + membership units; exact equation agreement; τ selection; α handling; log-sum-exp stability; OOV cases; entropy regression guard |
| `tests/test_cross_view_consistency.py` | 10 | Per-view dedup; agreement/disagreement; unnormalized-mean requirement; low-support flag; [0,1] property test |
| `tests/test_detection_uncertainty.py` | 12 | Clamping/NaN; accumulation; pooled-mean merge; merge-order independence |
| `tests/test_label_coherence.py` | 11 | Leave-one-out; visual-not-text prototypes; singleton/missing prototype; τ exclusion; relabel/delete/merge cache freshness |
| `tests/test_object_uncertainty_end_to_end.py` | 13 | Multi-view accumulation → object node → merge → recompute → JSON round-trip; idempotence; complement invariant; no-fusion assertion |
| `tests/test_object_semantic_metadata.py` | 6 | Object serialization |
| `tests/test_negative_labels.py` | 7 | Common-noun filter; normalized storage; cache versioning; count-limit selection; scene independence |
| `tests/test_room_containment*.py` | 16 | Room-level propagation (adjacent) |

### Coverage against the required list

| Required | Where |
|---|---|
| Exact numerical tests with synthetic embeddings | `test_semantic_matches_equation_on_a_worked_three_class_example`, `test_membership_matches_the_ratio_of_partition_sums`, `test_leave_one_out_margin_matches_the_equation` |
| Boundary / degenerate cases | `test_degenerate_embedding_is_undefined_not_a_fabricated_probability`, `test_cross_view_edge_cases`, `test_undefined_is_none_never_zero_or_one` |
| Synonym exclusion for τ | `test_tau_selects_which_class_competes`, `test_synonym_classes_are_excluded_from_competing_prototypes`, `test_synonym_class_is_never_the_semantic_runner_up` |
| CLIP logit-scale handling | `test_clip_logit_scale_is_applied_to_the_margin_not_the_cosine`, `test_membership_is_numerically_stable_at_the_clip_logit_scale` |
| Out-of-vocabulary examples | `test_out_of_vocabulary_object_shifts_mass_to_the_negative_bank`, `test_membership_and_semantic_confidence_are_distinct_quantities` |
| Cross-view agreement / disagreement | `test_agreeing_views_give_one_and_opposing_views_give_zero`, `test_orthogonal_views_give_the_norm_of_the_mean` |
| Leave-one-out prototypes | `test_object_is_excluded_from_its_own_class_prototype` |
| Single-instance / missing prototype | `test_single_instance_class_is_undefined`, `test_missing_competing_prototype_is_undefined` |
| Merge and relabel | `test_detection_merge_is_order_independent_across_a_chain`, `test_prototypes_refresh_after_relabeling`, `test_relabeling_refreshes_semantic_and_coherence` |
| Serialization / deserialization | `test_full_signal_round_trip_through_json`, `test_merged_object_recomputes_correctly_after_a_round_trip` |
| Entropy regression | `test_no_entropy_is_used_anywhere_in_the_uncertainty_implementation` |
| End-to-end creation + multi-view update | `test_multi_view_accumulation_feeds_the_object_node`, `test_later_views_update_an_existing_accumulator_incrementally`, `test_segment_objects_populates_every_signal` |

### Remaining coverage gaps

1. **No test runs the real SAM/CLIP pipeline.** Every graph-level test uses synthetic
   embeddings and stubbed `get_label_feats`; checkpoints and HM3D data are not present in
   this environment. The frame loop in `create_feature_map` is therefore exercised only
   through its extracted helpers, not end-to-end.
2. **The negative bank is never built from the real lexicon in tests** — `_candidate_words`
   is monkeypatched. The real mining path was validated manually (§7, item 3) but not in CI.
3. **No calibration test.** Whether `σ(100·m)` matches observed correctness rates cannot
   be measured without ground-truth labels.

---

## 7. Risks and recommendations

Ranked by priority.

**1 — `m_i` granularity (Partially compliant).** Cross-view Consistency averages
point-view observations rather than per-view object embeddings, so views are weighted by
how much of the object they saw. *Recommendation:* have `seq_merge`/`hierarchical_merge`
return the `(frame, mask) → object` correspondence and accumulate `frames_feats[f][m]`
(already unit-normalized) directly per object. This makes eq. (crossview) exact and
removes the point-level accumulator entirely — a smaller memory footprint as well.
Moderate change to core merge code; recommend doing it before any calibration study.

**2 — Uncalibrated detector scores.** SAM's `predicted_iou` predicts mask-quality IoU, not
P(object exists). Using it as `P^det` is the definition's premise, but the numbers are
unlikely to be calibrated. *Recommendation:* measure reliability against HM3D ground truth
before treating `P^det` as a probability; consider Platt scaling if it is miscalibrated.
Note that recalibration would break the "no fitted parameter" property the semantic signal
deliberately maintains, so keep any fitted mapping confined to this signal.

**3 — Negative-bank size shifts `P_mem` globally.** Eq. (membership) compares partition
*sums*, so `|N| / |C|` acts as a prior. With `HM3DSEM_LABELS` (`|C| = 1624`) and the
corrected mining, `|N| ≈ 5.3k` — a ~3:1 ratio (previously ~12:1, and 73% of that bank was
proper nouns). *Recommendation:* sweep `vocab_membership_negative_label_count` against
ground-truth OOV objects and pin a justified value; do not leave `null` for a paper
result. Also verify the corrected bank once it rebuilds — inspect
`hovsg/labels/negative_label_words.json` for residual noise.

**4 — Vocabulary-membership cost.** Each object costs a `(|N| × D)` matvec. The bank is
now normalized once per run rather than per object, so the remaining cost is the matvec
itself. *Recommendation:* if object counts grow, batch all objects into one
`(#objects × D) @ (D × |N|)` GEMM instead of looping.

**5 — Merge behaviour for embeddings.** `Object.__add__` still does
`mean([self.embedding, other.embedding])`, which for a chain of N merges weights
constituents 1/2, 1/4, 1/8 … — the same defect that was fixed for `P_det`. This is
pre-existing HOV-SG behaviour affecting `v_i` itself, so every signal reading `v_i`
inherits it. *Recommendation:* weight by point count, mirroring the detection fix. Left
unchanged here because it alters HOV-SG's core object representation and its retrieval
results, which is beyond an uncertainty audit. **Note:** `merge_objects_graph` is `false`
by default, so this path is inactive in the default configuration.

**6 — Semantic/coherence dependence.** The two estimate the same labelling-error event.
No code combines them today. *Recommendation:* keep it that way, and if a downstream
consumer needs one number, select rather than multiply.

**7 — Stale prototype caches.** `recompute_label_coherence` rebuilds everything on each
call, so there is no partial-update staleness. The residual risk is *forgetting to call it*
after mutating objects outside `build_graph`. *Recommendation:* the current mitigation —
`Object.__add__` nulls its own coherence fields so a missed recomputation surfaces as
`None` rather than a stale number — is adequate; a graph-level dirty flag would be
belt-and-braces.

**8 — Undefined-signal handling downstream.** `propagate_semantic_uncertainty_to_rooms`
skips objects with `p_obj is None`, and excludes classes whose every assigned object is
undefined. Spatial duplicates are merged as connected components before noisy-OR.
*Recommendation:* surface the skip count in the pipeline summary.

**9 — Calibration evaluation requires data not present here.** Measuring whether α = 100
matches observed correctness rates needs HM3D ground-truth instance labels aligned to
predicted objects. `hovsg/eval/hm3dsem_evaluator.py` provides the alignment machinery;
the calibration study itself is not implemented.

---

## Appendix — Modified and added files

**Modified**

- `hovsg/utils/uncertainty.py` — public field-name constants; `normalize_rows`;
  `assume_normalized` fast path; undefined handling for degenerate embeddings and empty
  eligible sets; expanded equation docstrings.
- `hovsg/utils/cross_view_consistency.py` — per-view deduplication; `point_count`-based
  low-support flag; non-finite guard; contiguity check on in-place scatter-add.
- `hovsg/utils/detection_uncertainty.py` — `object_confidence_sum_from_points`,
  `confidence_from_sum`; removed the unused `object_confidence_from_points`.
- `hovsg/utils/negative_labels.py` — `is_common_noun_lemma`; normalized bank storage;
  `mining_version`; distance-based count limit; untrusted-cache rebuild.
- `hovsg/graph/object.py` — `detection_conf_sum`/`detection_point_count`/
  `cross_view_point_count` fields with serialization; pooled-mean merge; shared field
  tuples.
- `hovsg/graph/graph.py` — normalized text-bank caching; `_assign_object_semantic_signals`
  and `_normalized_negative_text_feats` helpers; degenerate-embedding label handling;
  point-count plumbing; undefined-tolerant room propagation.
- `config/create_graph.yaml` — documented every uncertainty parameter and the `|N|/|C|`
  sensitivity.
- `tests/test_semantic_uncertainty.py`, `tests/test_cross_view_consistency.py`,
  `tests/test_detection_uncertainty.py`, `tests/test_negative_labels.py` — updated for
  corrected behaviour, plus new equation-agreement tests.

**Added**

- `tests/test_label_coherence.py` (11 tests)
- `tests/test_object_uncertainty_end_to_end.py` (13 tests)
- `OBJECT_LEVEL_UNCERTAINTY_IMPLEMENTATION_REPORT.md` (this file)

**Repository-wide rename**: `c_sem`/`c_det`/`c_mem`/`c_coh`/`c_view` →
`p_sem`/`p_det`/`p_mem`/`p_coh`/`p_view`, so confidence reads as `P` and uncertainty as
`U = 1 − P`. No aliases retained.
