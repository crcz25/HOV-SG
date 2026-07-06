1. End-to-end pipeline
application/create_graph.py is a thin Hydra entrypoint — it resolves dataset/save paths, instantiates Graph(cfg), then calls, in order:

create_feature_map() — builds the scene representation.
save_masked_pcds / save_full_pcd / save_full_pcd_feats — checkpoints to disk.
build_graph(save_path) (skipped for Replica/ScanNet, which only need the flat feature map for their own evaluators) — this does the actual hierarchical decomposition:
segment_floors() → histogram peaks along the vertical axis → Floor objects.
segment_rooms(floor) per floor → 2D wall-skeleton + watershed segmentation → Room objects, each with a set of representative CLIP view embeddings (k-means over per-frame global CLIP features of frames near the room, room.embeddings / room.represent_images).
segment_objects() → assigns every 3D mask to a floor (by z-range) then a room (by 2D polygon overlap, falling back to nearest room centroid for outliers), classifies it via identify_object(), and creates an Object.
optional room.merge_objects() — merges same-named, spatially-overlapping objects within a room.
create_graph() — builds the networkx.Graph (building → floor → room → object).
create_nav_graph() — a separate free-space Voronoi graph, unrelated to semantics.
save_graph() — serializes everything to disk.
2. Main functions/classes and responsibilities
Graph (hovsg/graph/graph.py, 1445 lines) is the orchestrator and owns almost all state: the CLIP model, SAM model, dataset loader, full_pcd, full_feats_array (per-point fused CLIP features over the whole scene), mask_pcds/mask_feats (per-candidate-object 3D point clouds and CLIP features), and floors/rooms/objects lists plus the networkx graph.

create_feature_map(): For each keyframe, runs SAM + extract_feats_per_pixel (ConceptFusion-style: local per-mask crop CLIP feature blended with global image CLIP feature, softmax-weighted by their cosine similarity, then scattered to a dense per-pixel feature map). These per-pixel features are back-projected and averaged over all frames into full_feats_array. Independently, per-frame 3D masks (frames_pcd) are geometrically merged (hierarchical_merge/seq_merge in graph_utils.py) into self.mask_pcds. Per-mask CLIP features are then recomputed from scratch: for each final merged mask, nearby points are looked up in full_feats_array and denoised via feats_denoise_dbscan (DBSCAN in cosine space, keep the largest cluster, take its mean) → self.mask_feats.
segment_objects(): calls identify_object() (argmax cosine similarity between mask_feats[i] and a fixed vocabulary's text features) to name each object, then instantiates Object, setting .pcd, .vertices, .embedding = mask_feats[i].
Object (hovsg/graph/object.py): object_id, room_id, name, gt_name (unused GT placeholder), vertices, embedding (single CLIP vector), pcd. save()/load() round-trip to .ply + .json. __add__ merges two objects by concatenating point clouds and naively averaging embeddings (np.mean([e1, e2]), not renormalized, not weighted by point count).

Room: holds objects list, embeddings (k representative view-level CLIP vectors), represent_images, plus merge_objects() and room-type inference methods.

Floor: holds rooms, pcd, vertical extent.

3. Where labels, CLIP embeddings, crops, and semantics live
Crops: sam_utils.crop_all_bounding_boxs produces, per SAM mask per frame, both an unmasked bbox crop and a background-blocked crop (512×512).
Per-crop CLIP embeddings: sam_clip_feats_extractor.extract_feats_per_pixel encodes both crop variants via get_img_feats_batch, blends them (clip_masked_weight config), and blends further with the whole-image global feature — this happens per frame, producing F_masks.
Critical finding: F_masks is collected into frames_feats in create_feature_map() (graph.py:198) but is never read again — confirmed by grep, it's dead code. The actual object embedding used downstream (self.mask_feats, later Object.embedding) is instead reconstructed after 3D merge from full_feats_array, which is itself a per-point running average of the ConceptFusion per-pixel features across all frames. So there is no single "the crop" for a merged object — the embedding is an aggregate over many frames/pixels, several transformations removed from any individual image crop.
Text label embeddings: hovsg/utils/label_feats.py::get_label_feats builds/caches (as .npy) CLIP text embeddings for a chosen vocabulary (cfg.pipeline.obj_labels, e.g. HM3DSEM_LABELS — a 1625-row CSV of Matterport/HM3D category counts, hovsg/labels/HM3D_CountsOfObjectTypes.csv) via get_text_feats_multiple_templates.
Label assignment: Graph.identify_object() computes similarity = object_feat @ text_feats.T and returns only classes[argmax(similarity)] — the similarity vector, including the winning score, is thrown away immediately after being computed.
I confirmed the HM3DSEM label vocabulary genuinely has the near-synonym problem the plan warns about: desk/workstation/desk chair/computer desk, sofa/couch/sofa chair/l-shaped sofa, etc. — the plan's "remove near-synonym pairs" control will matter a lot here.

4. Final graph representation
self.graph is an undirected networkx.Graph whose nodes are the Python Floor/Room/Object instances themselves (plus a sentinel 0 = "building"), not string IDs. Node attribute dicts only carry name/type; all real payload lives on the instances. self.objects, room.objects, and the graph nodes all reference the same instances (no copies), so mutating an Object in place propagates everywhere consistently. On disk, each Object is one .ply + one .json (object_id, vertices, room_id, name, embedding) — this JSON schema is the authoritative persisted contract.

5. Where changes would be needed for semantic uncertainty
Location	Change
hovsg/graph/object.py	Add label_cos_sim / semantic_uncertainty fields to __init__; add to save()/load() metadata dict (with .get()-style backward compatibility for old saved graphs — there's already an untracked hovsg/data/hm3dsem/outputs/ directory that may contain graphs saved under the old schema).
hovsg/graph/graph.py::identify_object()	Currently discards the similarity vector. Needs to also surface the winning cosine similarity (and ideally the full vector, for entropy) — either change its return signature or compute entropy inline in the caller since text_feats/classes are already in scope there.
hovsg/graph/graph.py::segment_objects()	Right after object.embedding = self.mask_feats[mask_idx], compute cosine similarity + entropy and store on the Object.
hovsg/graph/room.py::merge_objects() / Object.__add__	Merging averages embeddings without renormalizing. The uncertainty score should be recomputed from the merged embedding, not averaged from the two pre-merge scores — otherwise it's inconsistent with how the label itself would be re-derived.
build_graph() ordering	Best inserted as a pass over self.objects after merge_objects() and before create_graph(), so the stored score reflects final merged embeddings, not stale per-mask ones.
6. Assumptions, edge cases, risks
No literal "object crop" exists post-merge. Implementing the plan literally ("CLIP image embedding from the object/image crop") isn't directly possible without re-plumbing frame/mask provenance through the merge pipeline (hierarchical_merge/seq_merge only ever pass point clouds, never features, between merge stages). The pragmatic reading is to treat object.embedding (the aggregated, denoised, multi-view CLIP feature) as the "image embedding" side — which is also exactly what identify_object() already uses for classification, so the disagreement score is self-consistent with how the label was chosen in the first place. This is a deviation from a literal single-crop signal and worth flagging as a deliberate design choice.
Normalization: CLIP encodings are unit-normalized at extraction time, but feats_denoise_dbscan's cluster-mean and Object.__add__'s two-embedding average do not renormalize afterward. Cosine similarity must be computed as a true dot/(‖a‖‖b‖), not a raw dot product (unlike identify_object, which assumes normalization and uses a plain dot product) — small enough drift to matter for an entropy signal.
Degenerate embeddings: extract_feats_per_pixel/create_feature_map has a branch that fills a mask's features with an all-zero vector when no points are found (feats.shape[0] == 0). Cosine similarity against a zero vector is undefined — needs an explicit guard.
Entropy needs a temperature. Raw CLIP cosine similarities cluster in a narrow band (~0.2–0.35); a softmax over raw similarities will look near-uniform regardless of correctness. Standard practice (CLIP zero-shot) applies a scale (e.g. logit = 100 × cosine) before softmax — this is a free parameter the plan doesn't pin, so it's worth deciding/tuning explicitly rather than defaulting silently.
Closed vocabulary, not truly open: identify_object always argmaxes over a fixed label list (cfg.pipeline.obj_labels), so the entropy signal is a function of that vocabulary's size and redundancy, not "open-vocabulary" in the free-text sense. The HM3DSEM CSV vocabulary (1625 entries) has heavy synonym redundancy (confirmed above) — this directly matches the plan's Kill #1 "near-synonym" control and should be run explicitly against this vocabulary before trusting the raw signal.
Reuse, don't recompute: text_feats/classes for the assigned label are already computed once per scene in segment_objects() — the cosine similarity and entropy add negligible cost (no new CLIP inference needed), as long as the code is threaded through rather than re-deriving text embeddings per object.
Proposed implementation plan
Extend Object.__init__/save/load with label_cos_sim: float and semantic_uncertainty: float (default None), backward-compatible on load.
Add a small helper (e.g. in hovsg/utils/clip_utils.py or a new hovsg/utils/uncertainty.py) that takes a (re-normalized) object embedding + the full label similarity vector and returns (top1_cos_sim, entropy), with a configurable softmax temperature and a zero-norm guard.
Modify identify_object() (or add a sibling method) to return the similarity vector alongside the label, avoiding a second CLIP/text lookup.
In segment_objects(), after label assignment, compute and store both values on the Object.
Add a Graph method (e.g. recompute_semantic_uncertainty()) called once after merge_objects() in build_graph(), iterating self.objects to refresh scores post-merge using the merged, renormalized embedding.
Update Room.merge_objects()/Object.__add__ to renormalize the averaged embedding (needed for correctness of the recompute step regardless of uncertainty).
Surface the new fields through save_graph/load_graph round-trip (already covered by step 1 if Object.save/load is updated).
