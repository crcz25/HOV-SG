# 3DSG Uncertainty

## The Nugget Idea

> A robot's scene graph already contains uncertainty signals coming from detections, object positioning and open vocabulary semantic association (defined as cross-modal disagreement between visual and language embeddings). Propagating it through the containment hierarchy **calibrates** room-level beliefs without any additional inference.

**Framing:** "Epistemic Mapping". The robot builds not just a semantic map but a map of what it knows and doesn't know, structured by the scene's own topology. Differentiates from geometric SLAM uncertainty.


## The Core Claim

> Existing 3DSG systems perform better on a downstream room-level task when their own uncertainty signals are propagated through the containment hierarchy, compared to the same systems running without any uncertainty.

The novelty is the result that the signals already present in these systems, if surfaced and propagated, measurably improve downstream performance. The aggregation method is a design choice within that claim: start with the simplest possible (Bernoulli), and if it already beats the no-uncertainty baselines, it becomes the internal baseline for richer aggregation schemes.

**Baselines:** each 3DSG system as-is, no uncertainty propagation.

**Proposed:** same system + uncertainty propagation (Bernoulli first, then refined).

**Downstream task:** room-level object queries (e.g. "does this room contain a chair?") evaluated with ECE (calibration) and task accuracy.


## Step 1 -- Kill Experiments (validate the signals)

Each kill tests whether one uncertainty signal is actually informative: does it predict errors at the object level? Run them independently before building anything. If a signal dies, drop it from propagation. The kills also serve as the paper's justification: a reviewer will ask *why* propagation helps, and the kills provide the answer signal by signal.


### Kill #1 - Semantic Uncertainty

Take one open-vocabulary method's output on any dataset with ground-truth labels (ScanNet, Replica, HM3D). For each object node, extract the CLIP visual embedding and the CLIP text embedding of the assigned label. Compute cosine distance -- that is the **disagreement score**.

Check whether this score predicts labeling errors. Spearman ρ between disagreement score and whether the node is wrongly labeled (1) or correct (0). Rather than a fixed threshold, use a **bootstrap confidence interval**: resample the dataset many times, compute ρ each time, and report the 95% interval. If the lower bound is above zero, the signal is distinguishable from noise. Also report a p-value for the null hypothesis ρ = 0. Apply the same procedure after removing near-synonym pairs (desk/workstation, sofa/couch, where CLIP distances collapse regardless of correctness). If the signal disappears after removal, it was driven by synonyms not by genuine uncertainty. Apply the same bootstrap criterion in Kill #2 and Kill #3.

If the signal survives:
- **Calibration study:** bin nodes into deciles by disagreement score, plot accuracy per bin. A useful signal produces a monotonically decreasing curve.
- **Category breakdown:** which object categories have a well-calibrated signal, and which don't? This is where the 3DSG-specific insight lives.
- **Abstention strategy:** for nodes above a disagreement threshold, re-query or abstain. Measure accuracy vs. recall cost.


### Kill #2 - Detection Uncertainty

Same structure as Kill #1, but the signal is detector confidence and the error is a detection error (false positive or missed object).

For each detected object in an existing 3DSG system, Spearman ρ between detector confidence score and whether the detection is correct (1) or wrong (0). Same thresholds as Kill #1. If the signal is too weak, detector scores cannot be trusted as propagation input, drop them.

If the signal survives:
- **Calibration study:** bin detections by confidence, plot precision per bin. A well-calibrated detector gives a monotonically increasing curve.
- **Category breakdown:** which categories are systematically overconfident? An overconfident category will bias room-level beliefs if left uncorrected.


### Kill #3 - Positioning / Room Assignment Uncertainty

Tests whether proximity to a room boundary predicts room assignment errors. Room segmentations are geometrically fuzzy at doorways and transitional spaces, hence an object near a boundary may be assigned to the wrong room not because the detector failed but because the containment edge in the graph is uncertain.

For each object node, compute a **boundary proximity score**: distance from the object's centroid to the nearest room boundary. Spearman ρ between this score and whether the room assignment is wrong (1) or correct (0). If there is no correlation, the containment edges are clean enough to ignore, drop this signal.

*What "nearest room boundary" means is system-specific. In Hydra, rooms are clusters of place nodes in the free-space graph, so the boundary is the interface between clusters. Use the distance to the nearest place node belonging to a different room. In systems that output a dense room segmentation mask, use the Euclidean distance to the nearest voxel labeled as a different room. In systems with explicit structural elements (doors, openings), distance to the nearest door is a reasonable proxy. Pin the definition per system when running the experiment.*

If the signal survives:
- **Boundary case analysis:** what fraction of misassigned objects are within a threshold distance of a boundary? Does error rate drop sharply beyond that margin?
- **Room type breakdown:** are certain transitions (bedroom/corridor, kitchen/dining) systematically harder?


## Step 2 -- Downstream Task with Bernoulli Propagation

Once the kills are done, take the surviving signals and propagate them from object nodes to room nodes using the simplest possible aggregation:

> P(room contains a chair) = 1 − ∏(1 − pᵢ) for all objects i in the room

For the positioning signal, weight each object's contribution by its boundary proximity score before combining (objects deep inside the room count more than objects on the margin).

**Before running the main comparison, check signal correlation.** Compute pairwise Spearman correlation between all surviving signals across the dataset (detection score, semantic disagreement score, boundary proximity score). If any two signals are highly correlated (e.g., spearman between detection and semantic > 0.5), they are likely responding to the same underlying cause (e.g., detection confidence and semantic disagreement both firing on visually ambiguous objects). In that case, drop the weaker signal or take the maximum of the two rather than multiplying their complements: combining correlated signals as if they were independent double-counts the same uncertainty and will distort the propagated confidence.

**Stress test with ECE before the main experiment.** Once propagation is set up, compute ECE on the room-level scores before running the full downstream comparison. ECE answers "are the propagated confidence scores honest?" cheaply. If ECE is far above the no-uncertainty baseline (e.g., your 80%-confident predictions are right only 40% of the time), something is broken upstream: the independence assumption may be badly violated, or a signal passed the kill but is miscalibrated at the room level. Fix it before running the other comparison. Use ECE as a development diagnostic only; it is not the headline metric in the paper.

**The key comparison:** system with Bernoulli propagation vs. the same system with no uncertainty. If Bernoulli already outperforms the no-uncertainty baseline on ECE and task accuracy, the core claim is established. Bernoulli then becomes the internal baseline for Step 3.

Also compare against:
- **(a) Count-and-threshold:** fraction of objects with score > 0.5, used directly as room confidence.
- **(b) Temperature scaling:** learnable scalar applied to (a) on a validation set.


## Step 3 -- Refined Aggregation (if Bernoulli works)

The goal is to ask how much the aggregation method matters on top of the already-validated signal.

Options to explore:
- **Poisson-binomial:** exact distribution over the number of correct objects in a room, rather than the Bernoulli approximation.
- **Dirichlet / Gaussian propagation:** for continuous or multi-class signals (semantic categories rather than binary presence).
- **Learned edge weights:** allow the graph edges to be weighted by signal reliability, trained on a small validation set.

Compare each against Bernoulli. If none improve significantly, Bernoulli is the method and the story is about the signals, not the aggregation.


## Key Papers

1. **arXiv:2409.11972**: Millan-Romera et al., uncertainty-aware emergent concepts in factorized 3DSGs via GNNs.

2. **arXiv:2311.10018**: Marques et al., "On the Overconfidence Problem in Semantic 3D Mapping". Establishes the evaluation protocol: ECE and reliability diagrams on ScanNet/Matterport3D.

3. **arXiv:2402.03840**: Belief Scene Graphs 1 and 2. Learned room-level object distributions. Main baseline to beat in detection uncertainty, it must explain why the closed-form propagation is preferable to their learned approach.

4. **arXiv:2402.04655**: DAC, "Open-Vocabulary Calibration for Fine-tuned CLIP". Uses distance between predicted labels and base classes as a calibration signal.

5. **arXiv:2503.19764**: OpenLex3D. Tiered evaluation benchmark for open-vocabulary 3D scene representations. Useful for semantic uncertainty idea

6. **arXiv:2603.25450**: "Cross-Model Disagreement as a Label-Free Correctness Signal". Closest conceptual prior to semantic uncertainty (cross-model rather than cross-modal). We can frame the semantic uncertainty as the cross-modal instantiation of this principle inside 3DSG nodes.

7. **Farquhar et al., Nature 2024**: Semantic entropy for hallucination detection in LLMs. Conceptual anchor for semantic uncertainty "free signal" framing.


## Positioning

| Paper | What they do | What they don't do | Relation to this work |
|---|---|---|---|
| Millan-Romera 2409.11972 | GNN infers room boundaries from geometric planes and injects them as factors into a SLAM backend; "uncertainty" = covariances for geometric SLAM optimization | No semantic labeling, no open-vocabulary, no ECE or calibration evaluation | Different problem -- geometric room detection vs. semantic label calibration. Overlap is terminological. Not a threat. |
| Marques 2311.10018 (ICRA 2024) | Calibration study (ECE, reliability diagrams) on semantic 3D maps; demonstrates overconfidence at the voxel/object level on ScanNet/Matterport3D | Stops at the object level -- never ascends to room or floor; no scene graph structure | Closest methodological prior. Establishes the evaluation protocol you will use. Your contribution is going one level up: does the calibration problem compound or dissolve as you move up the hierarchy? |
| Belief Scene Graphs 2402.03840 (ICRA 2024) | GCN predicts which objects are likely present in rooms not yet observed -- probability distribution over missing content | Not about calibrating what has been observed; requires training data; no ECE | Complementary, not competing. They ask "what is probably there but unseen?" You ask "how confident should I be about what I have seen?" A reviewer may ask whether better-calibrated beliefs improve their object-search task -- that is a good robot experiment. |
| Belief Scene Graphs 2505.02405 (ICRA 2025) | Extends the above to predict spatial placement of missing objects; adds LLM-derived commonsense ontology | Same as above -- prediction of absent content, not calibration of present detections | Same relationship as the first BSG paper. Same group (Saucedo et al.). |
| DAC 2402.04655 (ICML 2024) | Uses CLIP distance between a predicted label and base class prototypes as a calibration signal for fine-tuned CLIP classifiers | Applied to image classification, not 3DSG nodes; no scene structure, no hierarchy, no robotics context | Partial prior to Idea 08. The core signal (CLIP distance predicts confidence) is related. Your differentiation: (1) instantiated on 3DSG node embeddings from open-vocabulary pipelines, (2) cross-modal (visual vs. text embedding of the same node), (3) propagated through the graph hierarchy, (4) evaluated for robot task utility. |

**The gap these five papers leave open:** no existing work propagates semantic label uncertainty through the 3DSG hierarchy and evaluates calibration (ECE, reliability diagrams) at the room and floor levels. Marques comes closest but stops at the object level. The others either address a different question (prediction of missing content) or a different domain (geometric SLAM, image classification).