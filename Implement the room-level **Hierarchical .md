# Room-level fused-probability propagation

`Graph.propagate_semantic_uncertainty_to_rooms()` groups assigned objects by
`label_idx`, skips objects whose fused `p_obj` is undefined, merges connected
near-duplicate components using the existing symmetric point-cloud overlap
metric, and represents each component by its maximum `p_obj`.

Each room stores a noisy-OR `class_containment_belief` entry for every
propagated class. Each entry contains the class ID, the label carried by the
contributing object nodes, and the belief value. The merge threshold is
configured by `pipeline.room_belief_merge_threshold`.
