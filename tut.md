# Detection uncertainty in HOV-SG

The current pipeline derives detection confidence from SAM's per-mask
`predicted_iou`. Point-level confidence is accumulated while the feature map is
created, averaged over each final object point cloud, and stored as `c_det`.
Detection uncertainty is its complement, `u_det = 1 - c_det`.

Semantic labels are assigned independently from CLIP image/text similarities.
The final object stores the assigned label index and the semantic margin
confidence `c_sem`, with `u_sem = 1 - c_sem`.

Frame masks are merged into persistent point-cloud objects before object
creation. Object merges average and renormalize embeddings and average the
available detection confidence values. Semantic fields are cleared and
recomputed from the merged embedding before graph serialization.

The final object contract is therefore the four confidence fields `c_sem`,
`u_sem`, `c_det`, and `u_det`, together with the supporting assigned-label and
semantic-margin fields. The implementation does not preserve a per-frame label
history or expose a separate label-vote uncertainty signal.
