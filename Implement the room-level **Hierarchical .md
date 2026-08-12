# Room-level fused-probability propagation

`Graph.propagate_object_probabilities_to_rooms()` walks the rooms and calls
`Room.compute_class_containment_beliefs()` on each. A room groups the objects
assigned to it by `label_idx`, which is the set O(r, c) of eq. (noisyor) in
`propagation.tex`, and applies the noisy-OR of that equation to the fused
object probabilities q_i = `p_obj` of the group:

    b(r, c) = 1 - prod_{o_i in O(r, c)} (1 - q_i)

Each room stores one `class_containment_belief` entry per class instantiated
in it, holding the class ID, the label carried by the contributing object
nodes, and the belief value. Objects whose q_i is undefined contribute no
factor and are left out. The propagation has no parameters.
