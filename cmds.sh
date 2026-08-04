for s in 00824-Dd4bFSTQ8gi 00829-QaLdnwvtxbs 00843-DYehNKdT76V 00847-bCPU9suPUw9 00849-a8BtkwhxdRV 00861-GLAQ4DNUx5U 00862-LT9Jq6dN3Ea 00873-bxsVRursffK 00877-4ok3usBNeis 00890-6s7QHgap2fW; do
python create_hm3dsem_walks_gt.py main.scene_id=$s main.split=val main.package_path=/workspace/HOV-SG/hovsg main.raw_data_path=/workspace/HOV-SG/data/hm3d main.dataset_path=/workspace/HOV-SG/data/hm3dsem_walks || break
done

# Run scene graph creation for each scene
cd /workspace/HOV-SG && python application/create_graph.py main.dataset=hm3dsem main.dataset_path=/workspace/HOV-SG/data/hm3dsem_walks main.split=val main.scene_id=00824-Dd4bFSTQ8gi main.save_path=/workspace/HOV-SG/data/scene_graphs

TxQKHP@b1S


bash zip_scene_graphs.sh hm3dsem/00824-Dd4bFSTQ8gi
