# HOV-SG
[![Static Badge](https://img.shields.io/badge/-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2403.17846)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://hovsg.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/-Video-FF0000?logo=youtube)](https://hovsg.github.io/static/images/hovsg_rss_final.mp4)



This repository is the official implementation of the paper:

> **Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation**
>
> [Abdelrhman Werby]()&ast;, [Chenguang Huang](http://www2.informatik.uni-freiburg.de/~huang/)&ast;, [Martin Büchner](https://rl.uni-freiburg.de/people/buechner)&ast;, [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/). <br>
> &ast;Equal contribution. <br> 
> 
> *arXiv preprint arXiv:2403.17846*, 2024 <br>
> (Accepted for *Robotics: Science and Systems (RSS), Delft, Netherlands*, 2024.)

<p align="center">
  <img src="media/teaser-hovsg-white.png" alt="HOV-SG allows the construction of accurate, open-vocabulary 3D scene graphs for large-scale and multi-story environments and enables robots to effectively navigate in them with language instructions." width="600" />
</p>

## 📰 Major Updates
- **[29 Aug 2024]** **We added `hm3dsem_walks` dataset generation and hierarchical scene graph evaluation code.** <br>
Please review the updated code structure and newly added dependencies for dataset construction. <br><br>
- [01 Jul 2024] Initial release of HOV-SG including mapping and graph construction engine.

## 🏗 Setup
1. Clone and set up the HOV-SG repository
```bash
git clone https://github.com/hovsg/HOV-SG.git
cd HOV-SG

# set up virtual environment and install habitat-sim afterwards separately to avoid errors.
conda env create -f environment.yaml
conda activate hovsg
conda install habitat-sim -c conda-forge -c aihabitat

# set up the HOV-SG python package
pip install -e .
```

### VS Code Dev Container with CUDA/NVIDIA GPU support
This repository includes a VS Code Dev Containers setup under `.devcontainer/`.
It uses `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04` as the base image,
creates the `hovsg` conda environment, installs Habitat-Sim and the Python
dependencies used by the project, and runs `pip install -e .` automatically
after the workspace is mounted.
The container creates `/opt/conda/envs/hovsg` directly from the repository
`environment.yaml`. For newer NVIDIA GPUs such as RTX 50-series cards, the
environment keeps Python 3.9 for Habitat-Sim compatibility and installs
PyTorch `2.8.0+cu128`, torchvision `0.23.0+cu128`, and torchaudio
`2.8.0+cu128` from the CUDA 12.8 wheel index.

Host requirements:
- Docker with the Compose v2 plugin.
- The NVIDIA driver and NVIDIA Container Toolkit installed on the host.
- VS Code with the Dev Containers extension.

Build and start from a terminal:
```bash
docker compose -f .devcontainer/docker-compose.yml build
docker compose -f .devcontainer/docker-compose.yml up -d
```

After rebuilding, verify that the Open English WordNet 2025+ lexicon is
available inside the image:
```bash
docker compose -f .devcontainer/docker-compose.yml run --rm hovsg-dev \
  python -c "import wn; w = wn.Wordnet('oewn:2025+'); print(w.synsets('dog', pos='n'))"

docker compose -f .devcontainer/docker-compose.yml run --rm hovsg-dev \
  python -c "import wn; print(wn.lexicons())"
```
The first command should print a non-empty list of synsets, and the second
should include `oewn:2025+`.

For GUI rendering on a Linux/X11 host, allow the container user to connect to
the host display before starting or rebuilding the container:
```bash
xhost +SI:localuser:$(id -un)
```

The compose service forwards `DISPLAY` and mounts `/tmp/.X11-unix`, so GUI
tools such as Matplotlib, Open3D, and PyVista can open windows on the host X
server. Revoke the display permission when you are done:
```bash
xhost -SI:localuser:$(id -un)
```

Open in VS Code:
1. Open this repository folder in VS Code.
2. Run `Dev Containers: Reopen in Container` from the Command Palette.
3. VS Code will build the image if needed, start the compose service, mount the
   repository at `/workspace/HOV-SG`, and use `/opt/conda/envs/hovsg/bin/python`
   as the Python interpreter.

The project source is bind-mounted from the host, so edits made inside the
container persist in the repository checkout. The top-level host `data/` and
`checkpoints/` directories are also bind-mounted explicitly to
`/workspace/HOV-SG/data` and `/workspace/HOV-SG/checkpoints`; place HM3DSem,
Replica, ScanNet, generated scene graphs, and model checkpoints there instead
of baking them into the image. Model checkpoints are not downloaded by the image
build; download the OpenCLIP and SAM checkpoints into `checkpoints/` using the
commands below when needed.

Verify GPU access inside the container:
```bash
nvidia-smi
python - <<'PY'
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device")
PY
```

### OpenCLIP
HOV-SG uses the Open CLIP model to extract features from RGB-D frames. To download the Open CLIP model checkpoint `CLIP-ViT-H-14-laion2B-s32B-b79K` please refer to [Open CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K).
```bash
mkdir checkpoints
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true -O checkpoints/temp_open_clip_pytorch_model.bin && mv checkpoints/temp_open_clip_pytorch_model.bin checkpoints/laion2b_s32b_b79k.bin
```
Another option is to use the OVSeg fine-tuned Open CLIP model, which is available under [here](https://github.com/facebookresearch/ov-seg):
```bash
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/17C9ACGcN7Rk4UT4pYD_7hn3ytTa3pFb5/view -O checkpoints/ovseg_clip.pth
```

### SAM
HOV-SG uses [SAM](https://github.com/facebookresearch/segment-anything) to generate class-agnostic masks for the RGB-D frames. To download the SAM model checkpoint `sam_v2` execute the following:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

## 🖼️ Dataset Preparation

### Habitat Matterport 3D Semantics
HOV-SG takes posed RGB-D sequences as input. In order to produce hierarchical multi-story scenes we make use of the Habitat 3D Semantics dataset ([HM3DSem](https://aihabitat.org/datasets/hm3d-semantics/)). 

- Download the [Habitat Matterport 3D Semantics](https://github.com/matterport/habitat-matterport-3dresearch) dataset. More specifically, download through the links corresponding to these filenames: [hm3d-val-habitat-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar), [hm3d-val-semantic-annots-v0.2.tar](https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar), [hm3d-val-semantic-configs-v0.2.tar](	https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar).
    <details>
    <summary>Make sure that the raw HM3D dataset has the following structure:</summary>
    
    ```
    ├── hm3d
    │   ├── hm3d_annotated_basis.scene_dataset_config.json # this file is necessary
    │   ├── val
    │   │   └── 00824-Dd4bFSTQ8gi
    │   │         ├── Dd4bFSTQ8gi.basis.glb
    │   │         ├── Dd4bFSTQ8gi.basis.navmesh
    │   │         ├── Dd4bFSTQ8gi.glb
    │   │         ├── Dd4bFSTQ8gi.semantic.glb
    │   │         └── Dd4bFSTQ8gi.semantic.txt
            ...
        ...
    ...
    ```

    </details>
We used the following scenes from the Habitat Matterport 3D Semantics dataset in our evaluation:
<details>
  <summary>Show Scenes ID</summary>
  
  1. `00824-Dd4bFSTQ8gi`
  2. `00829-QaLdnwvtxbs`
  3. `00843-DYehNKdT76V`
  4. `00861-GLAQ4DNUx5U`
  5. `00862-LT9Jq6dN3Ea`
  6. `00873-bxsVRursffK`
  7. `00877-4ok3usBNeis`
  8. `00890-6s7QHgap2fW`

</details>

#### Automatic walk and ground-truth preparation

The preparation scripts discover scene directories automatically; no scene IDs, repository-relative paths, or per-scene configuration edits are required. A raw scene is eligible for rendering when it contains exactly one `*.basis.glb` mesh, a matching `*.basis.navmesh`, matching `*.semantic.glb` and `*.semantic.txt` files, and the dataset root contains a `*.scene_dataset_config.json`. A trajectory file at `<pose_dir>/<scene_id>.txt` contains one whitespace-separated 4×4 camera-to-world matrix per line. A supplied trajectory always wins and is rendered in full; a scene without one gets a deterministic whole-building coverage trajectory synthesized from the navmesh, except for the fixed eight-scene evaluation exclusion set in the renderer, where a missing trajectory is reported as a data error instead.

The walk renderer validates every discovered scene, reports every unavailable source file, and continues after a skipped or failed scene. Walks span every storey of the building, which is what HOV-SG's hierarchical multi-story graphs are built from. Synthesized trajectories sample targets per storey so a multi-storey scene is covered evenly, and the camera is placed one sensor height above the navmesh, clamped inside the storey it stands on. The storey boundaries used are recorded in `camera_info.json`.

```bash
python hovsg/data/hm3dsem/gen_hm3dsem_walks_from_poses.py \
  --dataset-dir data/hm3d \
  --save-dir data/hm3dsem_walks \
  --pose-dir hovsg/data/hm3dsem/metadata/poses \
  --floor-metadata hovsg/data/hm3dsem/metadata/Per_Scene_Floor_Sep.csv
```

Use `--split <name>` or repeat `--scene-id <id>` to restrict discovery, and use `--dry-run` to report validity without rendering. Existing scene output is preserved unless `--overwrite` is given.

Generated trajectory coverage is controlled with `--trajectory-step`, `--trajectory-targets`, `--max-trajectory-poses`, and `--trajectory-seed`. Pass `--no-generate-missing-poses` to require externally supplied trajectories instead.

Ground-truth compilation validates that RGB, depth, semantic, and pose directories have the same non-empty frame stems before processing. It assigns every mapped region to the storey containing its mean height, then replaces the nominal storey boundaries with the extent of the regions actually observed there, and writes objects, regions, and scene metadata for every storey. Region vote and manual-label CSVs are optional enrichments; raw semantic object and region IDs remain authoritative.

Each region in `scene_info.json` includes a `centroid: [x, y, z]`: the arithmetic mean of its reconstructed 3-D points. The corresponding HOV-SG room JSON carries the mean of its final room point cloud under the same field. Both point clouds are reconstructed from the same RGB-D frames: image-space `(right, down, forward)` points are converted by `diag(1, -1, -1)` and transformed by the saved Habitat camera-to-world pose. Consequently both centroids use Habitat world coordinates in metres (`x` right, `y` up, `z` backward in the camera's zero-rotation basis); room segmentation's intermediate 2-D occupancy grid is used only to select points and does not redefine the output frame.

`scene_info.json` carries two object collections. `objects` is unchanged: the trajectory-visible objects, each backed by an `objects/<id>.ply` point cloud, and the only collection HOV-SG graph generation and the evaluator read. `all_objects` is a superset holding every annotated object of the complete HM3D-Sem semantic scene, whether or not the walk observed it, flagged by `observed_in_walk`; its unobserved entries carry the source Habitat AABB and OBB and have no `.ply`. It is ground-truth metadata for evaluation only. Unobserved objects take their storey from their annotated region when that region is on a known storey, otherwise from their centroid height against the storey separations, and keep `floor_id: null` with a warning when neither applies.

When supplied, floor metadata must contain `Scene Name` and `Separation Heights` columns, where the latter is an ordered list of floor boundaries. Region-vote metadata uses `Scene Name`, `Region #`, and `Weighted Room Proposal`; manual region metadata uses `Scene Name`, `Region #`, and `Region Category`. Missing optional region metadata leaves those category fields empty but does not remove the raw object or region annotations.

```bash
python hovsg/data/hm3dsem/create_hm3dsem_walks_gt.py \
  --dataset-dir data/hm3d \
  --walks-dir data/hm3dsem_walks \
  --floor-metadata hovsg/data/hm3dsem/metadata/Per_Scene_Floor_Sep.csv \
  --region-votes hovsg/data/hm3dsem/metadata/Per_Scene_Region_Weighted_Votes.csv \
  --region-labels hovsg/data/hm3dsem/metadata/Per_Scene_Region_Labels.csv
```

Each completed `<walks_dir>/<split>/<scene_id>` contains the aligned `rgb/`, `depth/`, `semantic/`, and `pose/` frame directories, plus `objects/`, `regions/`, `scene_rgb.ply`, `scene_panoptic.ply`, `scene_info.json`, and `semantic_label_map.csv`. Frame names retain the existing `<scene_name>_<zero-padded-index>` convention; `camera_info.json` is additional metadata used to preserve camera intrinsics and the selected floor. A failure in any one scene does not prevent the remaining discovered scenes from being processed.

Storey boundaries (`Separation Heights`: N+1 boundaries describe N storeys) are resolved from the `Scene Name,Separation Heights` CSV first, then `camera_info.json`, then a histogram of navigable heights from the navmesh. The CSV covers only the ten manually annotated scenes, and no HM3DSEM scene populates Habitat semantic levels, so the navmesh estimator is the operative source everywhere else; scored against the CSV it reproduces the annotated boundaries to within about 0.2 m.

#### Batch preparation

`scripts/process_hm3dsem_excluding_readme_eval.sh` drives both stages one scene at a time, so a per-scene failure is attributable and does not abort the rest of the run. The eight README evaluation scenes are excluded from bulk discovery but are processed when named explicitly.

```bash
# One scene
scripts/process_hm3dsem_excluding_readme_eval.sh 00824-Dd4bFSTQ8gi

# A selected list of scenes
scripts/process_hm3dsem_excluding_readme_eval.sh 00800-TEEsavR23oF 00802-wcojb4TFT35

# Every eligible validation scene except the eight README evaluation scenes
scripts/process_hm3dsem_excluding_readme_eval.sh --all
```

Outputs that validate as complete are skipped so an interrupted run can be resumed; pass `--force` to regenerate them and `--dry-run` to report the plan without writing. Use `--conda-env` when the interpreter is not already on `PATH`, and `--max-frames` to bound synthesized trajectories. The script never reads or writes `config/create_graph.yaml`; it verifies the file is unchanged at exit.

To evaluate semantic segmentation cababilities, we used [ScanNet](http://www.scan-net.org/) and [Replica](https://github.com/facebookresearch/Replica-Dataset).
### ScanNet
To get an RGBD sequence for ScanNet, download the ScanNet dataset from the [official website](http://www.scan-net.org/). The dataset contains RGB-D frames compressed as .sens files. To extract the frames, use the [SensReader/python](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python).
We used the following scenes from the ScanNet dataset:

<details>
  <summary>Show Scenes ID</summary>

  1. `scene0011_00`
  2. `scene0050_00`
  2. `scene0231_00`
  3. `scene0378_00`
  4. `scene0518_00`
</details>

### Replica
To get an RGBD sequence for Replica, Instead of the original Replica dataset, download the scanned RGB-D trajectories of the Replica dataset provided by [Nice-SLAM](https://github.com/cvg/nice-slam). It contains rendered trajectories using the mesh models provided by the original Replica datasets. 
Download the Replica RGB-D scan dataset using the downloading [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) in [Nice-SLAM](https://github.com/cvg/nice-slam#replica-1).

```bash
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -O data/Replica.zip && unzip data/Replica.zip -d data/Replica_RGBD && rm data/Replica.zip 
```

To evaluate against the ground truth semantics labels, you also need also to download the original Replica dataset from the [Replica](https://github.com/facebookresearch/Replica-Dataset) as it contains the ground truth semantics labels as .ply files.
```bash
git clone https://github.com/facebookresearch/Replica-Dataset.git data/Replica-Dataset
chmod +x data/Replica-Dataset/download.sh && data/Replica-Dataset/download.sh data/Replica_original
```
We only used the following scenes from the Replica dataset:
<details>
  <summary>Show Scenes ID</summary>
  
  1. `office0`
  2. `office1`
  3. `office2`
  4. `office3`
  5. `office4`
  6. `room0`
  7. `room1`
  8. `room2`

</details>

## 📂 Datasets file strutcre
The Data folder should have the following structure:

<details>
  <summary>Show data folder structure</summary>
  
```
├── hm3dsem_walks
│   ├── val
│   │   ├── 00824-Dd4bFSTQ8gi
│   │   │   ├── depth
│   │   │   │   ├── Dd4bFSTQ8gi_000000.png      # uint16, millimetres
│   │   │   │   ├── ...
│   │   │   ├── rgb
│   │   │   │   ├── Dd4bFSTQ8gi_000000.png
│   │   │   │   ├── ...
│   │   │   ├── semantic
│   │   │   │   ├── Dd4bFSTQ8gi_000000.npy      # uint32 instance IDs
│   │   │   │   ├── ...
│   │   │   ├── pose
│   │   │   │   ├── Dd4bFSTQ8gi_000000.txt      # flattened 4x4 camera-to-world
│   │   │   │   ├── ...
│   │   │   ├── objects                          # <object_id>.ply, all storeys
│   │   │   ├── regions                          # <region_id>.ply, all storeys
│   │   │   ├── camera_info.json
│   │   │   ├── scene_info.json                  # floor / region / object GT
│   │   │   ├── scene_panoptic.ply
│   │   │   ├── scene_rgb.ply
│   │   │   ├── semantic_label_map.csv
|   |   ├── 00829-QaLdnwvtxbs
|   |   ├── ..
├── Replica
│   ├── office0
│   │   ├── results
│   │   │   ├── depth0000.png
│   │   │   ├── ...
│   │   |   ├── rgb0000.png
│   │   |   ├── ...
│   │   ├── traj.txt
│   ├── office1
│   ├── ...
├── ScanNet
│   ├── scans
│   │   ├── scene0011_00
│   │   │   ├── color
│   │   │   │   ├── 0.jpg
│   │   │   │   ├── ...
│   │   │   ├── depth
│   │   │   │   ├── 0.png
│   │   │   │   ├── ...
│   │   │   ├── poses
│   │   │   │   ├── 0.txt
│   │   │   │   ├── ...
│   │   │   ├── internsics
│   │   │   │   ├── intrinsics_color.txt
│   │   │   │   ├── intrinsics_depth.txt
│   │   ├── ..
```

</details>



## :rocket: Run 

### Create scene graphs (only for Habitat Matterport 3D Semantics):
```bash
python application/create_graph.py main.dataset=hm3dsem main.dataset_path=data/hm3dsem_walks main.split=val main.scene_id=00824-Dd4bFSTQ8gi main.save_path=data/scene_graphs
```

`main.dataset_path` can point to the dataset root (`data/hm3dsem_walks`), the
split directory (`data/hm3dsem_walks/val`), or the exact scene directory
(`data/hm3dsem_walks/val/00824-Dd4bFSTQ8gi`). When using the dataset root or
split directory, pass the scene with `main.scene_id`.
<details>
  <summary>This will generate a scene graph for the specified RGB-D sequence and save it. The following files are generated:</summary>

```
├── graph
│   ├── floors
│   │   ├── 0.json
│   │   ├── 0.ply
│   │   ├── 1.json
│   │   ├── ...
│   ├── rooms
│   │   ├── 0_0.json
│   │   ├── 0_0.ply
│   │   ├── 0_1.json
│   │   ├── ...
│   ├── objects
│   │   ├── 0_0_0.json
│   │   ├── 0_0_0.ply
│   │   ├── 0_0_1.json
│   │   ├── ...
│   ├── nav_graph
├── tmp
├── full_feats.pt
├── mask_feats.pt
├── full_pcd.ply
├── masked_pcd.ply
```
The `graph` folder contains the generated scene graph hierarchy, the first number in the file name represents the floor number, the second number represents the room number, and the third number represents the object number. The `tmp` folder holds intermediate results obtained throughout graph construction. The `full_feats.pt` and `mask_feats.pt` contain the features extracted from the RGBD frames using the Open CLIP and SAM models. the former contains per point features and the latter contains the features for the object masks. The `full_pcd.ply` and `masked_pcd.ply` contain the point cloud representation of the RGB-D frames and the instance masks of all objects, respectively.

</details>

### Visualize scene graph
```bash
python application/visualize_graph.py graph_path=data/scene_graphs/hm3dsem/00824-Dd4bFSTQ8gi/graph
```
![hovsg_graph_vis](media/hovsg_graph_vis.gif)

### Interactive visualization of scene graphs and natural language queries

#### Setup OpenAI
In order to test graph queries with HOV-SG, you need to setup an OpenAI API account with the following steps:
1. [Sign up an OpenAI account](https://openai.com/blog/openai-api), login your account, and bind your account with at least one payment method.
2. [Get you OpenAI API keys](https://platform.openai.com/account/api-keys), copy it.
3. Open your `~/.bashrc` file, paste a new line `export OPENAI_KEY=<your copied key>`, save the file, and source it with command `source ~/.bashrc`. Another way would be to run `export OPENAI_KEY=<your copied key>` in the teminal where you want to run the query code.

#### Evaluate query against pre-built hierarchical scene graph 
```bash
python application/visualize_query_graph.py main.graph_path=data/scene_graphs/hm3dsem/00824-Dd4bFSTQ8gi/graph
```
After launching the code, you will be asked to input the hierarchical query. An example is `chair in the living room on floor 0`. You can see the visualization of the top 5 target objects and the room it lies in.
![hovsg_graph_query](media/hovsg_graph_query.gif)

### Extract feature map for semantic segmentation (only ScanNet and Replica)
```bash
python application/semantic_segmentation.py main.dataset=replica main.dataset_path=Replica/office0 main.save_path=data/sem_seg/office0
```

### Evaluate semantic segmentation (only ScanNet and Replica)
```bash
python application/eval/evaluate_sem_seg.py dataset=replica scene_name=office0 feature_map_path=data/sem_seg/office0
```

### Evaluate predicted scene graphs (only Habitat 3D Semantics)
- Define the scene identifiers and paths of ground truth and the predicted scene graph in the `config/eval_graph.yaml`.
- Run the graph evaluation method:
```bash
python application/eval/evaluate_graph.py 
```

## 📔 Abstract

Recent open-vocabulary robot mapping methods enrich dense geometric maps with pre-trained visual-language features. While these maps allow for the prediction of point-wise saliency maps when queried for a certain language concept, largescale environments and abstract queries beyond the object level still pose a considerable hurdle, ultimately limiting languagegrounded robotic navigation. In this work, we present HOVSG, a hierarchical open-vocabulary 3D scene graph mapping approach for language-grounded indoor robot navigation. Leveraging open-vocabulary vision foundation models, we first obtain state-of-the-art open-vocabulary segment-level maps in 3D and subsequently construct a 3D scene graph hierarchy consisting of floor, room, and object concepts, each enriched with openvocabulary features. Our approach is able to represent multistory buildings and allows robotic traversal of those using a cross-floor Voronoi graph. HOV-SG is evaluated on three distinct datasets and surpasses previous baselines in open-vocabulary semantic accuracy on the object, room, and floor level while producing a 75% reduction in representation size compared to dense open-vocabulary maps. In order to prove the efficacy and generalization capabilities of HOV-SG, we showcase successful long-horizon language-conditioned robot navigation within realworld multi-story environments. 

If you find our work useful, please consider citing our paper:
```
@article{werby23hovsg,
Author = {Abdelrhman Werby and Chenguang Huang and Martin Büchner and Abhinav Valada and Wolfram Burgard},
Title = {Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation},
Year = {2024},
journal = {Robotics: Science and Systems},
} 
```

## 👩‍⚖️  License

For academic usage, the code is released under the [MIT](https://opensource.org/licenses/MIT) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

This work was funded by the German Research Foundation
(DFG) Emmy Noether Program grant number 468878300, the
BrainLinks-BrainTools Center of the University of Freiburg,
and an academic grant from NVIDIA.
