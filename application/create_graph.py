import os
import hydra
from omegaconf import DictConfig
from hovsg.graph.graph import Graph

# pylint: disable=all


def _is_scene_dir(path):
    return all(os.path.isdir(os.path.join(path, name)) for name in ("rgb", "depth", "pose"))


def _resolve_dataset_path(dataset_path, split, scene_id):
    dataset_path = os.path.normpath(os.path.expanduser(str(dataset_path)))
    split = str(split)
    scene_id = str(scene_id)

    candidates = [
        dataset_path,
        os.path.join(dataset_path, scene_id),
        os.path.join(dataset_path, split, scene_id),
    ]

    for candidate in candidates:
        if _is_scene_dir(candidate):
            return candidate, os.path.basename(candidate)

    return candidates[-1], scene_id


def _resolve_save_path(save_path, dataset, scene_id):
    save_path = os.path.normpath(os.path.expanduser(str(save_path)))
    dataset = str(dataset)
    scene_id = str(scene_id)

    if os.path.basename(save_path) == scene_id:
        return save_path

    if os.path.basename(os.path.dirname(save_path)) == dataset and os.path.basename(save_path) == scene_id:
        return save_path

    return os.path.join(save_path, dataset, scene_id)


@hydra.main(version_base=None, config_path="../config", config_name="create_graph")
def main(params: DictConfig):
    # create logging directory
    dataset_path, scene_id = _resolve_dataset_path(
        params.main.dataset_path,
        params.main.split,
        params.main.scene_id,
    )
    params.main.scene_id = scene_id
    params.main.dataset_path = dataset_path

    save_dir = _resolve_save_path(params.main.save_path, params.main.dataset, params.main.scene_id)
    params.main.save_path = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # create graph
    hovsg = Graph(params)
    hovsg.create_feature_map() # create feature map

    # save full point cloud, features, and masked point clouds (pcd for all objects)
    hovsg.save_masked_pcds(path=save_dir, state="both")
    hovsg.save_full_pcd(path=save_dir)
    hovsg.save_full_pcd_feats(path=save_dir)
    
    # for debugging: load preconstructed map as follows
    # hovsg.load_full_pcd(path=save_dir)
    # hovsg.load_full_pcd_feats(path=save_dir)
    # hovsg.load_masked_pcds(path=save_dir)
    
    # create graph, only if dataset is not Replia or ScanNet
    print(params.main.dataset)
    if params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
        hovsg.build_graph(save_path=save_dir)
    else:
        print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")

if __name__ == "__main__":
    main()
