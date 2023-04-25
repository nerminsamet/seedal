from datasets.s3dis import S3DIS
from datasets.semantic_kitti import SK
from optimization import apply_optimization
from importlib import import_module
import argparse

def main(args):

    config = import_module(f'datasets.{args.dataset}_config')

    if args.dataset == 's3dis':
        dataset_instance = dataset_instance = S3DIS(config)
    elif args.dataset == 'semantic_kitti':
        dataset_instance = dataset_instance = SK(config)
    else:
        raise NotImplementedError("Only support for s3dis and semantic kitti dataset!")

    dataset_instance.extract_feature_vecs()
    dataset_instance.extract_data_stats()
    dataset_instance.extract_scene_clusters()
    dataset_instance.extract_scene_attributes()
    dataset_instance.extract_pairwise_scene_attributes()

    all_pairs, data_stats, pair_scores, reduction_size, area_threshold = dataset_instance.prepare_data_for_optimization()
    selected_scenes = apply_optimization(all_pairs, data_stats, pair_scores, reduction_size, area_threshold)

    dataset_instance.create_initial_set(selected_scenes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SeedAL framework.')
    parser.add_argument('-d', '--dataset', choices=['s3dis', 'semantic_kitti'], default='semantic_kitti')

    args = parser.parse_args()
    main(args)
