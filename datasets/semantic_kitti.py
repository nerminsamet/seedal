import os
import json
import pickle
import numpy as np
from PIL import Image
from .base_dataset import PCData


class SK(PCData):

    def __init__(self, config):

        base_config = getattr(config, "base_parameters")
        super().__init__(base_config)

        s3dis_parameters = getattr(config, "sk_parameters")
        for k, v in s3dis_parameters.items():
            setattr(self, k, v)

        self.cls_feature_vec_dict_file = self.save_path + f'{self.dataset_name}_{self.sim_object.sim_model_name}_cls_features.pkl'
        self.cls_feat_vec_image_dict = None

        self.sparsified_dataset_file = self.save_path + f'{self.dataset_name}_sparsified.pkl'
        self.sparsified_dataset = None

    def extract_feature_vecs(self):
        self.extract_cls_features()
        self.sparsify_dataset()
        self.extract_patch_features()

    def extract_data_stats(self):

        ret = self.load_data_stats()
        if ret:  # if we managed to load then no need to run again!
            return

        with open('./datasets/semantic_kitti_areas.json', 'r') as f:
            supvox_pts = json.load(f)
            supvox_keys = list(supvox_pts.keys())

        sk_stats = {}
        total_point_number = 0
        for im in self.sparsified_dataset:
            name_parse = im.split('/')
            seq = name_parse[0]
            im_id = name_parse[2][:-4]
            key_name = f'{seq}/velodyne/{im_id}.bin#'
            matching = [s for s in supvox_keys if key_name in s]
            area = 0
            for m in matching:
                area+=supvox_pts[m]
            d = {}
            d[f"{seq}_{self.scene_relative_path_to_rgb[1:]}_{im_id}"] = {}
            d[f"{seq}_{self.scene_relative_path_to_rgb[1:]}_{im_id}"]['area'] = area
            total_point_number+=area
            sk_stats.update(d)

        self.data_stats = sk_stats
        f = open(self.data_stats_file, "wb")
        pickle.dump(self.data_stats, f)
        f.close()
        print(f'Point Num in Total {total_point_number}')

    def extract_scene_clusters(self):
        ret = self.load_cluster_centers()
        if ret:  # if we managed to load then no need to run again!
            return
        self.load_feature_vec_dict()
        self.cluster_centers = {}
        all_scene_keys = list(self.feat_vec_image_dict.keys())
        for ind, s1 in enumerate(all_scene_keys):
            im_feats = self.feat_vec_image_dict[s1][0]

            cluster_num = self.cluster_num
            if len(im_feats) < cluster_num:
                cluster_num = len(im_feats)
            self.cluster_centers[s1] = self.cluster_scene(cluster_num, im_feats)

        f = open(self.cluster_centers_file, "wb")
        pickle.dump(self.cluster_centers, f)
        f.close()

    def extract_cls_features(self):

        ret = self.load_cls_feature_vec_dict()
        if ret:
            return
        cls_feat_vec_image_dict = {}
        print("Extracting Feature Vectors!")
        for file_name in self.all_images:
            print(self.main_2d_path + file_name)
            im = Image.open(self.main_2d_path+file_name)
            feature_vec = self.sim_object.get_sim_vec_single(im)
            name_split = file_name.split('/')
            scene = name_split[0]
            camera = name_split[1]
            frame = name_split[2].split('.')[0]
            cls_feat_vec_image_dict[f"{scene}_{camera}_{frame}"] = feature_vec

        self.cls_feat_vec_image_dict = cls_feat_vec_image_dict
        # lets dump
        with open(self.cls_feature_vec_dict_file, 'wb') as handle:
            pickle.dump(self.cls_feat_vec_image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def sparsify_dataset(self):

        ret = self.load_sparsified_dataset()
        if ret:
            return

        selected_frames = []
        for folder in self.train_scenes:
            matching = sorted([s for s in self.all_images if f'{folder}/' in s])
            p = 0
            for i, img in enumerate(matching):
                i = p
                key_i = matching[i].replace('/','_')[:-4]
                for j in range(i + 1, len(matching), 1):
                    key_j = matching[j].replace('/','_')[:-4]

                    feat_i = self.cls_feat_vec_image_dict[key_i]
                    feat_j = self.cls_feat_vec_image_dict[key_j]

                    feats = np.concatenate((feat_i, feat_j), axis=0)
                    ordered_D = self.sim_object.calculate_dino_aff_matrix_from_feats(feats)
                    sim = ordered_D[0][1]
                    if sim < self.sparsification_similarity_thr:
                        s_i = int((i + j) / 2)
                        selected_frames.append(matching[s_i])
                        p = j
                        break

                if p > len(matching):
                    break

        # lets dump
        self.sparsified_dataset = selected_frames
        with open(self.sparsified_dataset_file, 'wb') as handle:
            pickle.dump(self.sparsified_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_cls_feature_vec_dict(self):
        if os.path.exists(self.cls_feature_vec_dict_file):
            with open(self.cls_feature_vec_dict_file, 'rb') as handle:
                self.cls_feat_vec_image_dict = pickle.load(handle)
                print(f'{self.cls_feature_vec_dict_file} features are loaded')
                return True
        else:
            return False

    def load_sparsified_dataset(self):
        if os.path.exists(self.sparsified_dataset_file):
            with open(self.sparsified_dataset_file, 'rb') as handle:
                self.sparsified_dataset = pickle.load(handle)
                print(f'{self.sparsified_dataset_file} are loaded')
                return True
        else:
            return False

    def extract_patch_features(self):

        ret = self.load_feature_vec_dict()
        if ret:
            return

        ret = self.load_sparsified_dataset()
        if not ret:
            print('No sparsified dataset!')
            return

        feat_vec_image_dict = {}

        self.sim_object.set_sim_model_feat_type_dino('patch')

        print("Extracting Feature Vectors!")
        for file_name in self.sparsified_dataset:
            print(self.main_2d_path + file_name)
            im = Image.open(self.main_2d_path+file_name)
            feature_vec = self.sim_object.get_sim_vec_single(im)
            name_split = file_name.split('/')
            scene = name_split[0]
            camera = name_split[1]
            frame = name_split[2].split('.')[0]
            feat_vec_image_dict[f"{scene}_{camera}_{frame}"] = feature_vec

        self.feat_vec_image_dict = feat_vec_image_dict
        # lets dump
        with open(self.feature_vec_dict_file, 'wb') as handle:
            pickle.dump(self.feat_vec_image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.sim_object.set_sim_model_feat_type_dino('cls')

    def create_initial_set(self, selected_samples):

        f = open('./datasets/semantic_kitti_regions.json')
        all_regions = json.load(f)

        path = os.path.join('.', self.seed_name)
        os.mkdir(path)

        f = open(path+"/init_label_scan.json", "w")
        fu = open(path+"/init_ulabel_scan.json", "w")
        f.write("[\n")
        fu.write("[\n")

        all_keys = list(all_regions.keys())
        for scn in all_keys:
            search_str = scn.replace('_', '_image_2_')
            final_str = scn.replace('_','/velodyne/') + '.bin'
            if search_str in selected_samples:
                f.write(f'  "{final_str}",\n')
            else:
                fu.write(f'  "{final_str}",\n')

        f.seek(f.tell() - 2)
        fu.seek(fu.tell() - 2)
        f.write("\n]\n")
        fu.write("\n]\n")
        f.close()
        fu.close()

        f = open(path+"/init_label_large_region.json", "w")
        fu = open(path+"/init_ulabel_large_region.json", "w")
        f.write("{")
        fu.write("{")

        for i, scn in enumerate(all_regions):
            search_str = scn.replace('_','_image_2_')
            supervoxel_list = all_regions[scn]
            if search_str in selected_samples:
                f.write(f'"{scn}": {supervoxel_list}, ')
            else:
                fu.write(f'"{scn}": {supervoxel_list}, ')

        f.seek(f.tell() - 2)
        fu.seek(fu.tell() - 2)
        f.write("}")
        fu.write("}")
        f.close()
        fu.close()

    def get_scenes(self):

        sparsified_dataset =  self.sparsified_dataset
        sparsified_dataset_keys = []

        for scn in sparsified_dataset:
            sparsified_dataset_keys.append(scn.replace('/','_')[:-4])

        return sparsified_dataset, sparsified_dataset_keys





