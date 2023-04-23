import os
import json
import pickle
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from scipy import stats
from .base_dataset import PCData


class S3DIS(PCData):

    def __init__(self, config):

        base_config = getattr(config, "base_parameters")
        super().__init__(base_config)

        s3dis_parameters = getattr(config, "s3dis_parameters")
        for k, v in s3dis_parameters.items():
            setattr(self, k, v)

    def extract_feature_vecs(self):

        ret = self.load_feature_vec_dict()
        if ret:
            return
        feat_vec_image_dict = {}
        print("Extracting Feature Vectors!")
        for file_name in self.all_images:
            print(self.main_2d_path + file_name)
            im = Image.open(self.main_2d_path+file_name)
            feature_vec = self.sim_object.get_sim_vec_single(im)
            name_split = file_name.split('/')
            scene = name_split[0]
            further_split = name_split[3].split('_')
            camera = further_split[1]
            room = further_split[2] + '_' + further_split[3]
            frame = further_split[5]
            feat_vec_image_dict[f"{scene}_{room}_{camera}_{frame}"] = feature_vec

        self.feat_vec_image_dict = feat_vec_image_dict
        # lets dump
        with open(self.feature_vec_dict_file, 'wb') as handle:
            pickle.dump(self.feat_vec_image_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_data_stats(self):

        ret = self.load_data_stats()
        if ret:  # if we managed to load then no need to run again!
            return

        f = open('./datasets/s3dis_regions.json')
        ulabel_region = json.load(f)

        s3dis_stats = {}
        total_point_number = 0
        total_region_number = 0
        for folder in self.train_scenes:
            curr_dir = self.main_3d_path + folder + '/supervoxel'
            annot_dir = self.main_3d_path  + folder + '/labels'
            onlyfiles = [f for f in listdir(curr_dir) if isfile(join(curr_dir, f))]

            for room in onlyfiles:
                supvox = np.load(curr_dir + '/' + room)
                annots = np.load(annot_dir + '/' + room)
                preserving_labels = ulabel_region[f'{folder}#{room[:-4]}']

                (unique, counts) = np.unique(supvox, return_counts=True)
                (annot_unique, annot_counts) = np.unique(annots, return_counts=True)
                dict_annots = {}
                for au, ac in zip(annot_unique, annot_counts):
                    dict_annots[self.label_2_name[au]] = ac

                indices_preserving_labels = [unique.tolist().index(x) for x in preserving_labels]
                unique = unique[indices_preserving_labels]
                counts = counts[indices_preserving_labels]

                frequencies = np.asarray((unique, counts)).T
                d = {}
                key_name = folder + '_' + room[:-4]
                d[f"{key_name}"] = {}
                d[f"{key_name}"]['area'] = frequencies[:, 1].sum()

                mask = np.isin(supvox, preserving_labels)
                if frequencies[:, 1].sum() != mask.sum():
                    print('Something is wrong about AREA!')

                d[f"{key_name}"]['frequencies'] = frequencies
                d[f"{key_name}"]['preserving_labels'] = preserving_labels

                d[f"{key_name}"]['annot_stats'] = dict_annots

                total_point_number += frequencies[:, 1].sum()
                total_region_number += len(preserving_labels)

                s3dis_stats.update(d)

        self.data_stats = s3dis_stats
        f = open(self.data_stats_file, "wb")
        pickle.dump(self.data_stats, f)
        f.close()
        print(f'Point Num in Total {total_point_number}')
        print(f'Region Num in Total {total_region_number}')

    def extract_scene_clusters(self):

        ret = self.load_cluster_centers()
        if ret:  # if we managed to load then no need to run again!
            return

        self.all_scenes = self.get_scenes()
        self.cluster_centers = {}
        total_scene_number = len(self.all_scenes)
        all_scene_keys = list(self.all_scenes.keys())
        for ind, s1 in enumerate(all_scene_keys):
            im_feats = np.squeeze(np.asarray(self.all_scenes[s1]), axis=1)

            cluster_num = self.cluster_num
            if len(im_feats) < cluster_num:
                cluster_num = len(im_feats)

            self.cluster_centers[s1] = self.cluster_scene(cluster_num, im_feats)

        f = open(self.cluster_centers_file, "wb")
        pickle.dump(self.cluster_centers, f)
        f.close()

    def get_scenes(self):
        all_scenes = {}
        self.load_feature_vec_dict()
        for scene in self.train_scenes:
            for room in self.rooms:
                for i in range(self.max_room_num):
                    values = None
                    kk = scene + '_' + room + str(i) + '_'
                    values = [value for key, value in self.feat_vec_image_dict.items() if kk.lower() in key.lower()]
                    if values:
                        all_scenes[kk[:-1]] = values
        return all_scenes

    def extract_scene_attributes(self):

        ret = self.load_scene_attributes()
        if ret:  # if we managed to load then no need to run again!
            return

        self.load_cluster_centers()
        scene_attributes = {}
        for scene in self.train_scenes:
            for room in self.rooms:
                for i in range(self.max_room_num):
                    kk = scene + '_' + room + str(i)
                    xbb = self.cluster_centers.get(kk, None)

                    if xbb is not None:
                        xbb = self.cluster_centers[kk]
                        curr_dim = len(xbb)
                        ordered_D = self.sim_object.calculate_dino_aff_matrix_from_feats(xbb)
                        final_distance_list = list(ordered_D[np.triu_indices(curr_dim, 1)])

                        d = {}
                        d[f"{kk}"] = {}
                        d[f"{kk}"]['mean']  = stats.describe(final_distance_list).mean
                        d[f"{kk}"]['variance']  = stats.describe(final_distance_list).variance
                        d[f"{kk}"]['minmax'] = stats.describe(final_distance_list).minmax
                        scene_attributes.update(d)

        self.scene_attributes = scene_attributes
        f = open(self.scene_attributes_file, "wb")
        pickle.dump(self.scene_attributes, f)
        f.close()
        print(f'Len of final rooms {len(self.scene_attributes)}')

    def extract_pairwise_scene_attributes(self):

        ret = self.load_pairwise_scene_attributes()
        if ret:  # if we managed to load then no need to run again!
            return

        self.all_scenes = self.get_scenes()
        all_scene_keys = list(self.all_scenes.keys())
        total_scene_number = len(all_scene_keys)

        self.load_cluster_centers()

        pairwise_scene_attributes = {}

        for ind, s1 in enumerate(all_scene_keys):
            for st in range(ind + 1, total_scene_number):
                s2 = all_scene_keys[st]
                kk = f'{s1}*{s2}'

                cluster_centers_s1 = self.cluster_centers[s1]
                cluster_centers_s2 = self.cluster_centers[s2]

                curr_dim = len(cluster_centers_s1) + len(cluster_centers_s2)
                xbb = np.zeros((curr_dim, cluster_centers_s1.shape[1]), dtype=np.float32)
                for ii, v in enumerate(cluster_centers_s1):
                    xbb[ii] = v
                for ii, v in enumerate(cluster_centers_s2):
                    xbb[len(cluster_centers_s1) + ii] = v

                ordered_D = self.sim_object.calculate_dino_aff_matrix_from_feats(xbb)

                final_distance_list = list(
                    ordered_D[len(cluster_centers_s1):, 0: len(cluster_centers_s2)].flatten())

                d = {}
                d[f"{kk}"] = {}
                d[f"{kk}"]['mean'] = stats.describe(final_distance_list).mean
                d[f"{kk}"]['variance'] = stats.describe(final_distance_list).variance
                d[f"{kk}"]['minmax'] = stats.describe(final_distance_list).minmax
                pairwise_scene_attributes.update(d)

        self.pairwise_scene_attributes = pairwise_scene_attributes
        # create a binary pickle file
        f = open(self.pairwise_scene_attributes_file, "wb")
        pickle.dump(self.pairwise_scene_attributes, f)
        f.close()
        print(f'Len of final rooms {len(self.pairwise_scene_attributes)}')

    def create_initial_set(self, scene_list):

        self.load_data_stats()

        scan_num = len(self.data_stats)
        all_keys = list(self.data_stats.keys())
        all_values = list(self.data_stats.values())

        selected_samples = []
        for scn in scene_list:
            selected_samples.append(all_keys.index(scn))

        path = os.path.join('.', self.seed_name)
        os.mkdir(path)

        f = open(path + "/init_label_scan.json", "w")
        fu = open(path + "/init_ulabel_scan.json", "w")
        f.write("[\n")
        fu.write("[\n")

        for j in range(scan_num):
            curr_name = all_keys[j]
            splits = curr_name.split('_')
            if j in selected_samples:
                f.write(f'"{splits[0]}_{splits[1]}/coords/{splits[2]}_{splits[3]}.npy",\n')
            else:
                fu.write(f'"{splits[0]}_{splits[1]}/coords/{splits[2]}_{splits[3]}.npy",\n')

        f.seek(f.tell() - 2)
        fu.seek(fu.tell() - 2)
        f.write("\n]\n")
        fu.write("\n]\n")
        f.close()
        fu.close()

        f = open(path+"/init_label_region.json", "w")
        fu = open(path+"/init_ulabel_region.json", "w")
        f.write("{")
        fu.write("{")

        total_point_num = 0
        total_region_num = 0
        for j in range(scan_num):
            curr_name = all_keys[j]
            splits = curr_name.split('_')
            supervoxel_list = str(all_values[j]['frequencies'][:,0].tolist())
            if j in selected_samples:
                f.write(f'"{splits[0]}_{splits[1]}#{splits[2]}_{splits[3]}": {supervoxel_list},')
                total_point_num+= all_values[j]['area']
                total_region_num += len(all_values[j]['frequencies'])
            else:
                fu.write(f'"{splits[0]}_{splits[1]}#{splits[2]}_{splits[3]}": {supervoxel_list},')

        f.seek(f.tell() - 1)
        fu.seek(fu.tell() - 1)
        f.write("}")
        fu.write("}")
        f.close()
        fu.close()

        fi = open(path + "/info.txt", "w")
        fi.write(f'Region Num: {total_region_num} and Point Num: {total_point_num} in the current set')
        fi.close()

    def prepare_data_for_optimization(self):

        self.load_scene_attributes()
        self.load_data_stats()
        self.load_pairwise_scene_attributes()

        self.all_scenes = self.get_scenes()
        all_scene_keys = list(self.all_scenes.keys())
        total_scene_number = len(all_scene_keys)

        pair_scores = []
        all_pairs = []
        for i, scene_i in enumerate(all_scene_keys):
            for j in range(i + 1, total_scene_number):

                scene_j = all_scene_keys[j]
                dsim_i = 1 - self.scene_attributes[scene_i]['mean']
                dsim_j = 1 - self.scene_attributes[scene_j]['mean']

                dsim = dsim_i * dsim_j

                kk = f'{scene_i}*{scene_j}'

                pairwise_sim = self.pairwise_scene_attributes[kk]['mean']
                pairwise_dsim = 1 - pairwise_sim
                final_score = pairwise_dsim * dsim

                pair_scores.append(final_score)
                all_pairs.append((scene_i, scene_j))

        return all_pairs, self.data_stats, pair_scores, self.reduction_size, self.target_point_num




