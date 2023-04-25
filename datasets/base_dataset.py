import sys
sys.path.append('..')
from os import listdir
from os.path import isfile, join
import os
from similarity_model import SimilarityModel
import pickle
import pickle5 as pickle
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np

class PCData:

  def __init__(self, base_config):

      # path related attributes
      self.dataset_name = None
      self.main_2d_path = None
      self.main_3d_path = None
      self.scene_relative_path_to_rgb = None
      self.save_path = None

      # inheretied class related attributes
      self.train_scenes = None
      self.class_num = None
      self.cluster_num = None
      self.target_point_num = None
      self.reduction_size = None
      self.seed_name = None

      for k, v in base_config.items():
          setattr(self, k, v)

      isExist = os.path.exists(self.save_path)
      if not isExist:
          os.makedirs(self.save_path)

      self.sim_object = SimilarityModel()

      self.cluster_centers_file = self.save_path + f'{self.dataset_name}_scene_clusters_{self.cluster_num}.pkl'
      self.scene_attributes_file = self.save_path + f'{self.dataset_name}_attribute_dict.pkl'
      self.pairwise_scene_attributes_file = self.save_path + f'{self.dataset_name}_pairwise_attribute_dict.pkl'
      self.data_stats_file = self.save_path + f'{self.dataset_name}_stats.pkl'
      self.feature_vec_dict_file = self.save_path + f'{self.dataset_name}_{self.sim_object.sim_model_name}_features.pkl'

      # attributes to be calculated
      self.cluster_centers = None
      self.scene_attributes = None
      self.pairwise_scene_attributes = None
      self.data_stats = None
      self.feat_vec_image_dict = None

      self.all_images = self.get_all_data_rgb_names()
      self.all_scenes = None

  def load_feature_vec_dict(self):
      if os.path.exists(self.feature_vec_dict_file):
          with open(self.feature_vec_dict_file, 'rb') as handle:
              self.feat_vec_image_dict = pickle.load(handle)
              print(f'{self.feature_vec_dict_file} features are loaded')
              return True
      else:
          return False

  def load_data_stats(self):
      if os.path.exists(self.data_stats_file):
          with open(self.data_stats_file, 'rb') as handle:
              self.data_stats = pickle.load(handle)
              print(f'{self.data_stats_file} features are loaded')
              return True
      else:
          return False

  def load_cluster_centers(self):
      if os.path.exists(self.cluster_centers_file):
          with open(self.cluster_centers_file, 'rb') as handle:
              self.cluster_centers = pickle.load(handle)
              print(f'{self.cluster_centers_file} features are loaded')
              return True
      else:
          return False

  def load_scene_attributes(self):
      if os.path.exists(self.scene_attributes_file):
          with open(self.scene_attributes_file, 'rb') as handle:
              self.scene_attributes = pickle.load(handle)
              print(f'{self.scene_attributes_file} features are loaded')
              return True
      else:
          return False

  def load_pairwise_scene_attributes(self):
      if os.path.exists(self.pairwise_scene_attributes_file):
          with open(self.pairwise_scene_attributes_file, 'rb') as handle:
              self.pairwise_scene_attributes = pickle.load(handle)
              print(f'{self.pairwise_scene_attributes_file} features are loaded')
              return True
      else:
          return False

  def get_all_data_rgb_names(self):
      train_scenes = [x.lower() for x in self.train_scenes]

      all_files = []
      for folder in train_scenes:
          curr_dir = self.main_2d_path + folder + self.scene_relative_path_to_rgb
          onlyfiles = [f for f in listdir(curr_dir) if isfile(join(curr_dir, f))]
          onlyfiles = [(folder + self.scene_relative_path_to_rgb + '/' + word) for word in onlyfiles]
          all_files = all_files + onlyfiles

      return all_files

  def cluster_scene(self, cluster_num,im_feats):

      kmeans = KMeans(n_clusters=cluster_num, random_state=123).fit(im_feats)
      return kmeans.cluster_centers_

  def extract_scene_attributes(self):

      ret = self.load_scene_attributes()
      if ret:  # if we managed to load then no need to run again!
          return

      all_scenes, all_scene_keys = self.get_scenes()

      self.load_cluster_centers()

      scene_attributes = {}

      for ind, kk in enumerate(all_scene_keys):
          xbb = self.cluster_centers.get(kk, None)

          if xbb is not None:
              xbb = self.cluster_centers[kk]
              curr_dim = len(xbb)
              ordered_D = self.sim_object.calculate_dino_aff_matrix_from_feats(xbb)
              final_distance_list = list(ordered_D[np.triu_indices(curr_dim, 1)])

              d = {}
              d[f"{kk}"] = {}
              d[f"{kk}"]['mean'] = stats.describe(final_distance_list).mean
              d[f"{kk}"]['variance'] = stats.describe(final_distance_list).variance
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

      all_scenes, all_scene_keys = self.get_scenes()
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
      print(f'Len: {len(self.pairwise_scene_attributes)}')

  def prepare_data_for_optimization(self):

      self.load_scene_attributes()
      self.load_data_stats()
      self.load_pairwise_scene_attributes()

      all_scenes, all_scene_keys = self.get_scenes()
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




