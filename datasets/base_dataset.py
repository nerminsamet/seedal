import sys
sys.path.append('..')
from os import listdir
from os.path import isfile, join
import os
from similarity_model import SimilarityModel
import pickle
import pickle5 as pickle
from sklearn.cluster import KMeans

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



