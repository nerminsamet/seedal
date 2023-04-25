from dino_model.fe_dino import DinoModel
import numpy as np
from PIL import Image
import faiss


class SimilarityModel:
    def __init__(self, sim_model_name='dino', feat_type='cls'):
        self.sim_model_name = sim_model_name
        self.feat_type = feat_type
        self.load_sim_model()
        self.d = None

    def get_sim_vec_single(self, im_name_or_data):
        # Prepare an image
        image = self.load_image(im_name_or_data)
        if image is None:
            print('No Image')
            return np.zeros((1, self.d), dtype=np.float32)
        # test image
        feature_vec = self.sim_model(image).cpu().data.numpy().astype(np.float32)  # convert to numpy array
        return feature_vec

    def load_sim_model(self):
        if self.sim_model_name == 'dino':
            self.sim_model = self.load_sim_model_dino()
            self.d = 768
        else:
            raise Exception("sim_model_name must be dino!")

    def load_sim_model_dino(self):
        # Build models
        sim_model = DinoModel(self.feat_type)  # eval mode (batch norm uses moving mean/variance)
        return sim_model

    def set_sim_model_feat_type_dino(self, feat_type):
        self.feat_type = feat_type
        self.sim_model.feat_type = self.feat_type

    def load_image(self, image_path_or_data):
        try:
            if isinstance(image_path_or_data, str):
                img = Image.open(image_path_or_data)
                image = img.convert('RGB')
            elif isinstance(image_path_or_data, Image.Image):
                image = image_path_or_data.convert('RGB')
            else:
                raise Exception("image type must be str or PIL.Image!")

            return image

        except Exception as e:
            return None

    def calculate_dino_aff_matrix_from_feats(self, feats):
        curr_dim = len(feats)
        faiss.normalize_L2(feats)
        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats)  # add vectors to the index
        lims, D, I = index.range_search(feats, -1)
        ordered_D = np.reshape(D, (curr_dim, curr_dim))
        return ordered_D



