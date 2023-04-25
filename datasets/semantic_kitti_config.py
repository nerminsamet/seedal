base_parameters = dict(
dataset_name = 'sk',
main_2d_path = '/media/sametn/17d67dc8-250f-422d-af77-f3d227a31856/SemanticKitti/data_odometry_color/sequences/',
main_3d_path = '/media/sametn/17d67dc8-250f-422d-af77-f3d227a31856/SemanticKitti/data_odometry_velodyne/dataset/sequences',
scene_relative_path_to_rgb = '/image_2',
save_path = './sk_attribute_outputs/',
train_scenes = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
class_num = 19,
cluster_num = 19,
target_point_num = 22591773,
reduction_size = 1200,
seed_name = 'sk_seed'
)

sk_parameters = dict(
sparsification_similarity_thr=0.75,
)