
base_parameters = dict(
dataset_name = 's3dis',
main_2d_path = 'path/to/2D_modalities/',
main_3d_path = 'path/to/S3DIS_processed/',
scene_relative_path_to_rgb = '/data/rgb',
save_path = './s3dis_attribute_outputs/',
train_scenes = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'],
class_num = 13,
cluster_num = 13,
target_point_num = 6500000,
reduction_size = 80,
target_region_num = 790,
seed_name = 's3dis_seed'
)

s3dis_parameters = dict(
rooms = ['auditorium_', 'conferenceRoom_', 'copyRoom_', 'hallway_', 'lobby_', 'lounge_', 'office_', 'storage_', 'pantry_', 'WC_', 'openspace_'],
max_room_num = 40,
label_2_name = {0: 'ceiling',
                     1: 'floor',
                     2: 'wall',
                     3: 'beam',
                     4: 'column',
                     5: 'window',
                     6: 'door',
                     7: 'chair',
                     8: 'table',
                     9: 'bookcase',
                     10: 'sofa',
                     11: 'board',
                     12: 'clutter'},
)