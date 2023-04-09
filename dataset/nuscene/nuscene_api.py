from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap

# ---------------------------------------------------#
#  begin test
# ---------------------------------------------------#
nusc_map = NuScenesMap(dataroot='E:/mini', map_name='singapore-onenorth')

# step 1
fig, ax = nusc_map.render_layers(nusc_map.non_geometric_layers, figsize=1)

# step 2
bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
nusc_map.render_layers(['lane'], figsize=1, bitmap=bitmap)

# step 3
nusc_map.render_record('stop_line', nusc_map.stop_line[14]['token'], other_layers=[], bitmap=bitmap)

# step 4
patch_box = (300, 1700, 100, 100)
patch_angle = 0  # Default orientation where North is up
layer_names = ['drivable_area', 'walkway']
canvas_size = (1000, 1000)
map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)

figsize = (12, 4)
nusc_map.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=figsize, n_row=1)

# step 5
nusc_map.render_map_mask(patch_box, 45, layer_names, canvas_size, figsize=figsize, n_row=1)

# step 6
# Init nuScenes. Requires the dataset to be stored on disk.
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini' ,dataroot="E:/mini" ,verbose=True)

# Pick a sample and render the front camera image.
sample_token = nusc.sample[9]['token']
layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
camera_channel = 'CAM_FRONT'
nusc_map.render_map_in_image(nusc, sample_token, layer_names=layer_names, camera_channel=camera_channel)

# ---------------------------------------------------#
#  end test
# ---------------------------------------------------#