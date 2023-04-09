# =========================================
# 通用参数设定
# =========================================
DEBUG = False # 是否开启debug模式 默认不开启
USING_OCCLUDE = False # 是否使用occlude 默认开启
INCREASE_SPACE = True # 把显示相关的显示出来 默认不开启

INCREASE_SPACE_VECTOR = True # VECTOR可视化 默认开启
RANDOM_COLOR = False

MULTITHREAD = False # 服务器用多线程 本机不用多线程
DRAW_DYNAMIC = True # 是否画出动目标

# setting 1 出line
# USING_LINE = True # 是否出road-edge-line和crossroad-line
# REMOVE_ROAD_EDGE = True # 如果USING_LINE = True 是否remove road-edge-line的横着部分
# USING_LANE_LINE = True # 是否包括车道线 默认包括

# setting 2 出分割面
USING_LINE = False # 是否出road-edge-line和crossroad-line
REMOVE_ROAD_EDGE = False # 如果USING_LINE = True 是否remove road-edge-line的横着部分
USING_LANE_LINE = False # 是否包括车道线 默认包括

# =========================================
# BEV图尺度设定
# =========================================
# XBOUND = [-64.0, 64.0, 0.25]
# YBOUND = [-32.0, 32.0, 0.25]
# ZBOUND = [-10.0, 10.0, 20.0]

XBOUND = [-32.0, 32.0, 0.125]
YBOUND = [-16.0, 16.0, 0.125]
ZBOUND = [-10.0, 10.0, 20.0]

# =========================================
# 数据集设定
# =========================================
# INDEX_START = 0
# INDEX_END = 4
# VERSION = "mini"
# NUSCENE_DATA = "D:/Data/BEV/nuscene"

INDEX_START = 0
INDEX_END = 19
VERSION = "mini"
NUSCENE_DATA = "D:\\AutoDrive\\Data\\nuscene"

# INDEX_START = 0
# INDEX_END = 34148
# VERSION = "trainval"
# NUSCENE_DATA = "/data/zdx/Data/data_nuscene"

#---------------------------------------------------#
#  1. gen raw sample
#---------------------------------------------------#
# 数据生成配置
CAMERA = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

#---------------------------------------------------#
#  2. gen label static
#---------------------------------------------------#
# STATIC 设置
# 静态提取的原始label放在visual里面
# 坐标系方向为
#    | -----> y
#    |
#    |
#    x

if USING_LANE_LINE:
    STATIC_LAYER_NAME = ['drivable_area', "lane_divider", "road_divider", "ped_crossing"]
    STATIC_LAYER_ID = [200, 90, 90, 60]
    STATIC_LAYER_NAME_SEQUENCE = ['drivable_area', "lane_divider", "road_divider","ped_crossing"]

else:
    STATIC_LAYER_NAME = ['drivable_area', "ped_crossing"]
    STATIC_LAYER_ID = [200, 60]
    STATIC_LAYER_NAME_SEQUENCE = ['drivable_area', "ped_crossing"]

#---------------------------------------------------#
#  3. gen label dynamic
#---------------------------------------------------#
# DYNAMIC 设置
# 动态提取的原始label放在gt_label里面
# 坐标系方向为
#    | -----> y
#    |
#    |
#    x
SIGMA = 3
DYNAMIC_OBJECT_LIST = ['vehicle']
DYNAMIC_OBJECT_ID = [160]
IGNORE_INDEX = 255

#---------------------------------------------------#
#  4. gen label occlusion
#---------------------------------------------------#
# 车尺寸 m
CAR_WIDTH = 1.85
CAR_LENGTH = 4.085
OCCLUSION_COLOR = 240
BLOCKING_LABELS = ["vehicle"]
TALL_NON_BLOCKING_LABELS = []

#---------------------------------------------------#
#  5. gen train label
#---------------------------------------------------#
REGION2LINE = ['drivable_area', "ped_crossing"]
KERNEL_SIZE = {"drivable_area":(7,7),"ped_crossing":(5,5)}

## 精细化人行道
REFINE_ERODE_PED_CROSSING = (3,3)
REFINE_DILATE_PED_CROSSING = (3,3)
# area cross walk remove
REMOVE_AREA_PED = 40

## 精细化路沿
REFINE_ERODE_ROAD_EDGE = (5,5)
HARRIS_NEIGHBOR_SIZE = 5
HARRIS_SOBLE_SIZE = 5
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.6

POINTS_REMOVE_RADIUS = 9
REMOVE_AREA_ROAD = 200  # area road remove
REFINE_DILATE_ROAD_EDGE = (3,3)

## 加载顺序
BEV_VIS_MULTIPLY = 30
KERNEL_SIZE_OCCLUSION = (3,3)

# =========================================
# 9. gen vector line
# =========================================
PATCH_MARGIN = 0.5
SAMPLE_DISTANCE = 0.5
# 为了让车道线不贴人行横道紧
KERNEL_SIZE_DILATION_CROSSWALK = (3,3)
THICKNESS = 20
MAXCHANNEL = 3

MIN_CROSSWALK_LENGTH = 4
CONCAT_EDGE_THRESHOLD_CROSSWALK = 4

MIN_LINE_LENGTH = 20

# 路沿处理
EXPAND_PTS_NUM = 4
MIN_ROAD_EDGE = 12
CONCAT_EDGE_THRESHOLD = 8
