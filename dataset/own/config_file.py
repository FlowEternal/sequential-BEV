import os
#---------------------------------------------------#
#  配置文件
#---------------------------------------------------#
DATA_ROOT_DIR = "D:\\Data\\BEV\\own\\pack4"

package_name = os.path.basename(DATA_ROOT_DIR)

if package_name == "pack1":
    mapping_cam = {
        "front_left": "cam1",
        "front_middle": "cam0",
        "front_right": "cam3",
        "rear_left": "cam4",
        "rear_middle": "cam5",
        "rear_right": "cam2",
    }

if package_name == "pack2":
    mapping_cam ={
        "front_left": "cam3",
        "front_middle":"cam2",
        "front_right":"cam4",
        "rear_left":"cam1",
        "rear_middle": "cam0",
        "rear_right": "cam5",
    }

if package_name == "pack3":
    mapping_cam = {
        "front_left": "cam0",
        "front_middle": "cam2",
        "front_right": "cam1",
        "rear_left": "cam3",
        "rear_middle": "cam5",
        "rear_right": "cam4",
    }

if package_name == "pack4":
    mapping_cam = {
        "front_left": "cam0",
        "front_middle": "cam2",
        "front_right": "cam1",
        "rear_left": "cam3",
        "rear_middle": "cam5",
        "rear_right": "cam4",
    }

if package_name == "pack5":
    mapping_cam = {
        "front_left": "cam3",
        "front_middle": "cam2",
        "front_right": "cam0",
        "rear_left": "cam1",
        "rear_middle": "cam4",
        "rear_right": "cam5",
    }


#---------------------------------------------------#
#  camera parameters
#---------------------------------------------------#
# default

# latest
def get_parameter_one(key):
    FocalLengthX = 1030.09247393442
    FocalLengthY = 1036.35696377094
    PrincipalPointX = 1047.06480360665
    PrincipalPointY = 566.132172532619
    CamX = 2131.26641798888
    CamY = 32.9566011847421
    CamZ = 1327.03404560729
    thetaX = -1.54491094178678
    thetaY = -0.00753667033033095
    thetaZ = -1.59565170315961

    if key == "front_left":
        FocalLengthX = 900.624232506953
        FocalLengthY = 900.498247328538
        PrincipalPointX = 970.578858957295
        PrincipalPointY = 498.725579578888
        CamX = 2019.29352063991
        CamY = 973.139355939389
        CamZ = 934.424520837635
        # thetaX = -1.44448733491901
        # thetaY = -0.0183067440294678
        # thetaZ = -0.737130569961056
        thetaX = -1.45321398117898
        thetaY = 0.0349065850398866
        thetaZ = -1.31308922311918

    if key == "front_right":
        FocalLengthX = 901.575898546532
        FocalLengthY = 901.491615109014
        PrincipalPointX = 967.763381029462
        PrincipalPointY = 532.705838222282
        CamX = 2036.91574523126
        CamY = -972.430446206866
        CamZ = 940.180657247775
        thetaX = -1.4915231935967
        thetaY = 0.00961199046001827
        thetaZ = -2.47390090589524

    if key == "rear_right":
        FocalLengthX = 902.400905805432
        FocalLengthY = 902.615639643989
        PrincipalPointX = 981.268606453966
        PrincipalPointY = 523.365840632469
        CamX = 2256.7114444692
        CamY = -979.494135666165
        CamZ = 830.313727759134
        thetaX = -1.56411589946827
        thetaY = -0.0180380518949986
        thetaZ = 2.28000958807327

    if key == "rear_middle":
        FocalLengthX = 903.760639010756
        FocalLengthY = 903.868696052557
        PrincipalPointX = 976.572423760549
        PrincipalPointY = 567.360338902163
        CamX = -865.266379022931
        CamY = 4.86436501014609
        CamZ = 959.966789148538
        thetaX = -1.6221439666884
        thetaY = -0.000219718903492932
        thetaZ = 1.57862499874746

    if key == "rear_left":
        FocalLengthX = 901.004248213564
        FocalLengthY = 900.921859752525
        PrincipalPointX = 952.229406202478
        PrincipalPointY = 541.148908618167
        CamX = 2289.89398254791
        CamY = 954.791853352053
        CamZ = 869.39298697615
        thetaX = -1.62871469591893
        thetaY = -0.0234834003933299
        thetaZ = 0.744564838186129
    return (FocalLengthX,FocalLengthY,PrincipalPointX,PrincipalPointY,CamX,CamY,CamZ,thetaX,thetaY,thetaZ)


def get_parameter_two(key):
    FocalLengthX = 1031.43546857493
    FocalLengthY = 1037.82818984353
    PrincipalPointX = 1045.35941244934
    PrincipalPointY = 568.343995918169
    CamX = 2049.03262133755
    CamY = 17.3520426233780
    CamZ = 1356.31612406103
    thetaX = -1.55183829005273
    thetaY = -0.0138965790066937
    thetaZ = -1.59854070996435

    if key == "front_left":
        FocalLengthX = 900.085844433172
        FocalLengthY = 899.820193952508
        PrincipalPointX = 972.805642060124
        PrincipalPointY = 499.331609747132
        CamX = 2017.34432157132
        CamY = 979.224293984271
        CamZ = 936.277011786295
        thetaX = -1.48549773571960
        thetaY = 0.0164573936279063
        thetaZ = -0.658202015034656

    if key == "front_right":
        FocalLengthX = 901.472801724324
        FocalLengthY = 901.372893119882
        PrincipalPointX = 966.040629073843
        PrincipalPointY = 533.224415855245
        CamX = 2019.34648697419
        CamY = -933.846337499432
        CamZ = 938.938632250135
        thetaX = -1.50100541419735
        thetaY = 0.0140042956183853
        thetaZ = -2.62038537712517

    if key == "rear_right":
        FocalLengthX = 902.774452353930
        FocalLengthY = 902.798425789316
        PrincipalPointX = 981.941451230554
        PrincipalPointY = 523.859611851894
        CamX = 2230.88961495329
        CamY = -1019.77001479576
        CamZ = 873.600761059582
        thetaX = -1.56615849846830
        thetaY = -0.0167083582090326
        thetaZ = 2.26528279331194

    if key == "rear_middle":
        FocalLengthX = 903.760639010756
        FocalLengthY = 903.868696052557
        PrincipalPointX = 976.572423760549
        PrincipalPointY = 567.360338902163
        CamX =-841.504966427994
        CamY = 8.60520731661400
        CamZ = 991.980298601423
        thetaX = -1.63253288243424
        thetaY = -0.000757329950091724
        thetaZ = 1.56585301538608

    if key == "rear_left":
        FocalLengthX = 901.308552721250
        FocalLengthY = 901.165095316731
        PrincipalPointX = 952.914590209562
        PrincipalPointY = 540.666185673394
        CamX = 2242.05700973482
        CamY = 935.943405139994
        CamZ = 875.810154515316
        thetaX = -1.62095183584400
        thetaY = -0.00651279433932462
        thetaZ =0.714579763932213
    return (FocalLengthX, FocalLengthY, PrincipalPointX, PrincipalPointY, CamX, CamY, CamZ, thetaX, thetaY, thetaZ)

func_get_parameter = None
if package_name == "pack1" or package_name == "pack2":
    func_get_parameter = get_parameter_one
else:
    func_get_parameter = get_parameter_two