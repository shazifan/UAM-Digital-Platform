import setup_path
import airsim
import cv2
import numpy as np 
import pprint
import os
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import time


class VirtualLane():

    def __init__(self, num=9, L=300, W=5, H=5, l=6, v_speed=30.0):
        self.num = num
        self.length = L
        self.width = W
        self.height = H
        self.length_unit = l
        self.num_unit = int(self.length/self.length_unit)
        self.EnVehSpeed = v_speed

    def CreateEnvInforMatrix(self, EnvVehsInfor):
        """
        环境信息矩阵：包含感知车道空间中存在环境车辆位置和速度信息
        :param
            EnvVehsInfor: Nnm_Enveh * {
                EnVehInPose<Vector3r>(float): {'x', 'y', 'z'}
                EnvVehSpeed(float): default: 30 m/s
                }
        :return EnVehInPose, Num_EnvVeh

        """
        # EnvVehSpeed = 30.0
        CountVeh = 0
        if EnvVehsInfor:
            for EnvVehicle in EnvVehsInfor:
                CountVeh += 1
                EnvVehPose = EnvVehicle.relative_pose.position
                if CountVeh == 1:
                    E = np.array([EnvVehPose.x_val, EnvVehPose.y_val, EnvVehPose.z_val], dtype=int)
                else:
                    e = np.array([EnvVehPose.x_val, EnvVehPose.y_val, EnvVehPose.z_val], dtype=int)
                    E = np.vstack([E, e])

        return E, CountVeh

    def CreatVehSpeedField(self, EnvInforMatrix):
        """
        虚拟车道中存在的速度场，坐标为（车道id，纵向量化单元id）
        :param
            EnvInforMatrix: [shape=num_EnvVeh*3(x,y,z), dtype=float]
        :return
            VehSpeedField: [shape=9*50, value = 30.0 (m/s), dtype=float]

        """
        VehSpeedField = np.zeros([self.num, self.num_unit], dtype=float)
        # print(VehSpeedField.shape)
        # EnvInforMatrix = np.array([1,0,0])
        if EnvInforMatrix.shape == (3,):
            EnvInforMatrix = EnvInforMatrix[np.newaxis, :]
        EMatrix = EnvInforMatrix[:, 1:]
        EMatrix = np.where(EMatrix > 0, 5, EMatrix)
        EMatrix = np.where(EMatrix < 0, -5, EMatrix)
        index_i = np.dot(np.array([1, math.sqrt(self.num)]),
                         1/self.width * (5+EMatrix).T)
        index_j = 1/self.length_unit * EnvInforMatrix[:, 0].T
        index_i = np.array(index_i, dtype=int)
        index_j = np.array(index_j, dtype=int)
        # print("index_i:", index_i)
        # print("index_j:", index_j)
        for p in zip(index_i, index_j):
            VehSpeedField[p[0]][p[1]] = self.EnVehSpeed   #注意存在车辆速度重叠，归为30
        # print(VehSpeedField)
        # print(VehSpeedField[VehSpeedField == 30])

        return VehSpeedField


    def CreateVLDynamicHarzardMatrix(self, VehSpeedField, alpha=100, beta=1):
        """
        定义危险传播上三角矩阵，参数alpha、beta控制危险传播衰减速度，
        与车道空间速度场加权得到车道空间动态威胁分数
        :param
            VehSpeedField:
            alpha:
            beta:
        :return:

        """
        P = np.zeros([self.num_unit, self.num_unit], dtype=float)
        for i in range(self.num_unit):
            for j in range(self.num_unit):
                if i <= j:
                    P[i, j] = alpha * math.exp(-beta * math.sqrt(j-i))
        print(P.shape)
        VLDynamicHarzardMatrix = np.dot(VehSpeedField, P.T)
        # print(VLDynamicHarzardMatrix)



        return VLDynamicHarzardMatrix




if __name__ == '__main__':

    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, 0, 2).join()

    # set camera name and image type to request images and detections
    camera_name = "0"
    image_type = airsim.ImageType.Scene

    # set detection radius in [cm]
    client.simSetDetectionFilterRadius(camera_name, image_type, 300 * 100)
    # add desired object name to detect in wild card/regex format
    client.simAddDetectionFilterMeshName(camera_name, image_type, "Env_Vehicle*")

    #Create Virtual Lanes
    VL = VirtualLane()

    # create result_save folder
    folder_path = "./detection_result_data"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        for file in os.listdir(folder_path + "/"):
            os.remove(folder_path + "/" + file)

    i = 1
    flag = 0
    while True:
        rawImage = client.simGetImage(camera_name, image_type)
        if not rawImage:
            continue
        png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
        EnvVehicles = client.simGetDetections(camera_name, image_type)
        # print(EnvVehicles)

        #环境车辆目标检测
        if EnvVehicles:
            for EnvVehicle in EnvVehicles:
                cv2.rectangle(png, (int(EnvVehicle.box2D.min.x_val), int(EnvVehicle.box2D.min.y_val)),
                              (int(EnvVehicle.box2D.max.x_val), int(EnvVehicle.box2D.max.y_val)), (255, 0, 0), 2)
                cv2.putText(png, EnvVehicle.name,
                            (int(EnvVehicle.box2D.min.x_val), int(EnvVehicle.box2D.min.y_val - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12))

        cv2.imshow("AirSim", png)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            client.simClearDetectionMeshNames(camera_name, image_type)
        elif cv2.waitKey(1) & 0xFF == ord('a'):
            client.simAddDetectionFilterMeshName(camera_name, image_type, "Env_Vehicle*")


        #基于检测信息的威胁意图提取
        E, numVeh = VL.CreateEnvInforMatrix(EnvVehicles)
        print("Env:", E)
        print("\n", numVeh)
        S = VL.CreatVehSpeedField(E)
        print(S)
        H = VL.CreateVLDynamicHarzardMatrix(S)
        print(H)

        #绘制车道实时动态威胁热力图
        i += 1
        if i % 1 == 0:     #每n次循环更新一次热力图
            if flag == 0:  #只在首次画图输出右侧颜色条
                fig = plt.figure(num=1, figsize=(6, 4))
                ax = fig.add_subplot(211)
                sns.heatmap(data=S, cmap='hot', square=True)
                ax = fig.add_subplot(212)
                sns.heatmap(data=H, cmap='hot', square=True)
                plt.pause(0.5)
                flag = 1
                i = 1
            else:
                fig = plt.figure(num=1, figsize=(8, 4))
                ax = fig.add_subplot(211)
                sns.heatmap(data=S, cmap='hot', square=True, cbar=False)
                ax = fig.add_subplot(212)
                sns.heatmap(data=H, cmap='hot', square=True, cbar=False)
                plt.pause(0.5)
                i = 1
                # plt.close()
                # plt.show()

        print("\n"+"*"*20)

    #显示摄像机目标检测结果窗口
    cv2.destroyAllWindows()


