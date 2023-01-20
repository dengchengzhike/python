"""
飞正方形（速度控制）
"""
import os
import sys
from random import random

import airsim
import time
import numpy as np
import cv2
import pygame

client = airsim.MultirotorClient()  # connect to the AirSim simulator
client.confirmConnection()
"""
# 初始化 Pygame 模块
pygame.init()

# 设置 Pygame 窗口的标题
 pygame.display.set_caption("AirSim Keyboard Control")

# 创建 Pygame 窗口
screen = pygame.display.set_mode((640, 480))
"""
client.enableApiControl(True)  # 获取控制权
client.armDisarm(True)  # 解锁

kinematic_state_groundtruth = client.simGetGroundTruthKinematics(vehicle_name='')
# 无人机全局位置坐标真值
x = kinematic_state_groundtruth.position.x_val  # 全局坐标系下，x轴方向的坐标
y = kinematic_state_groundtruth.position.y_val  # 全局坐标系下，y轴方向的坐标
z = kinematic_state_groundtruth.position.z_val  # 全局坐标系下，z轴方向的坐标
success = client.simSetSegmentationObjectID("target_man", 255)
success = client.simSetSegmentationObjectID("^(?!target_man).*$", -1, True)
client.startRecording()

client.takeoffAsync().join()  # 第一阶段：起飞
"""
camera = client.getCamera("0")

# 开始连续拍摄，每秒拍一张照片
camera.start_recording(interval_sec=1)
count = 0


# 在每次拍照后，将照片保存到文件夹中
def save_image(response):
    global count
    filename = os.path.join("img", f"image_{count}.png")
    count += 1
    with open(filename, "wb") as f:
        f.write(response.image_data_uint8)


# 设置 "save_image" 函数为回调函数
camera.set_image_data_callback(save_image, False)

while True:
    responses = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.Scene, pixels_as_float=False, compress=False),
        airsim.ImageRequest(0, airsim.ImageType.Infrared, pixels_as_float=False, compress=True),
        airsim.ImageRequest(0, airsim.ImageType.DepthVis, pixels_as_float=False, compress=False)])
    img_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    img_scene = img_rgb.reshape(responses[0].height, responses[0].width, 3)
    cv2.imwrite('img/scene.png', img_scene)

    fInfrared = open('img/infrared.png', 'wb')
    fInfrared.write(responses[1].image_data_uint8)
    fInfrared.close()

    image1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
    image_depth: object = image1d.reshape(responses[2].height, responses[2].width, 3)
    # 灰度图保存为图片文件
    cv2.imwrite('img/depth.png', image_depth)
    time.sleep(1)
"""

client.moveToZAsync(-3, 10).join()  # 第二阶段：上升到2米高度

"""
  # 循环处理 Pygame 事件
while True:
    # 处理 Pygame 事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
       
        # 获取键盘输入
        keys = pygame.key.get_pressed()
        if event.type == pygame.KEYUP:
            client.moveByVelocityZAsync(0, 20, -3, 1)
            print("gogogogoh")
        if event.type == pygame.KEYDOWN:
            client.moveByVelocityZAsync(0, -20, -3, 1)
        if event.type == pygame.K_LEFT:
            client.moveByVelocityZAsync(-20, 0, -3, 1)
        if event.type == pygame.K_RIGHT:
            client.moveByVelocityZAsync(20, 0, -3, 1)
        """

# 飞正方形
# client.moveToPositionAsync(5, 0, -3, 1).join()     # 第三阶段：以1m/s速度向前飞8秒钟
# client.moveToPositionAsync(5, 5, -3, 1).join()     # 第三阶段：以1m/s速度向右飞8秒钟
# client.moveToPositionAsync(0, 5, -3, 1).join()    # 第三阶段：以1m/s速度向后飞8秒钟
# client.moveToPositionAsync(0, 0, -3, 1).join()    # 第三阶段：以1m/s速度向左飞8秒钟

# 悬停 2 秒钟
# client.hoverAsync().join()  # 第四阶段：悬停6秒钟
# time.sleep(0.5)
client.landAsync().join()  # 第五阶段：降落
# 停止拍摄
client.stopRecording()

"""
# get recording path
recording_path = client.getRecordingPath()

# open recording file
recording = airsim.Recording(recording_path)
"""

# access recorded IMU data
imu_data = client.getImuData()

# print recorded IMU data
# for data in imu_data:
print(imu_data)

print("position:", x, y, z)
kinematic_state_groundtruth.linear_velocity.x_val  # 速度信息
xa = kinematic_state_groundtruth.linear_acceleration.x_val  # 加速度信息
ya = kinematic_state_groundtruth.linear_acceleration.y_val
za = kinematic_state_groundtruth.linear_acceleration.z_val
print("acceleration:", xa, ya, za)
print(airsim.__version__)
print("Hello, World!")

kinematic_state_groundtruth.orientation  # 姿态信息
kinematic_state_groundtruth.angular_velocity  # 姿态角速率信息
kinematic_state_groundtruth.angular_acceleration  # 姿态角加速度信息

f = open('log.txt', 'w')
f.write("position:" + str(x) + str(y) + str(z) + '\n')
f.write("acceleration:" + str(xa) + str(ya) + str(za) + '\n')
f.close()
client.armDisarm(False)  # 上锁
client.enableApiControl(False)  # 释放控制权
"""
#image_files = os.listdir(image_package)

# Iterate through each image file
# for image_file in image_files:
    # Extract the image number from the file name
    #img_num = int(image_file.split('_')[-1].split('.')[0])
# print(img_num)
"""
path = os.getcwd()
print(path)
image_dir = 'D:\record'

# Get a list of all image files in the directory
image_package = os.listdir(image_dir)
print(image_package)