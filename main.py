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
from PIL import Image

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

# access recorded IMU data
imu_data = client.getImuData()

# print recorded IMU data
# for data in imu_data:
print(imu_data)

print("position:", x, y, z)
xv = kinematic_state_groundtruth.linear_velocity.x_val  # 速度信息
yv = kinematic_state_groundtruth.linear_velocity.y_val  # 速度信息
zv = kinematic_state_groundtruth.linear_velocity.z_val  # 速度信息
xa = kinematic_state_groundtruth.linear_acceleration.x_val  # 加速度信息
ya = kinematic_state_groundtruth.linear_acceleration.y_val
za = kinematic_state_groundtruth.linear_acceleration.z_val
print("acceleration:", xa, ya, za)
print(airsim.__version__)

f = open('log.txt', 'w')
f.write("position:" + str(x) + str(y) + str(z) + '\n')
f.write("acceleration:" + str(xa) + str(ya) + str(za) + '\n')
f.close()
client.armDisarm(False)  # 上锁
client.enableApiControl(False)  # 释放控制权

path = os.getcwd()
print(path)
path = 'D:/airsimproject/record'

# Get a list of all image files in the directory
image_package = os.listdir(path)
print(path)
print("Hello, World!")
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
last_folder = folders[-1]
last_folder_path = os.path.join(path, last_folder)
image_folder_path = os.path.join(last_folder_path, "images")
os.chdir(image_folder_path)
images = [f for f in os.listdir(image_folder_path) if f.endswith('.ppm')]
image_path = os.path.join(image_folder_path, images[0])
print(image_path)


# 读取图像
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

# 找到轮廓

contours, _ = cv2.findContours(thresh[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# 找到最小矩形框
for cnt in contours:
   x, y, w, h = cv2.boundingRect(cnt)
   cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
   print("top-left vertex:","(", x, "," ,y, ")")
   print("top-right vertex:","(", x+w, "," ,y, ")")
   print("bottom-left vertex:", "(", x, "," ,y+h, ")")
   print("bottom-right vertex:", "(", x+w, "," ,y+h, ")")



# 显示图像
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()