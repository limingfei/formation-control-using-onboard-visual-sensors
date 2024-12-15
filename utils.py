import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_grad_cam.utils.image import show_cam_on_image
from grad_cam import GradCAM
def obtain_relative_pose(follower,leader,info=None):

    follower_positon = follower.getPosition()
    leader_positon = leader.getPosition()
    if info is not None:
        info['follower_x'] = follower_positon[0]
        info['follower_y'] = follower_positon[1]
        info['leader_x'] = leader_positon[0]
        info['leader_y'] = leader_positon[1]
    follower_orien = np.array(follower.getOrientation())
    follower_orien = np.reshape(follower_orien,(3,3))
    leader_orien = np.array(leader.getOrientation())
    leader_orien = np.reshape(leader_orien,(3,3))
    position_a = follower_positon
    rotA = follower_orien
    position_b = leader_positon
    rotB = leader_orien

    # 机器人A的位置和旋转矩阵
    x1, y1, z1 = position_a # 例子中的值
    rotA = np.array(rotA)
    rotB = np.array(rotB)
 

    # 机器人B的位置和旋转矩阵
    x2, y2, z2 = position_b  # 例子中的值
  

    # 计算位置向量的差
    delta_position = np.array([x2 - x1, y2 - y1, z2 - z1])

    # 使用A的旋转矩阵的转置将delta_position旋转到A的局部坐标系
    local_position = rotA.T.dot(delta_position)

    # 计算局部旋转矩阵
    local_orientation = rotA.T.dot(rotB)
    r = R.from_matrix(local_orientation)
    el = r.as_euler('xyz')

    # 输出结果
    return local_position,el[-1]
def set_mecanum_wheel_speeds(vx, vy, omega,wheels):
    # 计算每个轮子的速度
    wheel_radius = 0.254/2 # 轮子半径，需要根据实际模型调整
    wheel_distance = 0.27+0.27  # 轮子距离中心的距离，同样需要调整

    fl_speed = (vx - vy - (wheel_distance * omega)) / wheel_radius
    fr_speed = (vx + vy + (wheel_distance * omega)) / wheel_radius
    bl_speed = (vx + vy - (wheel_distance * omega)) / wheel_radius
    br_speed = (vx - vy + (wheel_distance * omega)) / wheel_radius

    # 设置轮子速度
    # print(fl_speed,fr_speed,bl_speed,br_speed)
    for wheel,speed in zip(wheels,(fl_speed,fr_speed,bl_speed,br_speed)):
        wheel.setPosition(float('inf'))
        wheel.setVelocity(speed)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class MyDataset(Dataset):
    def __init__(self, img,low_features, teacher_action,relative_pose):
        self.img = img
        self.teacher_action = teacher_action
        self.low_features = low_features
        self.relative_pose = relative_pose

    def __len__(self):
        return len(self.teacher_action)

    def __getitem__(self, idx):
        img = self.img[idx]
        teacher_action = int(self.teacher_action[idx].item())
        low_features = self.low_features[idx]
        relative_pose = self.relative_pose[idx]
        return img,low_features, teacher_action,relative_pose
    
def random_occlusion(image, occlusion_prob=0.5):
    """
    对图像进行随机遮挡
    :param image: 输入图像
    :param occlusion_prob: 遮挡的概率
    :param occlusion_size: 遮挡的大小 (宽度, 高度)
    :return: 遮挡后的图像
    """
    h, w, = image.shape
    occluded_image = image.copy()
    occlusion_size=np.random.randint(20,100,size=(2))
    if np.random.random() < occlusion_prob:
        # 随机生成遮挡区域的左上角坐标
        top_left_x = np.random.randint(0, w - occlusion_size[0])
        top_left_y = np.random.randint(0, h - occlusion_size[1])

        # 随机生成遮挡区域的右下角坐标
        bottom_right_x = top_left_x + occlusion_size[0]
        bottom_right_y = top_left_y + occlusion_size[1]

        # 用黑色矩形遮挡图像区域
        occluded_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = np.random.randint(0,255,size=(bottom_right_y-top_left_y, bottom_right_x-top_left_x))

    return occluded_image

def random_bright(image):
    brightness_factor = np.random.uniform(0.8, 1.2)
    image = image.astype(np.float32)
    # 调整亮度
    image = image * brightness_factor
    # 确保像素值在[0, 255]范围内
    image = np.clip(image, 0, 255)
    return image
def random_contrast(image):
    contrast_factor = np.random.uniform(0.8, 1.2)
    # 将图像转换为浮点数以避免截断
    image = image.astype(np.float32)
    # 计算图像的平均值
    mean = np.mean(image)
    # 调整对比度
    image = (image - mean) * contrast_factor + mean
    # 确保像素值在[0, 255]范围内
    image = np.clip(image, 0, 255)
    # 将图像转换回uint8
    return image
def random_gaussian_blur(image,max_kernel_size=7):
    kernel_size = np.random.choice(range(1, max_kernel_size + 1, 2))
    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

class AttentionVis(object):
    def __init__(self,model):

        target_layers = [model.network[2]]
        self.cam = GradCAM(model=model, target_layers=target_layers)
    def visual(self,img,img_tensor,low_features):
        img = img.reshape(120,210,1)
        
        img = np.float32(img)
        # print(img.shape) 
        grayscale_cam = self.cam(inputs=[img_tensor,low_features],
                        targets=None,
                        eigen_smooth=False,
                        aug_smooth=False)
        grayscale_cam = grayscale_cam[0,:]
        cam_image = show_cam_on_image(img, grayscale_cam)
        return cam_image
def add_snow(image, severity=0.5):
    """
    在图像中添加雪花效果，severity 控制雪花的恶劣程度（0-1）
    :param image: 输入图像
    :param severity: 雪花恶劣程度，0-1 之间的值，0表示无雪，1表示极强的下雪天气
    :return: 添加雪花效果的图像
    """
    snow_image = image.copy()
    height, width, _ = snow_image.shape
    
    # 根据恶劣程度控制雪花数量
    num_snowflakes = int(severity * 1500)  # 最大5000个雪花
    
    for _ in range(num_snowflakes):
        # 随机选择雪花的位置
        x = np.random.randint(0, width - 1)
        y = np.random.randint(0, height - 1)
        
        # 根据恶劣程度控制雪花的大小（更强的雪会有更大的雪花）
        mean_size = severity*1.2
        std_dev_size = severity*0.5
        size = int(np.random.normal(mean_size, std_dev_size))  # 生成雪花的大小
        size = max(0, min(size, 2))  # 限制大小范围在 [2, 15]
                
        # size = np.random.randint(0.1, int(2 + severity * 0.1))  # 增加雪花最大尺寸
        
        # 添加雪花（白色圆点）
        cv2.circle(snow_image, (x, y), size, (255, 255, 255), -1)
    
    return snow_image
        # 显示图片
