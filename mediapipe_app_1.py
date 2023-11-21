import cv2
from collections import deque
import mediapipe as mp
import numpy as np
from src.utils import get_images, get_overlay
from src.config import *
import torch

# 初始化 MediaPipe Hands 模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 判断是否支持 GPU，加载模型
if torch.cuda.is_available():
    model = torch.load("trained_models/whole_model_quickdraw")
else:
    model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
model.eval()
predicted_class = None

# 打开摄像头
cap = cv2.VideoCapture(0)
points = deque(maxlen=512)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
is_drawing = False
is_shown = False
class_images = get_images("images", CLASSES)

# 使用 MediaPipe Hands 模块处理手部信息
with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            continue

        # 将图像的可写性标志设置为 False。这是因为 MediaPipe 库的处理可能会修改输入图像，但在此处我们希望防止修改原始图像，因此将其设置为不可写。
        image.flags.writeable = False
        # 将图像从 BGR（OpenCV 默认颜色通道顺序）转换为 RGB 颜色通道顺序。MediaPipe 库使用 RGB 格式的图像进行处理。
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 使用 MediaPipe 中的手部检测器（Hands）处理图像，返回检测结果。results 包含了关于检测到的手部的信息，例如手部关键点的位置等。
        results = hands.process(image)

        # 在图像上绘制手部标注
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 遍历每只手的21个关节点
                for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * 640), int(landmark.y * 480)
                    # 在关节点旁边显示坐标信息
                    cv2.putText(image, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                                1, cv2.LINE_AA)
                    cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

                    # # 输出每只手每个关节点的坐标
                    # print(f"Hand {hand_idx + 1}, Landmark {landmark_idx + 1}: ({x}, {y})")

            for hand_landmarks in results.multi_hand_landmarks:
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[12].y < \
                        hand_landmarks.landmark[11].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                    if len(points):
                        is_drawing = False
                        is_shown = True
                        # 处理绘制结果
                        # 将画布转换为灰度图
                        canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        # 中值滤波
                        canvas_gs = cv2.medianBlur(canvas_gs, 9)
                        # 高斯滤波
                        canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0)
                        # 获取非零元素的坐标
                        ys, xs = np.nonzero(canvas_gs)
                        # 如果存在非零元素
                        if len(ys) and len(xs):
                            # 计算坐标范围
                            min_y = np.min(ys)
                            max_y = np.max(ys)
                            min_x = np.min(xs)
                            max_x = np.max(xs)
                            # 裁剪图像
                            cropped_image = canvas_gs[min_y:max_y, min_x:max_x]
                            # 调整图像大小为28x28
                            cropped_image = cv2.resize(cropped_image, (28, 28))
                            cropped_image = np.array(cropped_image, dtype=np.float32)[None, None, :, :]
                            cropped_image = torch.from_numpy(cropped_image)
                            # 模型推理
                            logits = model(cropped_image)
                            predicted_class = torch.argmax(logits[0])
                            # 重置绘制点和画布
                            points = deque(maxlen=512)
                            canvas = np.zeros((480, 640, 3), dtype=np.uint8)

                else:
                    # 处于绘制状态，记录手指位置
                    is_drawing = True
                    is_shown = False
                    points.append((int(hand_landmarks.landmark[8].x * 640), int(hand_landmarks.landmark[8].y * 480)))
                    for i in range(1, len(points)):
                        cv2.line(image, points[i - 1], points[i], (0, 255, 0), 2)
                        cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 5)
                # 在图像上绘制手部标注
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                # 如果不在绘制状态且已显示绘制结果
                if not is_drawing and is_shown:
                    cv2.putText(image, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5,
                                cv2.LINE_AA)
                    # 在图像上显示绘制结果
                    image[5:65, 490:550] = get_overlay(image[5:65, 490:550], class_images[predicted_class], (60, 60))

        # 在水平方向翻转图像，使其呈现自拍视图
        cv2.imshow('MediaPipe Hands', image)
        # 按下 ESC 键退出循环
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 释放摄像头资源
cap.release()
