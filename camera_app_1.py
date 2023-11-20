"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
from collections import deque
import cv2
import numpy as np
import torch
from pycallgraph2 import PyCallGraph, Config, GlobbingFilter
from pycallgraph2.output import GraphvizOutput
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# from src.dataset import CLASSES
from src.config import *
from src.utils import get_images, get_overlay


# 函数：解析命令行参数
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-c", "--color", type=str, choices=["green", "blue", "red"], default="green",
                        help="指定摄像头能够捕捉到的颜色，作为画笔的颜色")
    parser.add_argument("-a", "--area", type=int, default=3000, help="被捕捉对象的最小面积")
    parser.add_argument("-d", "--display", type=int, default=3, help="预测结果显示的时间（秒）")
    parser.add_argument("-s", "--canvas", type=bool, default=False, help="是否显示黑白画布")
    args = parser.parse_args()
    return args


# 主函数，执行程序
def main(opt):
    # 定义颜色范围
    if opt.color == "red":
        color_lower = np.array(RED_HSV_LOWER)
        color_upper = np.array(RED_HSV_UPPER)
        color_pointer = RED_RGB
    elif opt.color == "green":
        color_lower = np.array(GREEN_HSV_LOWER)
        color_upper = np.array(GREEN_HSV_UPPER)
        color_pointer = GREEN_RGB
    else:
        color_lower = np.array(BLUE_HSV_LOWER)
        color_upper = np.array(BLUE_HSV_UPPER)
        color_pointer = BLUE_RGB

    # 初始化用于存储检测点的deque和用于绘制的画布
    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # 从摄像头加载视频（这里使用内置摄像头）
    camera = cv2.VideoCapture(0)
    is_drawing = False
    is_shown = False

    # 加载类别的图像
    class_images = get_images("images", CLASSES)
    predicted_class = None

    # 加载模型
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()

    while True:
        # 等待键盘输入，每10毫秒检查一次
        key = cv2.waitKey(10)
        # 如果按下 'q' 键，退出循环
        if key == ord("q"):
            break
        # 如果按下空格键，切换绘制状态
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                # 如果之前已经显示过绘制结果，重新初始化绘制相关变量
                if is_shown:
                    points = deque(maxlen=512)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False

        # 如果不在绘制状态且未显示绘制结果
        if not is_drawing and not is_shown:
            # 如果有绘制的点
            if len(points):
                # 将绘制的图像转为灰度图
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # 中值滤波
                median = cv2.medianBlur(canvas_gs, 9)
                # 高斯滤波
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # 使用Otsu的阈值处理
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 寻找图像轮廓
                contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    # 找到最大的轮廓
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    # 检查最大轮廓的面积是否大于指定阈值
                    if cv2.contourArea(contour) > opt.area:
                        x, y, w, h = cv2.boundingRect(contour)
                        # 提取感兴趣区域
                        image = canvas_gs[y:y + h, x:x + w]
                        # 调整图像大小为28x28
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, None, :, :]
                        image = torch.from_numpy(image)
                        logits = model(image)
                        predicted_class = torch.argmax(logits[0])
                        is_shown = True
                    else:
                        print("绘制的物体太小，请绘制一个较大的物体！")
                        points = deque(maxlen=512)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # 读取摄像头帧
        ret, frame = camera.read()
        # 水平翻转图像
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        # 掩码处理，提取感兴趣颜色
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查是否找到任何轮廓
        if len(contours):
            # 取最大的轮廓，因为可能有其他颜色在相机前面，颜色在我们预定义颜色范围内
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # 在轮廓周围画圆
            cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
            # 如果处于绘制状态，处理绘制逻辑
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], WHITE_RGB, 5)
                    cv2.line(frame, points[i - 1], points[i], color_pointer, 2)

        # 如果已经显示了绘制结果
        if is_shown:
            cv2.putText(frame, 'you are drawing: ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)
            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60, 60))

        # 显示相机图像
        cv2.imshow("Camera", frame)
        # 如果选择显示画布，显示黑白画布
        if opt.canvas:
            cv2.imshow("Canvas", 255 - canvas)

    # 释放摄像头资源，关闭窗口
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
    # config = Config()
    # 关系图中包括(include)哪些函数名。
    # 如果是某一类的函数，例如类gobang，则可以直接写'gobang.*'，表示以gobang.开头的所有函数。（利用正则表达式）。
    # config.trace_filter = GlobbingFilter(include=[
    #     'main',
    #     'get_args',
    #     'cv2.*',
    #     'touch.*'
    # ])
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'test.png'
    # with PyCallGraph(output=graphviz, config=config):
    #     get_args()
    #     main(opt)

