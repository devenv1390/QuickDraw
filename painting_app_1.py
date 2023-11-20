import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch


def main():
    # 加载模型
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()

    # 创建画布
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.namedWindow("Canvas")

    global ix, iy, is_drawing
    is_drawing = False

    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                # 在画布上绘制白色线条
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            # 在画布上绘制白色线条
            cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y

    # 设置鼠标回调函数
    cv2.setMouseCallback('Canvas', paint_draw)

    while True:
        # 显示画布
        cv2.imshow('Canvas', 255 - image)

        # 等待按键
        key = cv2.waitKey(10)

        # 按空格键进行预测
        if key == ord(" "):
            # 转换为灰度图
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 获取非零元素的坐标
            ys, xs = np.nonzero(image)
            min_y = np.min(ys)
            max_y = np.max(ys)
            min_x = np.min(xs)
            max_x = np.max(xs)

            # 裁剪图像
            image = image[min_y:max_y, min_x:max_x]
            image = cv2.resize(image, (28, 28))

            # 转换为模型输入格式
            image = np.array(image, dtype=np.float32)[None, None, :, :]
            image = torch.from_numpy(image)

            # 模型推理
            logits = model(image)
            print(CLASSES[torch.argmax(logits[0])])

            # 重置画布和坐标
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            ix = -1
            iy = -1

        # 按ESC键退出循环
        if key == 27:
            break

    # 关闭窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
