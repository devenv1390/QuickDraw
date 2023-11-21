# 代码解读(Chat-GPT)

## camera_app.py

### get_args()函数:

```python
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
```

这段代码定义了一个函数 `get_args()`，用于获取命令行参数。它使用了 Python 的 `argparse` 模块，该模块用于解析命令行参数，并生成帮助信息。

具体来说：
- `argparse.ArgumentParser` 创建了一个解析器对象，该对象用于指定命令行参数的规则和帮助信息。
- `add_argument` 方法用于添加命令行参数，并指定其名称、类型、默认值、帮助信息等。
  - `-c` 或 `--color` 参数是一个字符串，表示摄像头捕捉到的颜色，作为画笔的颜色。可选值为 "green"、"blue"、"red"，默认为 "green"。
  - `-a` 或 `--area` 参数是一个整数，表示被捕捉对象的最小面积，默认为 3000。
  - `-d` 或 `--display` 参数是一个整数，表示预测结果显示的时间，以秒为单位，默认为 3。
  - `-s` 或 `--canvas` 参数是一个布尔值，表示是否显示黑白画布，默认为 False。
- `parser.parse_args()` 解析命令行参数，并返回一个包含了解析结果的命名空间对象。

最后，函数返回解析得到的参数对象。这种做法使得在程序中可以方便地使用这些参数，以便在运行时定制程序的行为。例如，通过命令行可以指定摄像头的颜色、捕捉对象的最小面积等参数。

---

### main()函数：

```python
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
```

这段代码是根据命令行参数中指定的颜色选项，设置相应的颜色阈值和颜色指针。

- 如果 `opt.color` 为 "red"，则使用预定义的红色阈值和红色指针。
- 如果 `opt.color` 为 "green"，则使用预定义的绿色阈值和绿色指针。
- 如果 `opt.color` 既不是 "red" 也不是 "green"，则使用预定义的蓝色阈值和蓝色指针。

这里的阈值用于在图像中提取指定颜色的对象，一般以HSV（色调、饱和度、明度）颜色空间的形式表示。`color_lower` 和 `color_upper` 分别表示颜色的下限和上限，用于创建一个颜色的范围。

`color_pointer` 则表示指定颜色在RGB颜色空间的值。

这种设置方式允许根据用户的输入选择不同的颜色，以适应不同的应用场景或用户喜好。

```python
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
```

这段代码主要完成以下几个任务：

1. `points`: 创建了一个双向队列，用于存储绘制的点的坐标，最多保留512个点。这可能用于记录用户绘制的轨迹。

2. `canvas`: 创建了一个空白画布，尺寸为 (480, 640, 3)，即高480像素、宽640像素，3通道的彩色图像。这个画布可能用于在上面进行绘制。

3. `camera`: 初始化了一个摄像头对象，通过 OpenCV 的 `cv2.VideoCapture(0)` 方法表示使用默认的摄像头。这个对象可能用于捕捉摄像头的图像。

4. `is_drawing` 和 `is_shown`: 分别是标志变量，用于表示是否正在绘制和是否已经显示了预测结果。

5. `class_images` 和 `predicted_class`: 分别是通过调用 `get_images` 函数获取的类别图像和用于存储预测结果的变量。

总的来说，这段代码似乎是为了初始化一些必要的变量，准备开始使用摄像头进行交互式绘图，并与预测结果相关联。在这之前可能有一些其他的初始化和函数定义，以及导入其他的库和模块。

```python
    # 加载模型
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()
```

这段代码使用了 `PyTorch` 库，主要用于加载训练好的模型并设置其为评估模式。

1. `torch.cuda.is_available()`: 这个条件语句检查当前系统是否支持 CUDA（GPU 加速）。如果支持 CUDA，代码进入 `if` 分支，否则进入 `else` 分支。

2. 在 `if` 分支中，通过 `torch.load("trained_models/whole_model_quickdraw")` 加载了一个已经训练好的模型。这里假设模型是以文件 "trained_models/whole_model_quickdraw" 存储的。这个文件可能包含了模型的权重参数等信息。

3. 如果支持 CUDA，模型会被加载到 GPU 上。如果不支持 CUDA，`map_location=lambda storage, loc: storage` 的设置将模型加载到 CPU 上。

4. `model.eval()`: 设置模型为评估模式。在 `PyTorch` 中，调用 `eval()` 方法会将模型切换为评估模式，这通常会影响一些具体层（如 Dropout 层）的行为。在进行推理（而非训练）时，设置模型为评估模式是推荐的做法。

总的来说，这段代码的目的是加载预训练好的 `PyTorch` 模型，并将其设置为评估模式，以便用于后续的推理任务。请注意，由于 `torch.load` 返回的是模型的状态字典（state dictionary），加载模型后可能需要进一步的步骤，如将模型参数加载到相应的模型架构中。

---

#### while True:

```python
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
```

这段代码监听键盘输入，并根据按键的不同执行不同的操作：

1. `key = cv2.waitKey(10)`: 这一行等待10毫秒，检测键盘输入。返回的 `key` 存储了按下的键的 ASCII 值。

2. `if key == ord("q"):`: 如果按下的键是 "q" 键，就会跳出循环，从而退出程序。

3. `elif key == ord(" "):`: 如果按下的键是空格键，就会切换绘制状态。如果当前正在绘制，那么停止绘制；如果当前没有绘制，那么开始绘制。同时，如果开始绘制时画布已经显示了预测结果（`is_shown` 为真），则清空点的队列和画布，重新开始绘制。

总体来说，这部分代码实现了一个简单的交互式控制，允许用户通过键盘输入来控制程序的行为，例如开始或停止绘制，以及通过按下 "q" 键来退出程序。

```python
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
```

这部分代码在两个条件满足的情况下执行：

1. `not is_drawing and not is_shown`: 如果当前既不在绘制状态（`is_drawing` 为假），同时预测结果也没有显示（`is_shown` 为假）。

2. 在这种情况下，代码进行了一系列图像处理步骤，处理绘制的图像以进行预测：
   - 将彩色画布转换为灰度图像：`cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)`
   - 中值滤波：`cv2.medianBlur(canvas_gs, 9)`
   - 高斯模糊：`cv2.GaussianBlur(median, (5, 5), 0)`
   - 二值化：`cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`

3. 通过轮廓检测找到画布上的对象：
   - `cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)`

4. 如果找到了轮廓：
   - 找到面积最大的轮廓：`contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]`
   - 如果该轮廓的面积大于用户指定的最小面积 (`opt.area`)，则进行后续处理：
     - 获取边界框坐标：`x, y, w, h = cv2.boundingRect(contour)`
     - 从灰度图像中提取感兴趣的区域（ROI）：`image = canvas_gs[y:y + h, x:x + w]`
     - 调整 ROI 的大小为 (28, 28)：`image = cv2.resize(image, (28, 28))`
     - 将处理后的图像转换为 `PyTorch Tensor`，并进行模型推理：`logits = model(image)`
     - 获取预测类别：`predicted_class = torch.argmax(logits[0])`
     - 设置 `is_shown` 为 True，表示预测结果已经显示。

5. 如果找到的轮廓面积太小，输出提示信息并重新开始绘制：
   - `print("绘制的物体太小，请绘制一个较大的物体！")`
   - 清空点的队列和画布：`points = deque(maxlen=512)` 和 `canvas = np.zeros((480, 640, 3), dtype=np.uint8)`

总的来说，这段代码用于在绘制完成后，对绘制的图像进行处理并使用预训练的深度学习模型进行分类，然后显示预测结果。如果预测结果显示了一个足够大的物体，程序将设置 `is_shown` 为 True，表示预测结果已经显示。

---

```python
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
```

这段代码用于从摄像头读取帧，进行一系列图像处理步骤，最终获取感兴趣的物体的轮廓。

1. `ret, frame = camera.read()`: 从摄像头读取一帧图像，`ret` 表示是否成功读取帧，而 `frame` 存储了读取到的图像。

2. `frame = cv2.flip(frame, 1)`: 对图像进行水平翻转，可能是为了调整图像方向。

3. `hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`: 将图像从BGR颜色空间转换为HSV颜色空间，这样可以更容易地对颜色进行处理。

4. `kernel = np.ones((5, 5), np.uint8)`: 创建一个5x5的矩形结构元素，用于后续的图像腐蚀、膨胀等形态学操作。

5. `mask = cv2.inRange(hsv, color_lower, color_upper)`: 根据之前设定的颜色阈值 (`color_lower` 和 `color_upper`)，在HSV图像中创建一个掩码，标识出指定颜色的区域。

6. `mask = cv2.erode(mask, kernel, iterations=2)`: 对掩码进行腐蚀操作，用于去除噪声。

7. `mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)`: 使用开运算（morphological opening）操作，进一步去除噪声，通常用于平滑物体的边缘。

8. `mask = cv2.dilate(mask, kernel, iterations=1)`: 对掩码进行膨胀操作，弥合物体之间的空隙。

9. `contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`: 查找掩码图像中的轮廓，`contours` 存储了找到的轮廓，`_` 存储了轮廓的层次结构信息。这些轮廓通常代表了图像中的物体。

这部分代码的目的是通过对摄像头捕获的图像进行预处理，提取出指定颜色的物体的轮廓，以便后续处理。

```python
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
```

这部分代码在找到了轮廓的情况下进行一些绘制和处理逻辑：

1. `contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]`: 从所有找到的轮廓中选择面积最大的一个，因为可能有其他颜色的轮廓也在图像中，而我们只关心颜色范围内的物体。

2. `((x, y), radius) = cv2.minEnclosingCircle(contour)`: 使用最小外接圆方法得到包围轮廓的圆心坐标 `(x, y)` 和半径 `radius`。

3. `cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)`: 在原始图像上画一个圆，用黄色标记轮廓。

4. 如果处于绘制状态 (`is_drawing`)，则进行以下处理逻辑：
   - `M = cv2.moments(contour)`: 计算轮廓的矩（moments），获取轮廓的中心坐标。
   - `center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))`: 计算轮廓的中心坐标。
   - `points.appendleft(center)`: 将中心坐标加入到存储点的双向队列中。
   - 遍历队列中的点，使用 `cv2.line` 在画布上绘制连接线和在原始图像上显示。

这段代码的目的是在找到轮廓后，对轮廓进行一些可视化操作，同时在绘制状态下，将轮廓的中心坐标加入到队列中，并在画布和原始图像上绘制连接线。这可能用于实时显示用户绘制的图案。

```python
        # 如果已经显示了绘制结果
        if is_shown:
            cv2.putText(frame, 'you are drawing: ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)
            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60, 60))

        # 显示相机图像
        cv2.imshow("Camera", frame)
        # 如果选择显示画布，显示黑白画布
        if opt.canvas:
            cv2.imshow("Canvas", 255 - canvas)
```

这部分代码主要用于在图像上添加文字信息，显示用户当前是否处于绘制状态，以及显示预测的物体类别的相关信息。最后，通过 `cv2.imshow` 在窗口中展示原始图像和可能的画布。

1. 如果 `is_shown` 为真，表示预测结果已经显示，将在图像上添加一些文字信息：
   - `cv2.putText(frame, 'you are drawing: ', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)`: 在图像上添加文本，显示用户当前正在绘制的信息。
   - `frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60, 60))`: 在图像上添加一个类别的图像，可能是预测的物体类别的图标。`get_overlay` 函数的具体实现没有在提供的代码片段中，但很可能是用于将一个图像叠加在另一个图像上的函数。

2. `cv2.imshow("Camera", frame)`: 在名为 "Camera" 的窗口中展示原始图像 `frame`。

3. 如果 `opt.canvas` 为真，表示需要显示黑白画布，则执行：
   - `cv2.imshow("Canvas", 255 - canvas)`: 在名为 "Canvas" 的窗口中展示反转后的黑白画布，`255 - canvas` 用于反转画布颜色。

这段代码的目的是实时显示摄像头捕捉到的图像，并在图像上添加一些信息，包括用户是否正在绘制以及预测的物体类别等。如果选择显示画布 (`opt.canvas` 为真)，则还会在窗口中显示黑白画布。

---

#### get_overlay():

```python
def get_overlay(bg_image, fg_image, sizes=(40, 40)):
    fg_image = cv2.resize(fg_image, sizes)
    fg_mask = fg_image[:, :, 3:]
    fg_image = fg_image[:, :, :3]
    bg_mask = 255 - fg_mask
    bg_image = bg_image/255
    fg_image = fg_image/255
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)/255
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)/255
    image = cv2.addWeighted(bg_image*bg_mask, 255, fg_image*fg_mask, 255, 0.).astype(np.uint8)
    return image
```

这是一个函数 `get_overlay` 的实现，它的目的是将一个图像叠加在另一个图像上，并返回合成后的图像。

函数参数：
- `bg_image`: 背景图像，即另一个图像将要叠加在其上的图像。
- `fg_image`: 前景图像，即要叠加到背景上的图像。
- `sizes`: 叠加后的图像的大小，缺省为 (40, 40)。

函数实现步骤：
1. `fg_image = cv2.resize(fg_image, sizes)`: 调整前景图像的大小，使其与指定的尺寸匹配。
2. `fg_mask = fg_image[:, :, 3:]`: 获取前景图像的 alpha 通道，该通道表示图像中每个像素的透明度。
3. `fg_image = fg_image[:, :, :3]`: 前景图像中去除 alpha 通道，保留 RGB 通道。
4. `bg_mask = 255 - fg_mask`: 计算背景的遮罩，即前景遮罩的补集。
5. `bg_image = bg_image/255`: 将背景图像进行归一化，将像素值映射到 [0, 1] 的范围。
6. `fg_image = fg_image/255`: 将前景图像进行归一化。
7. `fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)/255`: 将前景的 alpha 通道转换为彩色图像，同样进行归一化。
8. `bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)/255`: 将背景的遮罩转换为彩色图像，同样进行归一化。
9. `image = cv2.addWeighted(bg_image*bg_mask, 255, fg_image*fg_mask, 255, 0.).astype(np.uint8)`: 使用加权叠加将前景和背景叠加在一起，其中加权的权重是 alpha 通道。最后，将图像转换为整数类型。

函数返回合成后的图像。这样的操作通常用于在图像上添加叠加的图标、标签等。

---

#### 装饰器`@typing.overload`

举例说明：

```python
@typing.overload
def findContours(image: UMat, mode: int, method: int, contours: typing.Sequence[UMat] | None = ..., hierarchy: UMat | None = ..., offset: cv2.typing.Point = ...) -> tuple[typing.Sequence[UMat], UMat]: ...
```

这个函数声明使用了 Python 的 `typing` 模块中的 `@typing.overload` 装饰器，表示下面的代码块包含了对同一个函数的多个重载（overload）版本。

这个特定的函数声明是 OpenCV（cv2 模块）中的 `findContours` 函数的一个重载版本，用于在图像中查找轮廓。以下是该函数的参数和返回值的说明：

- `image: UMat`: 表示输入的图像，使用 OpenCV 的 `UMat` 类型，可能是一个二维图像（单通道或多通道）。
- `mode: int`: 表示轮廓检索模式，指定了轮廓的层次结构。这是一个整数值。
- `method: int`: 表示轮廓的逼近方法，指定了轮廓的拟合精度。这是一个整数值。
- `contours: typing.Sequence[UMat] | None = ...`: 一个序列，用于存储找到的轮廓。这可以是一个 `UMat` 对象的列表。该参数是可选的，可以为 `None`。
- `hierarchy: UMat | None = ...`: 一个 `UMat` 对象，用于存储轮廓的层次结构信息。该参数是可选的，可以为 `None`。
- `offset: cv2.typing.Point = ...`: 表示轮廓坐标的偏移量，即轮廓的坐标将被此偏移量移动。这是一个 `cv2.typing.Point` 类型的参数。该参数是可选的，可以为 `...`。

返回值：
- 一个元组，包含两个元素：
  1. `typing.Sequence[UMat]`：找到的轮廓，可能是多个轮廓组成的列表。
  2. `UMat`：轮廓的层次结构信息。

这样的函数声明使用 `@typing.overload` 可以提供对函数多个版本的说明，以便在类型提示和文档生成等场景中更好地展示函数的使用方式。在实际调用时，解释器会根据传递的参数类型选择正确的函数版本。

---

### 信息架构

进行推理的信息架构通常涉及多个组件和数据流。在这个特定的代码中，以下是进行推理时可能涉及的主要信息架构：

1. **摄像头输入：**
   - 通过OpenCV库获取摄像头的实时视频流。

2. **颜色范围处理：**
   - 将摄像头捕捉到的图像转换为HSV颜色空间。
   - 使用预定义的颜色范围提取感兴趣的颜色（例如红色、绿色、蓝色）。

3. **图像处理和轮廓提取：**
   - 对提取到的颜色进行形态学处理（腐蚀、膨胀等）。
   - 使用OpenCV的`findContours`函数找到图像中的轮廓。

4. **绘制逻辑：**
   - 在绘制状态下，根据检测到的轮廓更新绘制的点。
   - 在画布上绘制相应的图案。

5. **模型推理：**
   - 加载预训练的`PyTorch`模型，该模型用于对绘制的物体进行分类。
   - 对从画布截取的图像进行预处理，如灰度转换、滤波等。
   - 将预处理后的图像传递给模型进行推理，获取分类结果。

6. **结果显示：**
   - 根据模型的分类结果，在摄像头图像上显示相应的信息，如分类标签、绘制状态等。
   - 如果开启了画布显示选项，显示绘制的图案。

7. **用户交互：**
   - 通过按键检测，实现用户对程序的控制，如切换绘制状态、退出程序等。

8. **实时图像显示：**
   - 使用OpenCV在窗口中实时显示摄像头捕捉到的图像。
   - 如果选择显示画布，同时在窗口中显示黑白画布。

总体而言，这个信息架构涵盖了从摄像头输入到模型推理再到结果显示的整个流程，同时考虑了用户的交互和控制。

---

## mediapipe_app.py

