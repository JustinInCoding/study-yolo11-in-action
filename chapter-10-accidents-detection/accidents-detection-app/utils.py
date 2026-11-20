from ultralytics import YOLO
import supervision as sv
import base64
import streamlit as st

# 加载预训练的 YOLO 模型
model = YOLO('../kaggle/working/runs/detect/train/weights/best.pt')

# 定义执行目标检测的函数
def detect_objects(image, conf_threshold, iou_threshold, box_annotator):
    # 使用模型进行预测
    results = model.predict(
        source = image,
        conf = conf_threshold,
        iou = iou_threshold
    )
    # 从结果中提取检测信息
    detections = sv.Detections.from_ultralytics(results[0])

    # 创建图像副本以便绘制边界框
    frame = image.copy()
    # 根据检测结果绘制边界框并标注类别和置信度
    # 添加边界框
    box_annotator.annotate(
        scene = frame,
        detections = detections
    )
    # 实例化 LabelAnnotator 类
    label_annotator = sv.LabelAnnotator()
    labels = [
        f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]
    # 添加标签到框中
    label_annotator.annotate(
        scene = frame,
        detections = detections,
        labels = labels
    )
    return frame

def set_background(image_path, opacity = 0.8):
    """
    设置应用的背景图片

    该函数读取给定路径的图片，将其编码为 Base64 数据，然后使用 html 和 css 将图片作为背景应用到应用中
    它允许通过 opacity 参数调整背景图片的透明度
    
    参数：
    - image_path (str): 背景图片的文件路径
    - opacity (float, 可选): 背景图片的透明度，取值范围为 0 ~ 1，其中 0 完全透明，1 完全不透明。默认值为 0.8 
    """
    # 读取图片文件并将其转换为二进制数据
    bin_image_data = open(image_path, 'rb').read()

    # 将二进制数据编码为 Base64，以便在 HTML 中使用
    b64_encoded_data = base64.b64encode(bin_image_data)

    # 构建包含 Base64 编码图片数据的字符串，用作背景图片的 URL
    bg_img_str = f"data:image/png;base64,{b64_encoded_data.decode()}"

    # 使用 CSS 样式设置页面背景，包括背景图片、大小、重复、固定和混合模式，以及透明度
    page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("{bg_img_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-blend-mode: multiply;
            opacity: {opacity}
        }}
        #MainMenu {{visibility: hidden}}
        footer {{visibility: hidden}}
        header {{visibility: hidden}}
        </style>
    """

    # 使用 streamlit 的 markdown 函数输出 HTML 代码，允许 HTML 代码中的不安全内容
    st.markdown(page_bg_img, unsafe_allow_html = True)

# 定义设置居中标题的函数
def set_centered_title(title):
    st.markdown(f'<div style = "text-align: center;" > <h1 style = "color: white; font-weight: bold;" >{title}</h1> </div>', unsafe_allow_html = True)