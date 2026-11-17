import numpy as np
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image

# 加载模型
model = YOLO("kaggle/working/runs/detect/train/weights/best.pt")

# 初始化会话状态变量
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = -1

def detect_objects(image, conf):
    """
    对图片进行目标检测，并返回标注后的图片

    参数：
    image: 待检测的图片
    conf: 置信度阈值

    返回：
    annotated_frame: 标注后的图像帧
    """
    results = model.predict(source = image, conf = conf, save = False)
    annotated_frame = results[0].plot()
    return annotated_frame

def upload_image():
    """
    上传图片，并显示在界面上

    返回：
    np.array(image): 转换为 numpy 数组的图片，如果未上传则返回 None
    """
    uploaded_file = st.sidebar.file_uploader("上传一张图片", type = ["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption = " 上传的图片 ", use_container_width = True)
        return np.array(image)
    return None

def display_current_detection():
    """
    显示当前的目标检测结果图片
    """
    if st.session_state.current_index >= 0 and st.session_state.current_index < len(st.session_state.history):
        st.image(
            st.session_state.history[st.session_state.current_index], 
            caption = f" 检测结果 {st.session_state.current_index + 1} \/ {len(st.session_state.history)}", 
            use_container_width = True
        )

def infer_uploaded_image(conf):
    """
    对上传的图片进行目标检测

    参数：
    conf: 置信度阈值
    """
    col1, col2, col3 = st.columns([3, 1, 3])
    with col1:
        image = upload_image()
    with col2:
        if st.button(" 检测 "):
            if image is not None:
                detected_image = detect_objects(image, conf / 100)
                st.session_state.history.append(detected_image)
                st.session_state.current_index = len(st.session_state.history) - 1
        if st.button(" 上一张 ") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
        if st.button(" 下一张 ") and st.session_state.current_index < len(st.session_state.history) - 1:
            st.session_state.current_index += 1
    with col3:
        display_current_detection()