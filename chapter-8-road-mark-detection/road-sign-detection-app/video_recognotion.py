import numpy as np
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile

# 加载模型
model = YOLO('kaggle/working/runs/detect/train/weights/best.pt')

def _display_detected_frames(conf, st_frame, image):
    """
    显示检测到的帧

    参数：
    conf (float): 置信度阈值
    st_frame (Streamlit object): 用于显示视频帧的 Streamlit 容器
    image (numpy array): 输入的视频帧图片

    返回：
    None
    """
    # 调整图片大小以适应特定的宽高比
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 使用模型进行预测
    res = model.predict(image, conf = conf)

    # 在图片上绘制预测结果
    res_plotted = res[0].plot()

    # 在 Streamlit 应用中显示处理后的图片
    st_frame.image(
        res_plotted,
        caption = "Detected Video",
        channels = "BGR",
        use_container_width = True
    )

def infer_uploaded_video(conf):
    """
    对上传的视频进行推理

    参数：
    conf (float): 置信度阈值

    返回：
    None
    """
    # 从侧边栏上传视频文件
    source_video = st.sidebar.file_uploader(
        label = "Choose a video ..."
    )

    # 如果上传了视频，则显示视频
    if source_video:
        st.video(source_video)

    # 如果上传了视频且单击了 “执行” 按钮，则进行推理
    if source_video:
        if st.button(" 检测 "):
            with st.spinner("Running..."):
                try:
                    # 创建一个临时文件夹存储上传的视频
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    # 使用 OpenCV 打开视频文件
                    vid_cap = cv2.VideoCapture(tfile.name)
                    # 创建一个空的 Streamlit 容器来显示视频帧
                    st_frame = st.empty()
                    # 逐帧读取视频
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            # 调用函数显示检测的帧
                            _display_detected_frames(conf / 100, st_frame, image)
                        else:
                            # 释放视频捕获对象并退出循环
                            vid_cap.release()
                            break
                except Exception as e:
                    # 如果发生错误，显示错误信息
                    st.error(f"Error loading video: {e}")

