from ultralytics import YOLO
import supervision as sv
import streamlit as st
import base64
import tempfile
import cv2
import numpy as np
import pandas as pd

# 加载预训练模型 
model = YOLO('kaggle/working/runs/detect/train/weights/best.pt')

# 定义执行目标检测的函数
def detect_objects(image, conf_threshold, box_shape, line_width, label_font_size):
    # 使用模型进行预测
    results = model.predict(source = image, conf = conf_threshold)

    # 从结果中提取检测信息
    detections = sv.Detections.from_ultralytics(results[0])

    # 创建图片副本以便绘制边界框
    frame = image.copy()

    # 根据检测结果绘制边界框并标注类别和置信度
    if box_shape == ' 矩形框 ':
        box_annotator = sv.BoxAnnotator(thickness = line_width)
    elif box_shape == ' 圆角框 ':
        box_annotator = sv.RoundBoxAnnotator(thickness = line_width)
    else:
        box_annotator = sv.BoxCornerAnnotator(thickness = line_width)
    # 使用注释器在图片上绘制边界框
    box_annotator.annotate(
        scene = frame,
        detections = detections
    )
    # 实例化 LabelAnnotator 类
    label_annotator = sv.LabelAnnotator(text_scale = label_font_size)
    # 获得各边界框的标签置信度
    labels = [
        f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]
    label_annotator.annotate(
        scene = frame,
        detections = detections,
        labels = labels
    )

    return results, frame

def infer_uploaded_video(conf_threshold, box_shape, line_width, label_font_size):
    # 从侧边栏上传视频
    source_video = st.sidebar.file_uploader(label = ' 上传视频 ')

    # 如果上传了视频，则显示视频
    if source_video:
        st.video(source_video)

    # 如果上传了视频并且点击了“检测”按钮，则进行推理
    if source_video:
        if st.button(' 检测 '):
            with st.spinner("Running..."):
                try:
                    # 创建一个临时文件夹来存储上传的视频
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    # 使用 OpenCV 打开临时视频文件
                    vid_cap = cv2.VideoCapture(tfile.name)
                    # 创建一个空的 Streamlit 容器来显示视频帧
                    st_frame = st.empty()
                    # 逐帧读取视频
                    while vid_cap.isOpened():
                        sources, image = vid_cap.read()
                        if sources:
                            # 调用函数显示检测到的帧
                            _, detected_image = detect_objects(image, conf_threshold, box_shape, line_width, label_font_size)
                            # 在 Streamlit 应用中显示处理后的图片
                            st_frame.image(detected_image, caption = 'Detected Video', channels = 'BGR', use_container_width = True)
                        else:
                            # 释放视频捕获对象并推出循环
                            vid_cap.release()
                            break
                except Exception as e:
                    # 如果发生错误，显示错误信息
                    st.error(f"Error loading video: {e}")
