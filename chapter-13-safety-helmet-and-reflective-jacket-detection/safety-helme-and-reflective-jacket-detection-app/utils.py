import streamlit as st
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import pandas as pd
import base64
import tempfile
from deepseek_llm import DeepSeekLLM
import os
from pandasai import SmartDataframe

# 加载预训练模型
model = YOLO('kaggle/working/runs/detect/train/weights/best.pt')

# 定义执行目标检测的函数
def detect_objects(image, conf_threshold, box_shape, selected_class_name, line_width, label_font_size):
    """
    检测图片中的目标对象，并绘制边界框和标签

    Args:
        image (np.array): 待检测的图片，应为 numpy 数据格式
        conf_threshold (float): 置信度阈值，用于筛选预测结果
        box_shape (str): 边界框的形状，可选值包括 “矩形框”， “圆角框”， “四角框”
        selected_class_name (str): 选择的类别名称，可选值包括 “安全头盔” 和 “反光背心”
        line_width (int): 边界框的线宽
        label_font_size (int): 标签的字体大小

    returns:
        np.ndarray: 绘制了边界框和标签的图片

    """
    # 使用模型进行预测
    results = model.predict(
        source = image, conf = conf_threshold
    )

    # 从结果中提取检测信息
    # 根据类别名称筛选检测结果
    if selected_class_name == " 所有类别 ":
        detections = sv.Detections.from_ultralytics(results[0])
    elif selected_class_name == " 安全头盔 ":
        detections = sv.Detections.from_ultralytics(results[0])
        detections = detections[detections.data['class_name'] == 'Safety-Helmet']
    else:
        detections = sv.Detections.from_ultralytics(results[0])
        detections = detections[detections.data['class_name'] == 'Reflective-Jacket']

    # 创建图片副本以便绘制边界框
    frame = image.copy()

    # 根据检测结果绘制边界框并标注类别和置信度
    # 添加边框
    # 根据用户选择的边界框形式创建对应的边界框注释器
    if box_shape == " 矩形框 ":
        # 创建矩形框注释器
        box_annotator = sv.BoxAnnotator(thickness = line_width)
    elif box_shape == " 圆形框 ":
        # 创建圆角框注释器
        box_annotator = sv.RoundBoxAnnotator(thickness = line_width)
    else: # 四角框
        # 创建四角框注释器
        box_annotator = sv.BoxCornerAnnotator(thickness = line_width)
    # 使用注释器在图片上绘制边界框
    box_annotator.annotate(
        scene = frame,
        detections = detections
    )

    # 实例化 LabelAnnotator 类
    # 创建标签注释器
    label_annotator = sv.LabelAnnotator(text_scale = label_font_size)
    # 获得各边框的标签及置信度
    labels = [
        # 生成标签文本，包括类别名称和置信度
        f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]

    # 使用注释器在图片上添加标签
    label_annotator.annotate(
        scene = frame,
        detections = detections,
        labels = labels
    )

    # 返回绘制了边界框和标签的图片
    return frame

def generate_detections_df(filename, model, image_np):
    # 初始化空的 DataFrame，列名为 ["Image Name", "Class Name", "Confidence", "Bounding Box"]
    detections_df = pd.DataFrame(columns = ["Image Name", "Class Name", "Confidence", "Bounding Box"])

    rows = []
    # 获取当前图片的检测结果
    results = model.predict(source = image_np)
    # 遍历每个边界框
    for result in results:
        for box in result.boxes.cpu().numpy():
            # 获取边界框的坐标，并转换为整数类型
            r = box.xyxy[0].astype(int)
            # 获取类别 ID
            class_id = int(box.cls[0])
            # 获取置信度，并保留两位小数
            conf = round(float(box.conf[0]), 2)
            # 根据类别 ID 获取类别名称
            class_name = model.names[class_id]
            # 将结果添加到 rows 列表中
            rows.append({
                "Image Name": filename,
                "Class Name": class_name,
                "Confidence": conf,
                "Bounding Box": str(r)
            })
    # 将 rows 列表转换为 DataFrame
    new_df = pd.DataFrame(rows)
    # 检查并过滤全为 NA 的列
    new_df = new_df.dropna(axis = 1, how = 'all')
    # 并与 detections 合并
    detections_df = pd.concat([detections_df, new_df], ignore_index = True)
    # 去除重复行
    detections_df = detections_df.drop_duplicates(subset = ["Image Name", "Class Name", "Confidence", "Bounding Box"])
    return detections_df

def process_natural_language_query(history_df):
    # 如果会话状态中不存在 "messages" 键，则进行初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 创建一个按钮，用于重置会话状态
    reset_button = st.sidebar.button(" 重置会话 ")
    if reset_button:
        st.session_state.messages.clear()

    # 遍历会话状态中的消息
    for message in st.session_state.messages:
        # 根据消息中的角色，显示相应的消息内容
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 通过文本输入框获取用户的自然语言查询
    nl_query = st.sidebar.text_input(" 输入自然语言查询 ", " 检测到的类别名称有哪些？ ")

    try:
        deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
    except KeyError:
        print("环境变量不存在")
    # 创建一个 OpenAI API 客户端实例，指定令牌
    llm = DeepSeekLLM(
        api_key = deepseek_api_key
    )

    # 创建一个 SmartDataframe 实例，传入历史数据框和配置信息
    smart_df = SmartDataframe(history_df, config = {"llm": llm})

    try:
        # 调用 SmartDataframe 实例的 chat 方法执行自然语言查询
        response = smart_df.chat(nl_query + " 用中文回答，不要画图 ")

        # 将用户输入的消息添加到会话状态中
        st.session_state.messages.append({
            "role": "user",
            "content": nl_query
        })
        # 显示用户输入的消息
        with st.chat_message("user"):
            st.write(nl_query)
        # 将助手返回的消息添加到会话状态中
        st.session_state.messages.append({
            "role": "assistent",
            "content": response
        })
        # 显示助手返回的消息
        with st.chat_message("assistent"):
            st.markdown(response)
    except Exception as e:
        # 如果执行查询时发生异常，则显示错误信息
        st.error(f"查询失败：{e}")

def infer_uploaded_video(conf_threshold, box_shape, selected_class_name, line_width, label_font_size):
    """
    根据给定的置信度阈值，框形状，所选类别名称，线条宽度和标签字体大小，对上传的视频进行对象检测

    Args:
        conf_threshold (float): 检测置信度阈值，用于过滤低置信度的检测结果
        box_shape (tuple): 包围框的形状，格式为 (width, height)
        selected_class_name (str): 所选类别的名称，用于过滤检测结果中的类别
        line_width (int): 绘制包围框时的线条宽度
        label_font_size (int): 绘制标签时的字体大小

    Return:
        None
    """
    # 从侧边栏上传视频文件
    source_video = st.sidebar.file_uploader(
        label = " 上传视频 "
    )
    # 如果上传了视频，则显示视频
    if source_video:
        st.video(source_video)
    # 如果上传了视频且单击了“检测”按钮，则进行推理
    if source_video:
        if st.button(" 检测 "):
            with st.spinner("Running..."):
                try:
                    # 创建一个临时文件夹来存储上传的视频
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    # 使用 OpenCV 打开临时视频文件
                    vid_cap = cv2.VideoCapture(
                        tfile.name
                    )
                    # 创建一个空的 streamlit 容器来显示视频帧
                    st_frame = st.empty()
                    # 逐帧读取视频
                    while (vid_cap.isOpened()):
                        sources, image = vid_cap.read()
                        if sources:
                            # 调用函数显示检测到的帧
                            detected_image = detect_objects(image, conf_threshold, box_shape, selected_class_name, line_width, label_font_size)
                            # 在 Streamlit 应用中显示处理后的图片
                            st_frame.image(detected_image, caption = "Detected Video", channels = "BGR", use_container_width = True)
                        else:
                            # 释放视频捕获对象并退出循环
                            vid_cap.release()
                            break
                except Exception as e:
                    # 如果发生错误，显示错误信息
                    st.error(f"Error loading video: {e}")

    
