import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import os
from deepseek_llm import DeepSeekLLM
from pandasai import SmartDataframe

# 创建 tmp 和 output 文件夹
def create_directories():
    os.makedirs("./tmp", exist_ok = True)
    os.makedirs("./output", exist_ok = True)

def save_uploaded_files(uploaded_files):
    image_paths = []
    for file in uploaded_files:
        file_path = os.path.join("./tmp", file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
        image_paths.append(file_path)
    return image_paths

def detect_images(model, image_paths, session_state):
    """
    使用指定的模型检测一组图片中的目标

    参数：
    - model: 目标检测模型，用于检测图片中的目标
    - image_path: 图片路径列表，表示需要检测的图片文件的位置
    - session_state: 会话状态对象，用于存储检测结果，避免重复检测相同的图片

    此函数不返回任何值，但会将检测到的图片及其结果保存在 session_state 中，并将标注后的图片保存到磁盘
    """
    # 检查 session_state 是否包含 detections 属性，如果不包含则将其初始化为空字典
    if 'detections' not in session_state:
        session_state.detections = {}

    # 遍历每个图片路径
    for image_path in image_paths:
        # 如果当前图片未被检测过，则进行检测
        if image_path not in session_state.detections:
            # 使用模型预测图片中的目标，但不保存模型的输出
            results = model.predict(source = image_path, save = False)[0]

            # 生成带有检测结果标注的图像帧
            annotated_frame = results.plot()

            # 构造标注后图片的保存路径
            annotated_frame_path = os.path.join("./output", f"annotated_{os.path.basename(image_path)}")

            # 将标注后的图像帧保存到磁盘
            cv2.imwrite(annotated_frame_path, annotated_frame)

            # 将图像路径和其对应的标注信息添加到会话状态中
            session_state.detections[image_path] = {
                'annotated_frame_path': annotated_frame_path,
                'results': results
            }

# 生成检测结果表格
def generate_detections_df(session_state, model):
    """
    生成数据检测结果的 DataFrame

    参数：
        session_state (dict): 包含检测结果的会话状态
        model (Model): 目标检测模型，包含类别名称等信息
    
    返回：
        pd.DataFrame: 包含图片路径，类别名称，置信度和边界框的检测结果的 DataFrame
    """
    # 初始化空的 DataFrame，列名为 ["Image Path", "Class Name", "Confidence", "Bounding Box"]
    detections_df = pd.DataFrame(columns = ["Image Path", "Class Name", "Confidence", "Bounding Box"])
    rows = []
    for image_path, detection in session_state.detections.items():
        # 获取当前图片的检测结果
        results = detection['results']
        # 遍历每个边界框
        for box in results.boxes.cpu().numpy():
            # 获取边界框的坐标并将其转换为整数类型
            r = box.xyxy[0].astype(int)
            # 获取类别 ID
            class_id = int(box.cls[0])
            # 获取置信度，保留两位小数
            conf = round(float(box.conf[0]), 2)
            # 根据类别 ID 获取类别名称
            class_name = model.names[class_id]
            # 将结果添加到 rows 列表中
            rows.append({
                "Image Path": image_path,
                "Class Name": class_name,
                "Confidence": conf,
                "Bounding Box": str(r)
            })
    # 将 rows 列表转换为 DataFrame
    new_df = pd.DataFrame(rows)
    # 检查并过滤数据全为 NA 的列
    new_df = new_df.dropna(axis = 1, how = "all")

    # 与 detections_df 合并
    detections_df = pd.concat([detections_df, new_df], ignore_index = True)
    # 去除重复的行
    detections_df = detections_df.drop_duplicates(subset = ["Image Path", "Class Name", "Confidence", "Bounding Box"])
    return detections_df

def update_history_df(detections_df):
    """
    更新历史检测数据 DataFrame

    参数：
        detections_df (pd.DataFrame): 包含新检测数据的 DataFrame

    返回：
        pd.DataFrame: 更新后的历史检测数据 DataFrame
    """
    # 历史数据文件的路径
    history_df_path = "./output/datections_history.csv"

    # 如果历史数据文件存在
    if os.path.exists(history_df_path):
        # 读取历史数据文件
        history_df = pd.read_csv(history_df_path)
        # 删除历史数据中的重复行
        history_df = history_df.drop_duplicates(subset = ["Image Path", "Class Name", "Confidence", "Bounding Box"])
    else:
        # 如果历史文件不存在，则创建一个新的空 DataFrame
        history_df = pd.DataFrame(columns = ["Image Path", "Class Name", "Confidence", "Bounding Box"])

    # 将新检测数据和历史数据合并
    history_df = pd.concat([history_df, detections_df], ignore_index = True)
    # 删除合并后的数据中的重复行
    history_df = history_df.drop_duplicates(subset = ["Image Path", "Class Name", "Confidence", "Bounding Box"])

    # 将更新后的数据保存到文件中
    history_df.to_csv(history_df_path, index = False)

    # 返回更新后的历史数据
    return history_df

def process_natural_language_query(history_df):
    # 如果会话状态中不存在 "message" 键，则对其进行初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 遍历会话中的消息
    for message in st.session_state.messages:
        # 根据消息中的角色，显示相应的消息内容
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

    # 通过文本输入框获取用户的自然语言查询
    nl_query = st.sidebar.text_input(" 输入自然语言查询 ", " 检测到哪些目标 ")

    try:
        deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
    except KeyError:
        print("环境变量不存在")
    
    # 创建一个 OpenAI API 客户端实例，指定 API 基础 URL、API 令牌和模型
    llm = DeepSeekLLM(
        api_key = deepseek_api_key
    )

    # 创建一个 SmartDataFrame 实例，传入历史数据框和配置信息
    smart_df = SmartDataframe(history_df, config = {'llm': llm})

    try:
        # 调用 SmartDataframe 实例的 chat 方法执行自然语言查询
        response = smart_df.chat(nl_query + ' 用中文回答，不要画图 ')

        # 将用户输入的消息添加到会话状态中
        st.session_state.messages.append({
            'role': 'user',
            'content': nl_query
        })
        # 显示用户输入的消息
        with st.chat_message('user'):
            st.write(nl_query)

        # 将助手返回的消息添加到会话状态中
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        # 显示助手返回的消息
        with st.chat_message('assistant'):
            st.markdown(response)
    except Exception as e:
        # 如果执行查询时发生异常，则显示错误信息
        st.error(f"查询失败：{e}")