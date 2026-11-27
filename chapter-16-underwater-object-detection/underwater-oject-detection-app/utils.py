from ultralytics import YOLO
import supervision as sv
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe, Agent
from deepseek_llm import DeepSeekLLM
import os

# 加载预训练的 YOLO 模型
model = YOLO('kaggle/working/runs/detect/train/weights/best.pt')

# 定义执行目标检测的函数
def detect_objects(image, conf_threshold):
    # 使用模型进行预测
    results = model.predict(source = image, conf = conf_threshold)

    # 从结果中提取检测信息
    detections = sv.Detections.from_ultralytics(results[0])

    # 创建图片副本以便绘制边界框
    frame = image.copy()

    # 根据检测结果绘制边界框并标注类别和置信度
    # 创建矩形框注释器
    box_annotator = sv.BoxAnnotator()

    # 使用注释器在图片上绘制边界框
    box_annotator.annotate(
        scene = frame,
        detections = detections
    )

    # 实例化 LabelAnnotator 类
    # 创建标签注释器
    label_annotator = sv.LabelAnnotator()
    # 获得各边界框的标签及置信度
    labels = [
        f"{class_name} {confidence:.2f}" for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]

    # 使用注释器在图片上添加标签
    label_annotator.annotate(
        scene = frame,
        detections = detections,
        labels = labels
    )

    return results, frame

def generate_detections_df(filename, image_np, conf_threshold):
    # 初始化空的 DataFrame
    detections_df = pd.DataFrame(
        columns = ["Image Name", "Class Name", "Confidence", "Bounding Box"]
    )

    rows = []
    # 获取当前图片的检测结果
    results = model.predict(
        source = image_np,
        conf = conf_threshold
    )
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
    # 并与 detections_df 合并
    detections_df = pd.concat([detections_df, new_df], ignore_index = True)
    # 去除重复行
    detections_df = detections_df.drop_duplicates(
        subset = ["Image Name", "Class Name", "Confidence", "Bounding Box"]
    )
    return detections_df

def process_natural_language_query(history_df):
    # 如果会话状态中不存在 "messages" 键，则进行初始化
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 创建一个按钮，用于重置会话状态
    reset_button = st.sidebar.button(" 重置会话 ")
    if reset_button:
        st.session_state.messages.clear()

    # 遍历会话状态中的信息
    for message in st.session_state.messages:
        # 根据消息中的角色，显示相应的消息内容
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # 通过文本输入框获取用户的自然语言查询
    nl_query = st.sidebar.text_input(
        "输入自然语言查询",
        "检测到的类别名称有哪些？"
    )

    try:
        deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
    except KeyError:
        print("环境变量不存在")

    # 构建一个 OpenAI API 客户端实例，传入历史数据框和配置信息
    llm = DeepSeekLLM(
        api_key = deepseek_api_key
    )

    smart_df = SmartDataframe(history_df, config = {'llm': llm})

    try:
        # 调用 SmartDataFrame 实例，传入历史数据框和配置信息
        response = smart_df.chat(nl_query + ' 用中文回答，不要画图 ')

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
            "role": "assistant",
            "content": response
        })
        # 显示助手返回的消息
        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        # 如果执行查询时发生异常，则显示错误信息
        st.error(f" 查询失败: {e}")