import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
from ultralytics import YOLO
from file_utils import save_results, get_output_paths
from video_recognotion import infer_uploaded_video

# 使用训练好的 best.pt 模型
model = YOLO("kaggle/working/runs/detect/train/weights/best.pt")

# 设置页面配置
# 页面标题为 "车道线检测平台"，布局为宽屏，侧边烂默认为展开
st.set_page_config(
    page_title = "车道线检测平台",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# 设置页面标题和描述
st.title("车道线检测平台")
st.markdown(
"""
这是一个基于 Ultralytics YOLO、Streamlit 和 PandasAI 的车道检测平台。
您可以上传图片或视频进行检测，并查看检测结果和保存的文件。
"""
)

# 定义输出目录
# 如果目录不存在，则创建它
output_dir = "runs/detect/predict"
os.makedirs(output_dir, exist_ok=True)

# 侧边栏的配置
# 用户可以通过侧边栏调整配置选项
with st.sidebar:
    st.header(" 配置选项 ")
    # 置信度阈值滑块，用于调整检测结果的置信度阈值
    confidence_threshold = st.slider(" 置信度阈值 ", min_value = 0.0, max_value = 1.0, value = 0.5)
    # 是否保存带有注释的图片或视频
    save_images = st.radio(" 是否保存带有注释的图片或视频 ", [" 是 ", " 否 "], index = 0) == ' 是 '
    # 是否保存检测结果到本地文件
    save_txt = st.radio(" 是否保存检测结果到本地文件 ", [" 是 ", " 否 "], index = 0) == ' 是 '
    # 是否在文本文件中包含置信度分数
    save_conf = st.radio(" 是否在文本文件中包含置信度分数 ", [" 是 ", " 否 "], index = 0) == ' 是 '

    # 用户可以选择输入类型：图片或视频
    input_type = st.radio(" 选择输入类型 ", [" 图片 ", " 视频 "])

# 如果上传了文件并且输入类型是图片
if input_type == " 图片 ":
    # 上传图片文件
    uploaded_file = st.sidebar.file_uploader(" 上 传 图 片 ", type = ["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # 将上传的文件读取为字节，并转换为 numpy 数组
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
        # 使用 OpenCV 从 numpy 数组中解码图片
        img = cv2.imdecode(image_bytes, 1)
        # 获取上传文件的名称（不带扩展名）
        base_name = os.path.splitext(uploaded_file.name)[0]

        # 使用模型进行图片检测，传入图片、置信度阈值、是否保存图片等多个参数
        results = model(
            img,
            conf = confidence_threshold,
            save = save_images,
            save_txt = save_txt,
            save_conf = save_conf,
            # project = output_dir,
            project = 'predict',
            name = '',
            exist_ok = True
        )

        # 创建两列布局用于展示原始图片和检测结果
        col1, col2 = st.columns(2)
        with col1:
            # 显示原始图片列的标题和图片内容
            st.subheader(" 原始图片 ")
            st.image(
                img,
                caption = " 原 始 图 片： " + base_name,
                use_column_width = True
            )
        with col2:
            # 显示检测结果列的标题和图片内容
            st.subheader(" 检测结果 ")
            if save_images:
                # 如果设置了保存图片，则获取标注后的图像帧并显示
                annotated_frame = results[0].plot()
                st.image(
                    annotated_frame,
                    caption = " 检测结果 ",
                    use_column_width = True
                )
                
                # 获取类别名称
                class_names = model.names

                # 设置输出路径
                img_path, txt_path = get_output_paths(output_dir, base_name)
                # 保存检测结果为图片和文本文件
                save_results(
                    results,
                    save_images,
                    save_txt,
                    save_conf,
                    img_path,
                    txt_path,
                    class_names
                )
else:
    infer_uploaded_video(confidence_threshold)

from file_utils import get_base_names, read_text_file
# 显示保存的文件信息
saved_files_path = output_dir
if os.path.exists(saved_files_path):
    # 获取保存文件的基本名称列表
    base_names = get_base_names(saved_files_path)

    # 创建一个下拉列表框，用于选择文件
    selected_base_name = st.selectbox(" 选择文件 ", base_names)

    # 当选择了一个文件时，显示相关的图片和文本内容
    if selected_base_name:
        # 根据指定的文件名获取图片和文本文件的路径
        img_path, txt_path = get_output_paths(saved_files_path, selected_base_name)

        # 显示检测结果和文本文件的标题
        st.subheader(" 检测结果和文本文件 ")
        # 创建两个列布局，用于同时显示图片和文本内容
        col1, col2 = st.columns(2)
        with col1:
            # 检查并显示检测结果图片
            if os.path.exists(img_path):
                detected_image = cv2.imread(img_path)
                st.image(
                    detected_image,
                    caption = " 检测结果 ",
                    channels = "BGR",
                    use_column_width = True
                )
            else:
                st.write(" 检测结果图片不存在 ")
        with col2:
            # 检查并显示检测结果的文本内容
            if os.path.exists(txt_path):
                file_content = read_text_file(txt_path)
                st.dataframe(file_content)
            else:
                st.write(" 检测结果文本不存在 ")

from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe, Agent

# 创建一个按钮，用于重置会话状态
reset_button = st.button("  重新会话 ")
if reset_button:
    st.session_state.clear()
# 允许用户输入自然语言查询
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

nl_query = st.sidebar.text_input("输入自然语言查询", "检测到哪些目标")
# 初始化 OpenAI API 客户端
# llm = OpenAI(
#     api_base = "https://api.deepseek.com",
#     api_token = "xxxxx",
#     model = "deepseek-chat"
# )
from deepseek_llm import DeepSeekLLM
llm = DeepSeekLLM(api_key="sk-xxxxxxx", model="deepseek-flash")

# 创建 SmartDataframe 实例，用于处理数据框和自然语言查询
smart_df = SmartDataframe(
    file_content,
    config = {"llm": llm}
)
try:
    # 执行用户输入的自然语言查询，并指定回答形式为文字
    response = smart_df.chat(
        nl_query + ' 用中文回答，不要画图 '
    )
    # 添加用户消息到会话状态
    st.session_state.messages.append(
        {
            "role": "user",
            "content": nl_query
        }
    )
    with st.chat_message("user"):
        st.write(nl_query)
    # 添加助手消息到会话状态
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
    with st.chat_message("assistant"):
        st.markdown(response)
except Exception as e:
    st.error(f"查询失败：{e}")


