import streamlit as st
from image_recognition import infer_uploaded_image
from video_recognition import infer_uploaded_video

# 设置页面配置
st.set_page_config(
    page_title = "YOLO 11 实战",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# 标题
st.title("交通标志识别平台")

# 置信度阈值滑块
conf = st.sidebar.slider(" 置信度阈值 ", min_value = 5, max_value = 100, value = 20)

# 数据类型选择框
data_type = st.sidebar.selectbox(" 数据类型 ", [" 图片 ", " 视频 "])

# 根据不同的数据类型返回不同结果
if data_type == " 图片 ":
    infer_uploaded_image(conf)
else:
    infer_uploaded_video(conf)
