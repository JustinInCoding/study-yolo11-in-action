import streamlit as st
import time
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from utils import infer_uploaded_video

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS'] # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

model = YOLO("../kaggle/working/runs/detect/train/weights/best.pt")

# 设置页面配置
st.set_page_config(
    page_title = "Pothold Detection App",
    layout = "wide"
)

# 加载自定义 css 样式，显示指定编码为 utf-8
with open("style.css", encoding = 'utf-8') as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html = True
    )

# 页面标题
st.markdown(
    "<h1 class='stHeader'>路面坑洞检测平台</h1>",
    unsafe_allow_html = True
)

# 侧边栏参数设置
with st.sidebar.form(key = 'my_form'):
    st.title(" 模型参数设置 ")
    # 添加置信度阈值输入框
    conf_thres = st.number_input(
        " 置信度阈值 ",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.5,
        step = 0.01
    )

    # 添加交并比阈值输入框
    iou_thres = st.number_input(
        " 交并比阈值： ",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.5,
        step = 0.01
    )

    # 添加保存带注释图片或视频的复选框
    save_annotated = st.checkbox(
        " 保存带注释图片 ",
        value = True
    )

    # 添加保存结果的复选框
    save_results = st.checkbox(
        " 保存结果 ",
        value = True
    )

    # 添加包含置信度的复选框
    include_confidence = st.checkbox(
        " 结果包含置信度 ",
        value = True
    )

    # 添加裁剪检测对象的复选框
    crop_objects = st.checkbox(
        " 保存裁剪对象 ",
        value = False
    )

    submit_button = st.form_submit_button(label = ' 提交 ')

# 创建图片和视频两个标签页，默认是图片界面
tabs = st.tabs(["Image", "Video"])

# 图片界面
with tabs[0]:
    st.markdown(
        "<p class='stHeader'>上传一张图片</p>",
        unsafe_allow_html = True
    )
    # 添加图片上传器
    image_file = st.file_uploader("", type = ["jpg", "jpeg", "png"])

    if image_file is not None:
        # 创建左中右三列布局
        col_left, col_middle, col_right = st.columns([2, 2, 1])

        with col_left:
            # 显示上传的图片
            st.image(
                image_file,
                caption = "Uploaded Image",
                # use_column_width = True
                width='stretch'
            )

        with col_middle:
            start_time = time.time()
            # 将上传的图片转换为 OpenCV 格式
            image_bytes = np.asarray(bytearray(image_file.read()), dtype = np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            # 处理图片并获取标注后的帧
            results = model(
                image,
                conf = conf_thres,
                iou = iou_thres,
                save = save_annotated,
                save_txt = save_results,
                save_conf = include_confidence,
                save_crop = crop_objects
            )
            annotated_frame = results[0].plot()
            end_time = time.time()

            detection_time = end_time - start_time

            # 显示检测结果
            st.image(
                annotated_frame,
                caption = "Detection Result",
                use_container_width = True
            )

        with col_right:
            # 创建容器，用于显示指标卡
            with st.container():
                st.subheader(" 检测结果 ")
                # 计算检测总目标数
                total_targets = len(results[0].boxes)

                # 计算坑洞总面积占比（假设每个检测框代表一个坑洞）
                areas = [box.xywh[0][2] * box.xywh[0][3] for box in results[0].boxes]
                total_area = sum(areas)
                image_height, image_width, _ = image.shape
                total_image_area = image_height * image_width
                pothole_area_percentage = (total_area / total_image_area) * 100

                # 展示指标卡
                st.metric(
                    label = " 总目标数 ",
                    value = total_targets
                )
                st.metric(
                    label = " 坑洞面积比例 (%) ",
                    value = f"{pothole_area_percentage:.2f}"
                )
                st.metric(
                    label = " 检测耗时 (s) ",
                    value = f"{detection_time:.2f}"
                )

        # 创建左右两列布局
        col1, col2 = st.columns([3, 2])
        with col1:
            # 创建一个空的列表来存储检测结果
            data = []
            for i, box in enumerate(results[0].boxes):
                confidence = box.conf.item()
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                width = xmax - xmin
                height = ymax - ymin
                area = width * height
                pothole_area_percentage = (area / (image.shape[0] * image.shape[1])) * 100
                data.append(
                    {
                        " 目标编号 ": i + 1,
                        " 置信度 ": confidence,
                        " 坑洞面积比例 (%) ": pothole_area_percentage,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }
                )

            # 定义一个函数来高亮最大值
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: skyblue' if v else '' for v in is_max]
            
            # 创建 DataFrame
            df = pd.DataFrame(data)

            # 应用样式并格式化数值列
            styled_df = df.style.apply(highlight_max).format(
                {
                    " 置信度 ": "{:.2f}",
                    " 坑洞面积比例 (%) ": "{:.2f}",
                    "xmin": "{:.2f}",
                    "ymin": "{:.2f}",
                    "xmax": "{:.2f}",
                    "ymax": "{:.2f}"
                }
            )

            # 显示 DataFrame
            st.dataframe(styled_df)

        with col2:
            # 提取置信度和坑洞面积比例
            confidences = df[" 置信度 "]
            area_percentages = df[" 坑洞面积比例 (%) "]

            # 创建条形图
            fig, ax = plt.subplots(figsize = (6, 4))
            index = np.arange(len(df))
            bar_width = 0.35

            bars1 = ax.bar(index, confidences, bar_width, label = ' 置信度 ')
            bars2 = ax.bar(index + bar_width, area_percentages, bar_width, label = ' 坑洞面积比例 (%) ')

            ax.set_xlabel(' 目标编号 ')
            ax.set_ylabel(' 值 ')
            ax.set_title(' 置信度和坑洞面积比例 ')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(df[" 目标编号 "])
            ax.legend()

            # 显示条形图
            st.pyplot(fig)


with tabs[1]:
    st.markdown(
        "<p class='stHeader'>上传一个视频</p>",
        unsafe_allow_html = True
    )
    # 添加视频上传器
    # video_file = st.file_uploader("", type = ["mp4", "avi", "mov"])

    # 显示视频及实时检测结果
    infer_uploaded_video(conf_thres)

