import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ultralytics import YOLO
import io
from utils import create_directories, save_uploaded_files, detect_images, generate_detections_df, update_history_df, process_natural_language_query


model = YOLO('kaggle/working/runs/detect/train/weights/best.pt')

st.markdown(
"""
<h1 style='color: darkblue; font-weight: bold; text-align: center'> 火灾及烟雾检测平台 </h1>
""",
unsafe_allow_html = True
)

tab1, tab2 = st.tabs([" 单图片检测 ", " 多图片检测 "])

with tab1:
    st.header(" 单图片检测 ")

    # 添加图片上传组件
    uploaded_image = st.file_uploader(" 上传图片 ", type = ["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # 读取上传的图片
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        # 进行目标检测
        results = model(image_np)

        # 将检测结果绘制在图片上
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = box.conf[0]
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(image_np, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                # cv2.putText(image_np, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # 使用 PIL 绘制中文标签
                pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                font_path = "/System/Library/Fonts/Supplemental/Songti.ttc" # 中文字体文件路径
                font = ImageFont.truetype(font_path, 36) # 设置字体和字号
                draw.text((r[0], r[1] - 40), label, font = font, fill = (0, 255, 0))

                # 转换回 OpenCV 格式
                image_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 将 NumPy 数组转换回 PIL 图片
        detected_image = Image.fromarray(image_np)

        # 左右布局显示原图和检测结果
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption = " 原图片 ")
        with col2:
            st.image(detected_image, caption = " 检测结果 ")

            # 将处理后的图片保存到内存中
            img_byte_arr = io.BytesIO()
            detected_image.save(img_byte_arr, format = 'PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # 添加 “下载检测结果” 按钮
            st.download_button(
                label = " 下载检测结果 ",
                data = img_byte_arr,
                file_name = "detected_image.png",
                mime = "image/png"
            )

with tab2:
    st.header(" 多图片检测 ")
    create_directories()

    uploaded_files = st.file_uploader(" 上传多张图片 ", type = ["jpg", "jpeg", "png"], accept_multiple_files = True)

    if uploaded_files:
        image_paths = save_uploaded_files(uploaded_files) # 将上传的文件保存到本地

        if not image_paths:
            st.error(" 没有找到有效的图片文件 ")
        else:
            detect_images(model, image_paths, st.session_state)

            # 仅在会话状态中还没有 index 时初始化为第一张，避免每次 rerun 都被重置为最后一张
            if "index" not in st.session_state:
                st.session_state.index = 0
            # 防止重新上传更少图片时旧索引越界
            st.session_state.index = min(st.session_state.index, len(image_paths) - 1)
            index = st.session_state.index

            # 当用户单击 “上一张” 按钮时，更新图片索引，确保图片索引不会小于 0
            if st.button(" 上一张 "):
                index = max(index - 1, 0)
                st.session_state.index = index
            # 当用户单击 “下一张” 按钮时，更新图片索引，确保索引不会超过图片路径列表的长度
            if st.button(" 下一张 "):
                index = min(index + 1, len(image_paths) - 1)
                st.session_state.index = index

            col21, col22 = st.columns(2)
            with col21:
                # 根据当前索引获取图片路径，并展示该图片
                current_image_path = image_paths[index]
                st.image(current_image_path, caption = f"Original Image ({current_image_path})", use_column_width = True)
            with col22:
                # 从会话状态中获取当前图片的检测结果和标注的图片路径
                results = st.session_state.detections[current_image_path]['results']
                annotated_frame_path = st.session_state.detections[current_image_path]['annotated_frame_path']
                # 展示标注后的图片
                st.image(annotated_frame_path, caption = "Detected Objects", use_column_width = True)

            # 展示检测结果表格
            detections_df = generate_detections_df(st.session_state, model)
            history_df = update_history_df(detections_df)
            st.dataframe(history_df)

            # 允许用户输入自然语言查询
            process_natural_language_query(history_df)

    else:
        st.info(" 请上传多张图片进行预测 ")




    
