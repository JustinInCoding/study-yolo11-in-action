import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import re

def get_output_paths(output_dir, base_name):
    """
    根据输出目录和基础名称生成图片和文本文件的路径

    参数：
    output_dir: 输出文件的目录
    base_name: 文件的基础名称

    返回：
    img_path: 图片文件的路径
    txt_path: 文本文件的路径
    """
    img_path = os.path.join(output_dir, f"{base_name}.jpg")
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    return img_path, txt_path

def save_results(results, save_images, save_txt, save_conf, img_path, txt_path, class_names):
    """
    保存检测结果为图片和文本文件

    参数：
    results: 检测结果，包含检测框和类别信息
    save_images: 是否保存图片文件
    save_txt: 是否保存文本文件
    save_conf: 是否保存置信度信息
    img_path: 图片文件路径
    txt_path: 文本文件路径
    class_names: 类别的名称列表
    """
    if save_images:
        annotated_frame = results[0].plot()
        cv2.imwrite(
            img_path,
            annotated_frame[:, :, ::-1] # 转换颜色空间以保存正确格式
        )

    if save_txt:
        txt_data = results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy(), results[0].boxes.xyxy.cpu().numpy()
        with open(txt_path, 'w') as f:
            for cls_id, conf, bbox in zip(*txt_data):
                class_name = class_names[int(cls_id)]
                f.write(
                    f"{int(cls_id)} \"{class_name}\" {conf:.6f} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                )

def get_base_names(directory):
    """
    从指定目录中获取所有 .jpg 文件和 .txt 文件的基本名称（不带扩展名）

    参数：
    directory (str): 文件所在的目录路径 

    返回：
    list: 排序后的基本名称列表，不包含重复项
    """
    # 筛选目录下所有 .jpg 和 .txt 文件
    files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.txt')]
    # 提取文件的基本名称（不带扩展名），去重并排序
    base_names = sorted(set([os.path.splitext(f)[0] for f in files]))
    return base_names

def read_text_file(file_path):
    """
    获取文本文件中的数据，并返回一个 DataFrame 对象

    参数：
    file_path (str): 文本文件的路径

    返回：
    DataFrame: 包含文件数据的 DataFrame 对象
    """
    pattern = re.compile(
        r'^(\d+)\s+"([^"]*)"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$'
    )
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                rows.append(m.groups())
    
    return pd.DataFrame(
        rows,
        columns=["Class_id", "Class_name", "Confidence", "X_min", "Y_min", "X_max", "Y_max"]
    )

    # 读取文本文件，自定义列名
    # return pd.read_csv(
    #     file_path,
    #     header = None,
    #     delimiter = ' ',
    #     names = ["Class_id", "Class_name", "Confidence", "X_min", "Y_min", "X_max", "Y_max"]
    # )
