from ultralytics import YOLO
import supervision as sv

# 定义执行目标检测的函数
def detect_objects(image, model_type, conf_threshold):

    # 根据模型类型选择不同的标题和模型文件
    if model_type == " 轴向 ":
        model = YOLO("axial_t1wce_best.pt")
    elif model_type == " 冠状位 ":
        model = YOLO("coronal_t1wce_best.pt")
    elif model_type == " 失状位 ":
        model = YOLO("sagittal_t1wce_best.pt")

    # 使用模型进行预测
    results = model.predict(
        source = image,
        conf = conf_threshold
    )

    # 从结果中提取检测信息
    detections = sv.Detections.from_ultralytics(results[0])

    # 创建图片副本以便绘制边界框
    frame = image.copy()

    # 根据检测结果绘制边界框并标注类别和置信度
    # 创建矩形框注释器
    box_annotator = sv.BoxAnnotator(color = sv.Color.from_hex('#FFFFFF'))

    # 使用注释器在图片上绘制边界框
    box_annotator.annotate(
        scene = frame,
        detections = detections
    )

    # 实例化 LabelAnnotator 类
    # 创建标签注释器
    label_annotator = sv.LabelAnnotator(
        text_scale = 0.3,
        color = sv.Color.from_hex('#FFFFFF'),
        text_color = sv.Color.from_hex('#00FF00')
    )
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
    return results, frame
