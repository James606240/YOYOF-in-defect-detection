from operator import indexOf
import xml.etree.ElementTree as ET
import cv2

def get_annots(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = list()
    bboxes = list()
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = obj.find('difficult')
        difficult = 0 if difficult is None else int(difficult.text)
        bnd_box = obj.find('bndbox')
        # TODO: check whether it is necessary to use int
        # Coordinates may be float type
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        names.append(name)
        bboxes.append(bbox)
    return names, bboxes

def draw_bboxes(image, names, bboxes, colors, classes):
    for i, box in enumerate(bboxes):
        color = colors[int(classes.index(names[i]))][::-1]
        class_name = names[i]
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            image,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            color,
            1, cv2.LINE_AA
        )
        cv2.putText(
            image,
            str(class_name),
            (int(x_min), int(y_min)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85, color,
            thickness=2,
            lineType=cv2.LINE_AA
        )
    return image