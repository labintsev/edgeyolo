import os
import xml.etree.ElementTree as et
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple

import PIL
from PIL import ImageDraw
from PIL.JpegImagePlugin import JpegImageFile


class Rectangle(NamedTuple):
    """Хранит координаты прямоугольника (xmin, ymin) - (xmax, ymax)"""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def w(self) -> int:
        """Ширина"""
        return self.xmax - self.xmin

    @property
    def h(self) -> int:
        """Высота"""
        return self.ymax - self.ymin

    @property
    def square(self) -> float:
        """Площадь"""
        return self.w * self.h

    def __repr(self) -> str:
        return f'Rectangle(x1={self.xmin},y1={self.ymin},x2={self.xmax},y2={self.ymax})'


class Annotation(NamedTuple):
    """Аннотация к изображению - bbox + класс объекта"""
    label: str
    bbox: Rectangle


class AnnotationFileReader:
    """Чтение файла с аннотациями из LADD (Pascal VOC)"""

    def __init__(self, filepath: str) -> None:
        self.filepath: Path = Path(filepath)

    def read_annotations(self) -> List[Annotation]:
        annotations = []
        root = et.parse(str(self.filepath)).getroot()
        for obj in root.iter('object'):
            bndbox = obj.find('bndbox')
            assert bndbox is not None
            annotation = Annotation(
                label=self._text(obj.find('name'), default=''),
                bbox=Rectangle(
                    xmin=int(self._text(bndbox.find('xmin'), default='0')),
                    ymin=int(self._text(bndbox.find('ymin'), default='0')),
                    xmax=int(self._text(bndbox.find('xmax'), default='0')),
                    ymax=int(self._text(bndbox.find('ymax'), default='0')),
                )
            )
            annotations.append(annotation)
        return annotations

    def _text(self, element: Optional[et.Element], default: str) -> str:
        if element is None:
            return default
        text = element.text
        if text is None:
            return default
        return text

    def __repr__(self) -> str:
        path = str(self.filepath)
        return f"AnnotationFile('{path}')"


def scale(src, x_factor, y_factor) -> Annotation:
    """Масштабирование координат"""
    return Annotation(
        label=src.label,
        bbox=Rectangle(
            xmin=round(src.bbox.xmin * x_factor),
            xmax=round(src.bbox.xmax * x_factor),
            ymin=round(src.bbox.ymin * y_factor),
            ymax=round(src.bbox.ymax * y_factor)
        )
    )


def shift(src, x_shift, y_shift) -> Annotation:
    """Сдвиг координат"""
    return Annotation(
        label = src.label,
        bbox = Rectangle(
            xmin = round(src.bbox.xmin - x_shift),
            xmax = round(src.bbox.xmax - x_shift),
            ymin = round(src.bbox.ymin - y_shift),
            ymax = round(src.bbox.ymax - y_shift)
        )
    )


def overlap_annotations(scaled_anns, left, top, right, bottom, crop_size) -> List:
    """Пересечение аннотаций с кропом изображения"""
    crop_anns = []
    for ann in scaled_anns:
        if ann.bbox.xmin >= left and ann.bbox.ymin >= top:
            if ann.bbox.xmax <= right and ann.bbox.ymax <= bottom:
                crop_anns.append(shift(ann, left, top))
            else:
                if ann.bbox.xmax - right < ann.bbox.w/3 and ann.bbox.ymax - bottom < ann.bbox.h/3:
                    crop_anns.append(Annotation(label=ann.label, bbox=Rectangle(
                        xmin=ann.bbox.xmin-left, ymin=ann.bbox.ymin-top, 
                        xmax=min(crop_size, ann.bbox.xmax - left), ymax=min(crop_size, ann.bbox.ymax - top))))
    return crop_anns


def load_img_and_annotations(idx: int, voc_dir: str) -> Tuple[JpegImageFile, List[Annotation]]:
    img_path = os.path.join(voc_dir, 'JPEGImages', f'{idx}.jpg')
    img = PIL.Image.open(img_path)
    ann_path = os.path.join(voc_dir, 'Annotations', f'{idx}.xml')
    anns = AnnotationFileReader(ann_path).read_annotations()
    return img, anns


def crop_sample(img, annotations, crop_size, top_n, width_crops, height_crops
                ) -> List[Tuple[JpegImageFile, List[Annotation]]]:
    """Crop image for width_crops * height_crops,
    find new coordinates of annotations in every crop,
    resize every img to crop_size

    @return: top_n crops with maximum count of bounding boxes"""

    k_x = width_crops * crop_size / img.width
    k_y = height_crops * crop_size / img.height
    scaled_anns = [scale(a, k_x, k_y) for a in annotations]

    out = []
    img_r = img.resize(size=(width_crops * crop_size, height_crops * crop_size))
    for w in range(width_crops):
        for h in range(height_crops):
            left = w * crop_size
            top = h * crop_size
            right = (w + 1) * crop_size
            bottom = (h + 1) * crop_size

            crop_img = img_r.crop((left, top, right, bottom))
            crop_anns = overlap_annotations(scaled_anns, left, top, right, bottom, crop_size)
            out.append((crop_img, crop_anns))

    return sorted(out, key=lambda x: len(x[1]), reverse=True)[:top_n]


def save_annotations_to_file(idx: int, annotations: List[Annotation], out_dir='./') -> bool:
    """
    Save VOC annotations to VisDrone file
    @param idx: index of image and annotation file
    @param annotations: list of bboxes coordinates [left, top, width, height]
    @param out_dir: directory to save output file
    @return: True if saving is done.
    """
    out = []
    for ann in annotations:
        label = ann.label
        box_left = ann.bbox.xmin
        box_top = ann.bbox.ymin
        box_width = ann.bbox.xmax - box_left
        box_height = ann.bbox.ymax - box_top
        score, truncation, occlusion = 1, 0, 0
        out.append(
            f'{box_left},{box_top},{box_width},{box_height},\
{score},1,{truncation},{occlusion}\n')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(out_dir, f'{idx}.txt')

    with open(out_path, 'w') as f:
        f.writelines(out)
    return True


def draw_boxes(i, image_dir, annot_dir)-> JpegImageFile:
    img_path = os.path.join(image_dir, f'{i}.jpg')
    img = PIL.Image.open(img_path)

    ann_path = os.path.join(annot_dir, f'{i}.txt')
    with open(ann_path) as f:
        anns_txt_lines = f.readlines()

    bboxes = []
    for ann in anns_txt_lines:
        ann = ann.split(',')
        bboxes.append(list(map(int, ann[:5])))

    draw = ImageDraw.Draw(img)
    for b in bboxes:
        draw.rectangle((b[0], b[1], b[0]+b[2], b[1]+b[3]), outline='red')
    return img
