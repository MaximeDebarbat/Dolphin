
import os
import argparse
from typing import Dict, Tuple, List, Union
import cv2
import gdown
import numpy as np
import dolphin as dp


def download_required_data(args: argparse.Namespace,
                           onnx_link: str,
                           video_link: str):

    if not os.path.exists(args.onnx):
        gdown.download(url=onnx_link, output=args.onnx, quiet=False, fuzzy=True)

    if not os.path.exists(args.video):
        gdown.download(url=video_link, output=args.video, quiet=False, fuzzy=True)


def prepare_classes() -> List[Dict[str, str]]:
    """
    This function creates a list of dictionnaries containing
    the coco classes and a random color for each class.
    """
    coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    res = []

    for element in coco_classes:
        res.append({
            "name": element,
            "color": tuple(np.random.randint(0, 255,
                                             size=(3,)).astype(int).tolist())
        })

    return res


def draw(frame: np.ndarray,
         classes: List[Dict[str, str]],
         output: Dict[str, Union[dp.darray, np.ndarray]],
         r: Tuple[float, float],
         dwdh: Tuple[int, int],
         fps: int) -> np.ndarray:

    if isinstance(output["det_boxes"], dp.darray):
        boxes = output["det_boxes"].to_numpy()
        num_dets = output["num_dets"].to_numpy()
        score_dets = output["det_scores"].to_numpy()
        class_dets = output["det_classes"].to_numpy()
    else:
        boxes = output["det_boxes"]
        num_dets = output["num_dets"]
        score_dets = output["det_scores"]
        class_dets = output["det_classes"]

    line_thickness = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
    label_thickness = max(line_thickness - 1, 1)  # font thickness

    frame = cv2.rectangle(frame,
                          (0, 0),
                          (int(750), int(40)),
                          (0, 0, 0),
                          -1,
                          cv2.LINE_AA)
    frame = cv2.putText(frame, f"Number of images processed/s: {fps:.2f}",
                        (0, 30),
                        0,
                        1e-3 * frame.shape[0],
                        (255, 255, 255),
                        2)

    for batch in range(num_dets.shape[0]):
        for index in range(len(boxes[batch][:num_dets[batch][0]])):
            bbox = boxes[batch][:num_dets[batch][0]][index]
            score = score_dets[batch][:num_dets[batch][0]][index]
            label = class_dets[batch][:num_dets[batch][0]][index]
            class_name = classes[int(label)]["name"]
            class_color = classes[int(label)]["color"]

            t_size = cv2.getTextSize(f"{class_name} : {score:.2f}",
                                     0,
                                     fontScale=line_thickness / 3,
                                     thickness=label_thickness)[0]

            c1 = (int((bbox[0] - dwdh[0]) / r),
                  int((bbox[1] - dwdh[1]) / r))

            c2 = (int((bbox[2] - dwdh[0]) / r),
                  int((bbox[3] - dwdh[1]) / r))

            frame = cv2.rectangle(frame,
                                  c1,
                                  c2,
                                  color=class_color,
                                  thickness=line_thickness,
                                  lineType=cv2.LINE_AA
                                  )

            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

            frame = cv2.rectangle(frame,
                                  c1,
                                  c2,
                                  class_color,
                                  -1,
                                  cv2.LINE_AA)

            frame = cv2.putText(frame,
                                f"{class_name} : {score:.2f}",
                                (c1[0], c1[1] - 2),
                                0,
                                label_thickness / 3,
                                [225, 255, 255],
                                thickness=line_thickness,
                                lineType=cv2.LINE_AA)

    return frame


def letterbox(frame: np.ndarray,
              new_shape: Tuple[int, int]
              ) -> Tuple[np.ndarray, float, Tuple[int, int]]:

    shape = frame.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        frame = cv2.resize(frame,
                           new_unpad,
                           interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    frame = cv2.copyMakeBorder(frame,
                               top,
                               bottom,
                               left,
                               right,
                               cv2.BORDER_CONSTANT,
                               value=(127, 127, 127))  # add border
    frame = frame.transpose((2, 0, 1))/255  # hwc to chw

    return frame, r, (dw, dh)
