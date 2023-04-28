
import argparse
import time
from typing import List, Any
from utils import prepare_classes, download_required_data, draw, letterbox
import numpy as np
import cv2
import onnxruntime as ort


def set_execution_provider() -> List[Any]:

    freeGpuMem = 2 * 1024 * 1024 * 1024
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": freeGpuMem,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider"
    ]


def run(opt: argparse.Namespace):

    # We create or load the engine here, activate verbose to see the logs
    # verbosity=True
    ort_sess = ort.InferenceSession(opt.onnx,
                                    providers=set_execution_provider())

    # We create the stream
    cap = cv2.VideoCapture(opt.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_output = cv2.VideoWriter(opt.export,
                                   fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
                                   fps=fps,
                                   frameSize=(width, height))

    # We get the classes
    classes = prepare_classes()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.time()

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, r, dwdh = letterbox(processed_frame, (640, 640))
        processed_frame = np.expand_dims(processed_frame, 0)

        output = ort_sess.run(None, {"images": processed_frame.astype(np.float32)})[0]

        t2 = time.time() - t1

        print(f"FPS: {1/t2}")

        output = {
            "det_boxes": np.array([output[:, 1:5]]),
            "num_dets": np.array([[len(output)]]),
            "det_classes": np.array([output[:, 5]]),
            "det_scores": np.array([output[:, 6]])
        }

        drawn_frame = draw(frame=frame,
                           classes=classes,
                           output=output,
                           r=r,
                           dwdh=dwdh,
                           fps=1/t2)
        video_output.write(drawn_frame)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="yolov7-tiny-ort.onnx",
                        help="Path to \
                        the onnx file. If not specified, \
                        it will be downloaded from Google Drive.")
    parser.add_argument("--video", type=str, default="dolphin.mp4",
                        help="Path to \
                        the video file. If not specified, \
                        it will be downloaded from Google Drive.")
    parser.add_argument("--export", type=str, default="result-ort.mp4",
                        help="Path to the result video.")

    args = parser.parse_args()

    ONNX_PATH: str = "https://drive.google.com/file/d/1xQltee1bzurl5r22qCNcPKv2PgKLVM6Z/view?usp=share_link"
    VIDEO_PATH: str = "https://drive.google.com/file/d/10wrpxH2R5DN81V0RNp01Y8A_0RAMlOnk/view?usp=share_link"

    download_required_data(args, ONNX_PATH, VIDEO_PATH)
    run(args)
