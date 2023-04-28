
import argparse
import time
import numpy as np
import cv2
from utils import prepare_classes, download_required_data, draw, letterbox
import dolphin as dp


def run(opt: argparse.Namespace):

    # We create or load the engine here, activate verbose to see the logs
    # verbosity=True
    engine = dp.Engine(opt.onnx,
                       opt.engine,
                       mode="fp16",
                       verbosity=True)

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

    stream = dp.Stream()

    inference_frame = dp.dimage(shape=(3, 640, 640),
                                dtype=dp.float32,
                                stream=stream)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.time()

        # We process the frame
        # 1. We convert the frame into RGB
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. We resize and transpose the frame
        processed_frame, r, dwdh = letterbox(processed_frame, (640, 640))

        inference_frame.ndarray = processed_frame.astype(np.float32)

        output = engine.infer({"images": inference_frame})

        t2 = time.time() - t1

        print(f"FPS: {1/t2}")

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
    parser.add_argument("--onnx", type=str, default="yolov7-tiny.onnx",
                        help="Path to \
                        the onnx file. If not specified, \
                        it will be downloaded from Google Drive.")
    parser.add_argument("--engine", type=str, default="yolov7-tiny.engine",
                        help="Path to \
                        the engine file. If not specified, \
                        it will be created.")
    parser.add_argument("--video", type=str, default="dolphin.mp4",
                        help="Path to \
                        the video file. If not specified, \
                        it will be downloaded from Google Drive.")
    parser.add_argument("--export", type=str, default="result-naiveTRT.mp4",
                        help="Path to the result video.")

    args = parser.parse_args()

    ONNX_PATH: str = "https://drive.google.com/file/d/14VOCCX88rQVbko9E-KVObk5HFWfl3xnJ/view?usp=share_link"
    VIDEO_PATH: str = "https://drive.google.com/file/d/10wrpxH2R5DN81V0RNp01Y8A_0RAMlOnk/view?usp=share_link"

    download_required_data(args, ONNX_PATH, VIDEO_PATH)
    run(args)
