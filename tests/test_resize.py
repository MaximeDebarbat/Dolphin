import numpy as np
import dolphin as dp
import cv2


def main():

    image = np.random.rand(50, 70) * 100
    image = image.astype(np.uint8)
    dimage = dp.dimage(array=image)

    dimage_res = dp.dimage(shape=(100, 200), dtype=dp.uint8)
    res = dimage.resize((200, 100), dst=dimage_res)
    res_np = cv2.resize(dimage.np, (200, 100), interpolation=cv2.INTER_NEAREST)

    for i in range(100):
        for j in range(200):
            if res_np[i, j] != res.np[i, j]:
                print(f"Error at {i}, {j} : {res_np[i, j]} != {res.np[i, j]}")

if __name__ == "__main__":
    main()