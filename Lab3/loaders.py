from numpy import ndarray
from cv2 import imread, imshow, waitKey, destroyAllWindows
from os.path import join
from typing import List

def read_image(
        i: int,
        folder: str,
        is_raw_image: bool = True,
        load_gray: bool = True,
        ) -> ndarray:
    return imread(join(folder, "input/in%06d.jpg" % i if is_raw_image else "groundtruth/gt%06d.png" % i), 0 if load_gray else -1)


def load_sequence(
        folder: str,
        is_raw_image: bool = True,
        load_gray: bool = True,
        ) -> List[ndarray]:
    with open(join(folder, "temporalROI.txt")) as f:
        start, end = [int(x) for x in next(f).split()]
    return [read_image(i, folder, is_raw_image, load_gray) for i in range(start, end)]


def visualize(images: List[ndarray]) -> None:
    for image in images:
        imshow("I",image)
        waitKey(10)
    destroyAllWindows()

if __name__ == "__main__":
    pass
