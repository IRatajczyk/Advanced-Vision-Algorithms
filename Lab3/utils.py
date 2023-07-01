from numpy import ndarray, zeros, sum as np_sum, logical_and, uint8, float64, argmax, int as np_int, logical_and
from typing import List, Tuple, Callable
from cv2 import imread, imshow, waitKey, destroyAllWindows, connectedComponentsWithStats, rectangle, putText, FONT_HERSHEY_SIMPLEX
from os.path import join

def read_image(
        i: int,
        folder: str,
        is_raw_image: bool = True,
        load_gray: bool = True,
        ) -> ndarray:
    image = imread(join(folder, "input/in%06d.jpg" % i if is_raw_image else "groundtruth/gt%06d.png" % i), 0 if load_gray else -1)
    return image if is_raw_image else (image>200).astype(uint8)*255


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

def visualize_bounding_box(raw_images: List[ndarray], processed_images: List[ndarray]) -> None:
    for raw_img, processed_img in zip(raw_images, processed_images):
        image = raw_img.copy()
        _, _, stats, centroids = connectedComponentsWithStats(processed_img)
        if stats.shape[0] > 1:
            pi = argmax(stats[1:,4]) + 1
            rectangle(image, (stats[pi, 0],stats[pi, 1]), (stats[pi, 0]+stats[pi, 2], stats[pi, 1]+stats[pi, 3]), (255,0,0), 2)
            putText(image,"%f" % stats[pi,4], (stats[pi,0], stats[pi, 1]), FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            putText(image,"%d" %pi,(np_int(centroids[pi,0]), np_int(centroids[pi,1])), FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        imshow("I", image)
        waitKey(10)
    destroyAllWindows()


def calculate_metrics(images: List[ndarray], ground_truth: List[ndarray]) -> Tuple[float]:
    """
    Calculate the metrics for the given images and ground truth
    :param images: List of images
    :param ground_truth: List of ground truth images
    :return: Tuple of F1-score, precision and recall respectively
    """
    true_positive: float = 0
    false_positive: float = 0
    false_negative: float = 0
    for (image, truth) in zip(images, ground_truth):
        true_positive += np_sum(logical_and((image == 255), (truth == 255)).astype(float64))+1.
        false_positive += np_sum(logical_and((image == 255), (truth == 0)).astype(float64))+1.
        false_negative +=np_sum(logical_and((image == 0), (truth == 255)).astype(float64))+1.
    p = true_positive/(true_positive+false_positive)
    r = true_positive/(true_positive+false_negative)
    return 2*p*r/(p+r), p, r