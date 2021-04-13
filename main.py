# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from facenet_pytorch import MTCNN
import torch
from detector import *
import cv2


# from utils import *
WIDTH = 360
HEIGHT = 240


def main():
    drone = initialize()

    width = WIDTH
    height = HEIGHT

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(device=dev)

    detect(drone, mtcnn, width, height)
    shutdown(drone)
    exit(0)


if __name__ == '__main__':
    main()
