# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from djitellopy import Tello
import cv2 as cv


def initialize():
    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    print(drone.get_battery())
    drone.streamoff()
    drone.streamon()
    return drone


def get_frame(drone, w=360, h=240):
    frame = drone.get_frame_read()
    frame = frame.frame
    img = cv.resize(frame, (w, h))
    return img


def main():
    drone = initialize()

    # width = 360
    # height = 240

    while True:
        img = get_frame(drone)

        cv.imshow("Image", img)
        c = cv.waitKey(0)
        if 'q' == chr(c & 255):
            # drone.land()
            drone.end()
            break


if __name__ == '__main__':
    main()
