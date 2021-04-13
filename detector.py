import cv2
from djitellopy import Tello


def initialize():
    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    print(drone.get_battery())

    drone.streamon()
    return drone


def shutdown(drone):
    drone.streamoff()
    drone.land()
    exit(0)


def detect(drone, mtcnn, width=360, height=240):

    while True:

        # read stream from drone
        got_frame = drone.get_frame_read()
        frame = got_frame.frame
        vid = cv2.resize(frame, (width, height))
        vid = cv2.resize(vid, (width, height))

        box, prob, lms = mtcnn.detect(vid, landmarks=True)
        draw(vid, box, prob, lms)
        cv2.imshow("Drone Feed", vid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def draw(frame, box, prob, lms):

    if box is not None and prob is not None:
        for b, p, ld in zip(box, prob, lms):
            x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])

            # draw box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            # display probability
            cv2.putText(frame, str(p), (w, h), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)