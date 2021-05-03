import logging
import queue
import socket
import sys


import cv2
from facenet_pytorch import MTCNN
import torch

# set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# set up queue
q = queue.Queue()


def initialize(debug=False):
    # local address
    host_address = ""
    host_port = 8889
    host = (host_address, host_port)

    # create UDP socket
    try:
        sock = socket.socket(family=socket.AF_INET,
                             type=socket.SOCK_DGRAM)
        sock.bind(host)
    except socket.error as err:
        print("--- ERROR: (1)", err)
        return False, False, False

    # facial detection algo
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(device=dev)

    if debug:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(device=dev)
        sock = None
        drone = False
    else:
        drone_address = "192.168.10.1"
        drone_cmd_port = 8889
        drone = (drone_address, drone_cmd_port)

        # open comms with drone
        try:
            command(drone, sock, "command")
            command(drone, sock, "streamon")
        except socket.error as err:
            print("--- ERROR: (2)", err)
            drone = False

    return drone, sock, mtcnn


def info(drone, sock):
    '''
    info(to_address, from_address) returns the internal state of the machine
    :param drone:
        address of the drone
    :param sock:
        host socket
    :return:
        displays state to std out
    '''

    battery = sock.sendto(b"battery?", drone)
    speed = sock.sendto(b"speed?", drone)
    flight_time = sock.sendto(b"time?", drone)
    height = sock.sendto(b"height?", drone)
    temp = sock.sendto(b"temp?", drone)
    baro = sock.sendto(b"baro?", drone)
    accel = sock.sendto(b"acceleration?", drone)
    att = sock.sendto(b"attitude?", drone)

    print("****INTERNAL INFO****")
    print(f"Battery: {battery:0.2f} %")
    print(f"Speed: {speed} cm/s")
    print(f"Flight Time: {flight_time} sec")
    print(f"Height: {height} cm")
    print(f"Temperature {temp} deg C")
    print(f"Pressure: {baro}")
    print(f"Acceleration: {accel}")
    print(f"Attitude: {att}")


def command(drone, sock, cmd):
    """
    command(to_address, from_address, command)
        sends command to drone from host
    :param drone:
        drone address
    :param sock:
        host address
    :param cmd:
        string command for drone
    :return:
        nothing
    """
    logger.info({"action": "send_command",
                 "command": cmd})
    action = cmd.encode("utf-8")
    sock.sendto(action, drone)


def launch(drone, sock):
    command(drone, sock, "takeoff")


def land(drone, sock):
    command(drone, sock, "land")


def rcv_response(sock):
    try:
        response, ip = sock.recvfrom(1024)
        logger.info({"action": "rcv_response",
                     "response": response})
    except socket.error as err:
        logger.error({"action": "rcv_response",
                      "error": err})


def draw(frame, box, prob, lms):
    if box is not None and prob is not None:
        for b, p in zip(box, prob):
            x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])

            # draw box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            # display probability
            cv2.putText(frame, str(p), (w, h), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)


def video(drone, sock, mtcnn, width=340, height=240):
    print("Starting video thread")
    if drone is not None and sock is not None:
        address = "udp://@0.0.0.0:11111"
    else:
        address = 0

    vid = cv2.VideoCapture(address)
    success, frame = vid.read()
    print("Capturing video")

    while success:
        success, frame = vid.read()

        if not success:
            vid = cv2.VideoCapture(address)
            success, frame = vid.read()

        frame = cv2.resize(frame, (width, height))
        box, prob, lms = mtcnn.detect(frame, landmarks=True)
        draw(frame, box, prob, lms)
        cv2.imshow("Drone Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            vid.release()
            cv2.destroyAllWindows()
            break


def video_display(mtcnn, width, height):

    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu = True
            print("Enabling GPU acceleration")
    except:
        gpu = False
        print("GPU not used")

    while True:
        if q.empty() is False:
            frame = q.get()

            if gpu:
                gpuFrame = cv2.cuda_GpuMat()
                gpuFrame.upload(frame)

                resized = cv2.cuda.resize(gpuFrame, (width, height))
                frame = resized.download()

            else:
                frame = cv2.resize(frame, (width, height))

            frame = cv2.resize(frame, (width, height))

            box, prob, lms = mtcnn.detect(frame, landmarks=True)
            draw(frame, box, prob, lms)
            cv2.imshow("Drone Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # vid.release()
                cv2.destroyAllWindows()
                break


def video_rcv(drone, sock):
    print("Starting video thread")
    if drone is not None and sock is not None:
        address = "udp://@0.0.0.0:11111"
    else:
        address = 0

    vid = cv2.VideoCapture(address)
    print("Capturing video")

    success, frame = vid.read()
    q.put(frame)

    while success:
        success, frame = vid.read()
        q.put(frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     vid.release()
        #     break