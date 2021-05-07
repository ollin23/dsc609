from controls import *

import face_recognition
import cv2
import numpy as np
from pynput import keyboard


color_known = (0, 200, 0)
color_unknown = (0, 255, 255)
speed = 10

# set detection flags
mtcnn_on = False
haar_on = False
dlib_on = False


def on_press(key):
    global mtcnn_on, haar_on, dlib_on
    try:
        if key.char == "1":
            mtcnn_on = not mtcnn_on
        if key.char == "2":
            haar_on = not haar_on
        if key.char == "3":
            dlib_on = not dlib_on
    except:
        print(f"special key pressed: {key}")


def main():
    debug = True
    drone, sock, mtcnn = initialize(debug)


    # keyboard listener
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    if drone is not None and sock is not None:
        address = "udp://@0.0.0.0:11111"
    else:
        address = 0

    vid = cv2.VideoCapture(address)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load faces and obtain encodings
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # specific to implementation
    face = face_recognition.load_image_file("images/nick/nick1.jpg")
    face_enc = face_recognition.face_encodings(face)[0]

    known_encodings = [face_enc]
    known_faces = ["Nick"]

    face_locs = []
    face_encodings = []
    face_names = []
    process_this = True

    # scaling factors
    scale: float = 0.25

    # launch drone
    if drone:
        launch(drone, sock)

    # main video loop
    while True:

        success, frame = vid.read()
        frame_x = frame.shape[1]
        frame_y = frame.shape[0]
        frame_area = frame_x * frame_y

        # resize for faster processing
        smaller_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # BGR to RGB
        rgb_frame = cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2RGB)

        # process faces in video
        if process_this:

            # find faces and respective encodings
            face_locs = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locs)


            # label images with name(s)
            face_names = []
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "unknown"

                face_distances = face_recognition.face_distance(known_encodings, encoding)
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    name = known_faces[best_match_idx]

                face_names.append(name)
        process_this = not process_this

        # draw bounding boxes
        for (top, right, bottom, left), name in zip(face_locs, face_names):
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)

            startpoint = (right, top)
            endpoint = (left, bottom)

            # MTCNN in red
            if mtcnn_on:
                box, prob, lms = mtcnn.detect(frame, landmarks=True)
                draw(frame, box, prob, lms)

            # Haar Cascade in blue
            if haar_on:
                haar_cascade(frame)

            # DLIB, known in green uknown in yellow
            if dlib_on:
                if name == "unknown":
                    cv2.rectangle(frame,
                                  startpoint,
                                  endpoint,
                                  color_unknown,
                                  1)
                    face_x = (left + right) // 2
                    face_y = (top + bottom) // 2

                    diff_x = frame_x - face_x
                    diff_y = frame_y - face_y
                    area = (right - left) * (bottom - top)
                    face_percent = area / frame_area

                    # keep drone centered
                    if drone:
                        x, y, z = 0, 0, 0
                        if diff_x > 30:
                            y = 30
                        if diff_x < -30:
                            y = -30

                        if diff_y > 20:
                            z = 20
                        if diff_y < -20:
                            z = -20

                        if face_percent > 0.30:
                            x = -30
                        if face_percent < 0.05:
                            x = 30

                        command(drone, sock, f"go {x} {y} {z} {speed}")


                else:
                    cv2.rectangle(frame,
                                  startpoint,
                                  endpoint,
                                  color_known,
                                1)
                cv2.putText(frame, name, (left+5, bottom+5), cv2.FONT_HERSHEY_SIMPLEX,
                                          1.0, (255,255,255), 1)

        # display video
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # land drone
    if drone:
        land(drone, sock)

    # release UDP socket, close OpenCV window, quit script/app
    vid.release()
    cv2.destroyAllWindows()
    quit()

if __name__ == "__main__":
    main()
