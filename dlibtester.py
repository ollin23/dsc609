import face_recognition
import cv2
import numpy as np
from facenet_pytorch import MTCNN


def main():
    vid = cv2.VideoCapture(0)
    mtcnn = MTCNN(device="cuda")

    # load and learn
    nick = face_recognition.load_image_file("images/nick/nick1.jpg")
    nick_encoding = face_recognition.face_encodings(nick)[0]

    alex = face_recognition.load_image_file("images/alex4.jpg")
    alex_encoding = face_recognition.face_encodings(alex)[0]

    # arrays of known faces and encodings
    known_encodings = [nick_encoding, alex_encoding]
    known_faces = ["Nick", "Alex"]

    face_locs = []
    face_encodings = []
    face_names = []
    process_this = True

    # scaling factors
    scale: float = 0.25

    # main video loop
    while True:
        success, frame = vid.read()

        # resize for faster processing
        smaller_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # BGR to RGB
        rgb_frame = cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2RGB)

        # process faces in video
        if process_this:

            # find faces and respective encodings
            face_locs = face_recognition.face_locations(rgb_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locs)

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
            color = (0, 255, 0)

            box, prob, lms = mtcnn.detect(frame, landmarks=True)
            draw(frame, box, prob, lms)

            # cv2.rectangle(frame,
            #               startpoint,
            #               endpoint,
            #               color,
            #               1)

            cv2.putText(frame, name, (left+5, bottom+5), cv2.FONT_HERSHEY_SIMPLEX,
                                      1.0, (255,255,255), 1)

        # display video
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release UDP socket, close OpenCV window, quit script/app
    vid.release()
    cv2.destroyAllWindows()
    quit()


def draw(frame, box, prob, lms):

    for b, p in zip(box, prob):
        x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])

        # draw box
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

        # display probability
        cv2.putText(frame, str(p), (w, h), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()