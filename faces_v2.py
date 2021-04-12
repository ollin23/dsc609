import cv2
from facenet_pytorch import MTCNN
import torch


class Detector(object):

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def draw(self, frame, box, prob, lms):

        if box is not None and prob is not None:
            for b, p , ld in zip(box, prob, lms):
                x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])

                # draw box
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                # display probability
                cv2.putText(frame, str(p), (w, h), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

    def run(self):
        vid = cv2.VideoCapture(1)

        if not vid.isOpened:
            print("--(!)ERROR: Could not open video capture")
            exit(0)

        while True:
            _, frame = vid.read()

            box, prob, lms = self.mtcnn.detect(frame, landmarks=True)
            self.draw(frame, box, prob, lms)

            cv2.imshow("Face Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        vid.release()
        cv2.destroyAllWindows()


# EXECUTE SCRIPT

# use GPU if nVidia GPU available
dev = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(device=dev)
detector = Detector(mtcnn)
detector.run()
