from controls import *
import threading
import time

def main():
    debug = True
    tello, host, mtcnn = initialize(debug)

    if host is None or host is False:


        if host is not False:
            video_thr = threading.Thread(target=video,
                                         args=(tello, host, mtcnn, ))
            video_thr.start()
            video_thr.join()
        else:
            print("Could not establish connection. App terminated.")

        sys.exit(0)
    else:
        response_thr = threading.Thread(target=rcv_response,
                                        args=(host, ))
        response_thr.start()

        video_thr = threading.Thread(target=video,
                                     args=(tello, host, mtcnn, ))
        video_thr.start()

        response_thr.join()
        video_thr.join()
        # launch(tello, host)
        # time.sleep(2)
        # land(tello, host)

        # kill threads and connections
        host.close()
        sys.exit(0)


if __name__ == '__main__':
    main()
