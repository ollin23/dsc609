from controls import *
import queue
import threading
import time


def main():
    debug = True
    tello, host, mtcnn = initialize(debug)
    width = 720
    height = 480

    if host is None or host is False:

        if host is not False:
            vid_thr = threading.Thread(target=video,
                                       args=(tello, host, mtcnn, width, height,))
            vid_thr.start()
            # rcv_video_thr = threading.Thread(target=video_rcv,
            #                                  args=(tello, host, ))
            #
            # video_thr = threading.Thread(target=video_display,
            #                              args=(mtcnn,))
            #
            # rcv_video_thr.start()
            # video_thr.start()
            #
            # rcv_video_thr.join()
            # video_thr.join()
        else:
            print("Could not establish connection. App terminated.")

        quit()
        sys.exit()

    else:
        response_thr = threading.Thread(target=rcv_response,
                                        args=(host, ))

        rcv_video_thr = threading.Thread(target=video_rcv,
                                         args=(tello, host,))

        video_thr = threading.Thread(target=video_display,
                                     args=(mtcnn,))

        response_thr.start()
        rcv_video_thr.start()
        video_thr.start()

        # response_thr.join()
        # rcv_video_thr.join()
        # video_thr.join()

        # launch(tello, host)
        # time.sleep(2)
        # land(tello, host)

        # kill threads and connections
        host.close()
    quit()
    sys.exit(0)

if __name__ == '__main__':
    main()
