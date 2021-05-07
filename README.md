# dsc609
This machine learning project will implement facial recognition and tracking
on a drone. If time permits, object tracking and commands will be implemented
as well.


<b>NOTES:</b><br>
7 May 2021:
- added minor functionality:  use keyboard to toggle filters
- allows for keyboard control of drone
- Moving Forward: keyboard controls of UAV, get better or build UAV

19 Apr 2021
- tactical pause to do another project for a couple weeks

18 Apr 2021
- Finally functional on workstation
- Ubuntu 20.04 LTS, Python 3.8, OpenCV 4.3, CUDA 10.2, CUDNN 8.1, PyTorch 1.8.1, NumPy 1.20.2

17 Apr 2021
- after 3 OS crashes, finally installed OpenCV with GPU acceleration on Linux. Pro-gamer tip: DON'T USE THE NEW HOTNESS, stick with slightly older versions
- refactoring with _controls.py_ for the controls

13 Apr 2021
- using _detector_ as a utility file for face detection
- tested; works
- next: recognition engine. This requires labelled video images of objects to identify (which will not be uploaded because privacy).

12 April 2021
- Committed first files

10 April 2021
- Selected PyTorch for the ML framework.
- Selected MTCNN for facial detection. Considered Haar Cascade, but implementation for GPU is not feasible from the ground up given limited timeframe.

9 April 2021
- Researching different methods for the pipeline.
- Facial recognition for a drone implies:
  - drone with camera
  - GPU on static system
  - language and libraries to use GPU
  - facial detection algorithm
  - facial recognition algorithm
  - (if time) facial/object tracking
- Drone chosen. Working through the theoretical and pragmatic
- challenges of different detection algorithms.

7 April 2021
- Wrote main script as tester to ensure drone viablility.
