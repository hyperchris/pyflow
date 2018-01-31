## Python Dense Optical Flow

**Python** wrapper for Ce Liu's [C++ implementation](https://people.csail.mit.edu/celiu/OpticalFlow/) of Coarse2Fine Optical Flow. This is **super fast and accurate** optical flow method based on Coarse2Fine warping method from Thomas Brox. This python wrapper has minimal dependencies, and it also eliminates the need for C++ OpenCV library. For real time performance, one can additionally resize the images to a smaller size.

Run the following steps to download, install and demo the library:
  ```Shell
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```

## Process video
Process a video and save the extracted of images and frames into two folders. Command: ```python proc_video.py --vid_src [video_src] --of_dir [of_folder_path] --img_dir [frames_folder_path] ```

*The algorithm credits to [CVPR 2017 paper on Unsupervised Learning using unlabeled videos](http://cs.berkeley.edu/~pathak/unsupervised_video/).*