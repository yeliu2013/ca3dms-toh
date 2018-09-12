# RGBDMS
source code for context-aware RGBD mean-shift tracker with occlusion handling

dependency: OpenCV (2.4.9 other versions should also be OK)

To run the code, you should provide two folders named as "rgb" and "depth" which contains the rgb and depth images for tracking.
The images are named as 0.png, 1.png, 2.png...

you should also provide a "init.txt" which contains the initial boulding box.

"param.txt" contains the camera parameters: fx, fy, cx, cy

the tracking results are saved as "box.txt", which is formated as:

left_top_corner_x,left_top_corner_y,right_bottom_corner_x,right_bottom_corner_y,occlusion_flag

This code is tested with videos capture with OpenNI for Kinect 1.0, it should also be suitable for Kinect 2.0 or other types of RGBD 
sensors with camera parameters in "param.txt" modified, but it has not been tested.

