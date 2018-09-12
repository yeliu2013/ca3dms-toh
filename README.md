# RGBDMS
source code for context-aware RGBD mean-shift tracker with occlusion handling

To run the code, you should provide two folders named as "rgb" and "depth" which contains the rgb and depth images for tracking.
The images are named as 0.png, 1.png, 2.png...

you should also provide a "init.txt" which contains the initial boulding box.

the tracking results are saved as "box.txt", which is formated as:

left_top_corner_x,left_top_corner_y,right_bottom_corner_x,right_bottom_corner_y,occlusion_flag
