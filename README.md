## Panaroma-stitching-using-Python
This work is completed as a part of Course assignment in 3D Computer Vision at IIT Gandhinagar

# Algorithm Followed to develop the Code:

1) The Code is initiated with two right most images from the images of the static scene.
2) The Homography is estimated using the RANSAC Algorithm.
3) Now the Origin of the right image is shifted as per Homography calculations and the whole right image is pasted at those co-ordinates in the blank canvas.
4) Now left image of both is pasted in the remaining blank canvas (defined with double the size of images)
5) The remaining blank canvas is removed and only image part is retrieved.
6) The above steps are repeated for the remaining images and the Instead of the right image now whole stitched part in previous steps is shifted in canvas using Homography between corresponding images.
7) Now the remaining blank part in canvas is removed and the saved in the same folder as of images.

# Executing the code:

Create the folder named 'img_set_1' and place the images to be stitched in the folder. We experimented with 8 images.

Step-1 : The root directory is to be maintained as seen and naming of the images and their folders to be followed in the same fashion.

Step-2 : The Code is fixed for the set of 8 images, for different no. of images alter the range mentioned in the panaroma_stitching.py (commented in the code)

Step-3 : The no. of inliers in the ransac.py script can be changed to get the better homography matrix.

Step-4 : The no. of iterations for the RANSAC Algorithm can also be changed as per user requirement. (Both the above are commented in the code itself)

Step-5 : To see the intermediate progress, uncomment the part mentioned in panaroma_stitching.py. (the intermediate images will be saved in the same directory in BGR format)

Step-6 : Currently the code is written to stitch 8 panaromic images, but one can modify as per ones needs.

# Drawbacks:

Good Horizontal shift is seen but poor vertical shift is observed (can be limited with good dataset with minimum hand glitches while clicking the images.

Intensity differenced led to many mismatches (Especially in image set 4).

Pixel to Pixel Transformation is leading to very poor results.

# References:

For Homography Estimation: Class Notes

For RANSAC Algorithm : Class Notes, https://engineering.purdue.edu/kak/courses-i-teach/ECE661.08/solution/hw4_s1.pdf
