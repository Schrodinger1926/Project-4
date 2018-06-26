import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

IMG_INPUT_DIR = 'test_images'
IMG_OUTPUT_DIR = 'test_images'

CALIB_DATA_DIR = 'camera_cal'
RND_DIR = 'rnd'

M, Minv = None, None

def get_img_points(img_abs_path):

    # Read image (RGB)
    img = mpimg.imread(img_abs_path)

    img_size = (720, 1280)

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # find chess corners in the given image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    return ret, corners


# gradient and magnitude thresholding
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    omapper = {
            'x': (1, 0),
            'y': (0, 1)
    }
    grad = cv2.Sobel(gray, cv2.CV_64F, omapper[orient][0], omapper[orient][1], ksize = sobel_kernel)
    grad = np.absolute(grad)
    sgrad = np.uint8(255*grad/np.max(grad))
    bgrad = np.zeros_like(sgrad)
    bgrad[(sgrad > thresh[0]) & (sgrad < thresh[1])] = 1

    return bgrad


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient in both x and y
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Resultant
    mag = np.sqrt(gradx**2 + grady**2)

    # Scale
    smag = np.uint8(255*mag/np.max(mag))

    # Get binary image
    bmag = np.zeros_like(smag)
    bmag[(smag > thresh[0]) & (smag < thresh[1])] = 1

    return bmag


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient in both x and y
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Get angle
    grad = np.arctan2(np.absolute(grady), np.absolute(gradx))
    bgrad =  np.zeros_like(grad)

    # Apply threshold
    bgrad[(grad > thresh[0]) & (grad < thresh[1])] = 1

    return bgrad


def s_select(img, thresh=(0, 255)):
    # Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Apply a threshold to the S channel
    img_s_channel = img_hls[:, :, 2]

    # Apply threshold
    binary_output = np.copy(img_s_channel) # placeholder line
    binary_output[(img_s_channel > thresh[0]) & (img_s_channel < thresh[1])] = 1
    return binary_output



def get_warped_image(image):
    global M
    global Minv

    # h, w = image.shape[0], image.shape[1]
    h, w = (300 ,420 )

    #offset = 150 # offset for dst points
    offset = 50 # offset for dst points
    """
    src1 = np.float32([[557, 477],
                       [370, 610],
                       [959, 626],
                       [727, 477]])
    """
    src1 = np.float32([[564, 466],
                       [301, 644],
                       [998, 644],
                       [711, 466]])

    dst = np.float32([[offset, offset],
                      [offset, h],
                      [w - offset, h],
                      [w - offset, offset]])

    M = cv2.getPerspectiveTransform(src1, dst)
    Minv = cv2.getPerspectiveTransform(dst, src1)
    # Supply custom shape
    #img_size = (420 , 420)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, (w, h), flags = cv2.INTER_LINEAR)

    bwarped = np.zeros_like(warped)
    bwarped[warped >= 0.90] = 1
    #bwarped[330:, :] = 0
    #print(bwarped.shape)
    return bwarped


# window settings
window_width = 50
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids


def r_channel_threshold(image, thresh = (0, 255)):
    r_channel = image[:, :, 2]
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel > thresh[0]) & (r_channel < thresh[1])] = 1
    return r_binary


def g_channel_threshold(image, thresh = (0, 255)):
    g_channel = image[:, :, 1]
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel > thresh[0]) & (g_channel < thresh[1])] = 1
    return g_binary


def l_select(img, thresh=(0, 255)):
    # Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Apply a threshold to the L channel
    img_l_channel = img_hls[:, :, 1]

    # Apply threshold
    binary_output = np.copy(img_l_channel) # placeholder line
    binary_output[(img_l_channel > thresh[0]) & (img_l_channel < thresh[1])] = 1
    return binary_output


def process_image(image):

    # undistort image
    image = cv2.undistort(image, mtx, dist, None, mtx)

    ksize = 3
    # Gradient thresholding
    gradx = abs_sobel_thresh(image, orient = 'x', sobel_kernel = ksize, thresh = (100, 255))

    # Magnitude thresholding
    mag_binary = mag_thresh(image, sobel_kernel = ksize, thresh = (100, 255))

    # Directional thresholding
    dir_binary = dir_threshold(image, sobel_kernel = 5, thresh = (0.7, 1.3))

    # saturation thresholding (yellow lines)
    s_binary = s_select(image, thresh=(170, 255))

    # Luminence thresholding (white lines)
    l_binary = l_select(image, thresh=(200, 255))

    # R-channel thresholding
    r_binary = r_channel_threshold(image, thresh=(200, 255))

    # G-channel thresholding
    g_binary = g_channel_threshold(image, thresh=(200, 255))

    # Combination step
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (l_binary == 1) & (r_binary)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
    combined[((r_binary == 1) & (g_binary == 1))| (( gradx == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    #plt.imshow(combined, cmap = 'gray')
    #plt.show()
    # Select area of interest

    # Perpective transform
    warped = get_warped_image(combined)
    #print(cv2.resize(warped_image, (10, 10)))

    # Get pixel count histogram
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0).astype(np.int32)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
          l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
          r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
          # Add graphic points from window mask here to total pixels found 
          l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
          r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(np.uint8(warpage), 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)


    # Display the final results
    """
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()
    """

    ploty = []
    leftx, rightx = [], []
    for level, centroid in enumerate(window_centroids):
        leftx.append(centroid[0])
        rightx.append(centroid[1])
        ploty.append(warped.shape[0] - level*window_height)

    ploty = np.array(ploty)
    leftx = np.array(leftx)
    rightx = np.array(rightx)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    #cv2.imshow('title', warped)
    #plt.imshow( output)
    #cv2.waitKey(0)
    #plt.imshow(result)
    #plt.show()
    #quit()
    return result


if __name__ == "__main__":

    # Calibration params
    img_size = (720, 1280)
    nx, ny = 9, 6
    objP = np.zeros((nx*ny, 3), np.float32)
    objP[:,:2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)

    # calibrate camera
    print("Calibrating Camera ..")
    obj_points = []
    img_points = []
    for filename in os.listdir(CALIB_DATA_DIR):
        ret, corners = get_img_points(img_abs_path = os.path.join(CALIB_DATA_DIR, filename))
        if ret:
            obj_points.append(objP)
            img_points.append(corners)

    # Get coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    if ret:
        print("Calibration: PASSED")
        """
        image = mpimg.imread(os.path.join(CALIB_DATA_DIR, 'calibration1.jpg'))
        
        image = cv2.undistort(image, mtx, dist, None, mtx)
        plt.imshow(image)
        plt.title('sanity_check') 
        plt.show()
        """

      
    # Process images
    for filename in os.listdir(IMG_INPUT_DIR):
        sys.stdout.write('\r Processing : {}'.format(filename))
        sys.stdout.flush()
        image = cv2.imread(os.path.join(IMG_INPUT_DIR, filename))

        # undistort image
        image = cv2.undistort(image, mtx, dist, None, mtx)
        process_image(image)

    video_filename = "project_video.mp4"
    clip1 = VideoFileClip(video_filename)

    white_output = "output_{}".format(video_filename)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)



