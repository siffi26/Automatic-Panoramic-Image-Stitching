



import cv2
import numpy as np 
from copy import deepcopy
from numpy.linalg import inv
import getopt
import sys
import random
import scipy.ndimage as ndimage




def readImage(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img


# This draws matches and optionally a set of inliers in a different color
# Note: I lifted this drawing portion from stackoverflow and adjusted it to my needs because OpenCV 2.4.11 does not
# include the drawMatches function
def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

#
# Runs sift algorithm to find features
#
def findFeatures(img):
    print("Finding Features...")
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.png', img)

    return keypoints, descriptors

#
# Matches features given a list of keypoints, descriptors, and images
#
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    print("Matching Features...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    matchImg = drawMatches(img1,kp1,img2,kp2,matches)
    cv2.imwrite('Matches.png', matchImg)
    return matches


#
# Computers a homography from 4-correspondences
#
def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))
        print('len',len(maxInliers),'lencor',len(corr),thresh)
        L=len(corr)
        L=L*float(thresh)
        print('type',type(len(corr)))		
        if int(len(maxInliers)) > L:
            break
    return finalH, maxInliers
def get_stitched_image(img1, img2, M):

	# Get width and height of input images	
	w1,h1 = img1.shape[:2]
	w2,h2 = img2.shape[:2]

	# Get the canvas dimesions
	img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
	img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


	# Get relative perspective of second image
	img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

	# Resulting dimensions
	result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

	# Getting images together
	# Calculate dimensions of match points
	[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	
	# Create output array after affine transformation 
	transform_dist = [-x_min,-y_min]
	transform_array = np.array([[1, 0, transform_dist[0]], 
								[0, 1, transform_dist[1]], 
								[0,0,1]]) 

	# Warp images to get the resulting image
	result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
									(x_max-x_min, y_max-y_min))
	result_img[transform_dist[1]:w1+transform_dist[1], 
				transform_dist[0]:h1+transform_dist[0]] = img1

	# Return the result
	return result_img





def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img


img1 = cv2.imread('./test1/3.jpg',0)
img2 = cv2.imread('./test1/4.jpg',0)

img1n = cv2.imread('./test1/3.jpg',-1)
img2n = cv2.imread('./test1/4.jpg',-1)

img1 = ndimage.gaussian_filter(img1, sigma=(10, 10), order=0)
img2 = ndimage.gaussian_filter(img2, sigma=(10, 10), order=0)
img1n = equalize_histogram_color(img1n)
img2n = equalize_histogram_color(img2n)
estimation_thresh = 0.70



#find features and keypoints
correspondenceList = []
finalH = 0;
if img1 is not None and img2 is not None:
    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    #print("Found keypoints in " + img1name + ": " + str(len(kp1)))
    #print("Found keypoints in " + img2name + ": " + str(len(kp2)))
    keypoints = [kp1,kp2]
    matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
    for match in matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    corrs = np.matrix(correspondenceList)

    #run ransac algorithm
    finalH, inliers = ransac(corrs, estimation_thresh)
    
result_image = get_stitched_image(img2n, img1n, (finalH))
cv2.imwrite("result.jpg", result_image)






img1 = cv2.imread('./result.jpg',0)
img2 = cv2.imread('./test1/5.jpg',0)

img1n = cv2.imread('./result.jpg',-1)
img2n = cv2.imread('./test1/5.jpg',-1)

img1 = ndimage.gaussian_filter(img1, sigma=(10, 10), order=0)
img2 = ndimage.gaussian_filter(img2, sigma=(10, 10), order=0)
img1n = equalize_histogram_color(img1n)
img2n = equalize_histogram_color(img2n)
estimation_thresh = 0.70



#find features and keypoints
correspondenceList = []
finalH = 0;
if img1 is not None and img2 is not None:
    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    #print("Found keypoints in " + img1name + ": " + str(len(kp1)))
    #print("Found keypoints in " + img2name + ": " + str(len(kp2)))
    keypoints = [kp1,kp2]
    matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
    for match in matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    corrs = np.matrix(correspondenceList)

    #run ransac algorithm
    finalH, inliers = ransac(corrs, estimation_thresh)
    

result_image = get_stitched_image(img2n, img1n, (finalH))
cv2.imwrite("result2.jpg", result_image)




img1 = cv2.imread('./result2.jpg',0)
img2 = cv2.imread('./test1/6.jpg',0)

img1n = cv2.imread('./result2.jpg',-1)
img2n = cv2.imread('./test1/6.jpg',-1)

img1 = ndimage.gaussian_filter(img1, sigma=(10, 10), order=0)
img2 = ndimage.gaussian_filter(img2, sigma=(10, 10), order=0)
img1n = equalize_histogram_color(img1n)
img2n = equalize_histogram_color(img2n)
estimation_thresh = 0.70



#find features and keypoints
correspondenceList = []
finalH = 0;
if img1 is not None and img2 is not None:
    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    #print("Found keypoints in " + img1name + ": " + str(len(kp1)))
    #print("Found keypoints in " + img2name + ": " + str(len(kp2)))
    keypoints = [kp1,kp2]
    matches = matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
    for match in matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    corrs = np.matrix(correspondenceList)

    #run ransac algorithm
    finalH, inliers = ransac(corrs, estimation_thresh)
    

result_image = get_stitched_image(img2n, img1n, (finalH))
cv2.imwrite("result3.jpg", result_image)
#cv2.imshow ('Result', result_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



