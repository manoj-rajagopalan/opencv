import cv2 as cv
import numpy as np
import sys

def center_and_normalize(v):
    v_mean = np.mean(v.astype(np.float64))
    v_result = (v - v_mean)
    v_norm = np.linalg.norm(v_result)
    v_result = np.zeros_like(v_result) if (v_norm == 0.0) else (v_result / v_norm)
    return v_result

class Point:
    def __init__(self, x,y):
        self.x = x
        self.y = y
# /Point

class Rect:
    def __init__(self, lower_left, upper_right):
        '''Both are Point objects'''
        self.upper_left = upper_left
        self.lower_right = lower_right    
# /Rect

def compute_disparities_using_ncc(img1, img2, semi_block_size, D):
    '''
        semi_block_size: window is (2 * semi_block_size + 1) centered at each pixel
        D : max absolute disparity
    '''
    assert img1.shape == img2.shape
    assert 0 < D and D <= 127
    disparities = np.full((img1.shape[0], img1.shape[1]),
                           0, # background fill value
                           dtype = np.int8)
    I, J, _ = img1.shape
    for i in range(I):
        print('@i =', i, end = ' ', flush=True)
        for j in range(J):
            # print('j = ', j, end=' ')
            sub_img1 = img1[max(0,i-semi_block_size):1+min(I-1,i+semi_block_size),
                            max(0,j-semi_block_size):1+min(J-1,j+semi_block_size)]
            if len(sub_img1) < 10:
                continue
            sub_img1_centered_and_normalized = center_and_normalize(sub_img1)
            max_disp = 0
            max_ncc_value = -np.Inf
            for d in np.arange(-D,D+1):
                ii = i + d
                sub_img2 = img2[max(0,ii-semi_block_size):1+min(I-1,ii+semi_block_size),
                                max(0,j-semi_block_size):1+min(J-1,j+semi_block_size)]
                if len(sub_img2) < 10 or sub_img2.shape != sub_img1.shape:
                    continue
                sub_img2_centered_and_normalized = center_and_normalize(sub_img2)
                ncc_value = np.inner(sub_img1_centered_and_normalized.flatten(),
                                     sub_img2_centered_and_normalized.flatten())
                if ncc_value > max_ncc_value:
                    max_ncc_value = ncc_value
                    max_disp = d
            # /for d

            disparities[i,j] = max_disp
        # /for j
    # /for i
    print()

    # check and scale
    print('Num disparities > 0 = ', np.sum(disparities > 0))
    print('Num disparities < 0 = ', np.sum(disparities < 0))
    print('Max, min disparity values = ', np.max(disparities), np.min(disparities))
    disparities = -np.max(disparities, 0)
    disparities = (disparities.astype(np.float64) / np.max(disparities)).astype(np.uint8) * 255
    return disparities

def main():
    assert len(sys.argv) == 3, "Usage: {}  <left image>  <right image>".format(sys.argv[0])
    l_img = cv.imread(sys.argv[1])
    r_img = cv.imread(sys.argv[2])
    assert l_img.shape == r_img.shape
    print('image shape = ', l_img.shape)

    print('Doing SGBM ...')
    stereo_sgbm = cv.StereoSGBM_create(minDisparity=-32, numDisparities=64, blockSize=11)
    disparities = stereo_sgbm.compute(l_img, r_img)
    cv.imwrite('disparity-sgbm.png', disparities)

    print('Doing BM ...')
    stereo_bm = cv.StereoBM_create(numDisparities=64, blockSize=11)
    disparities = stereo_bm.compute(np.mean(l_img, axis=2, dtype=np.uint8), np.mean(r_img, axis=2, dtype=np.uint8))
    cv.imwrite('disparity-bm.png', disparities)

    print('Doing brute force NCC ...')
    disparities = compute_disparities_using_ncc(l_img, r_img, 5, 32)
    cv.imwrite('disparity-ncc.png', disparities)

# /main()

if __name__ == "__main__":
    main()

