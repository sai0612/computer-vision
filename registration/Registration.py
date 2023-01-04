import cv2
import numpy as np
import numpy.linalg
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import random
import math

ransac_thr = 30
ransac_iter = 1000

def find_match(img1, img2):
    # To do
    sift = cv2.SIFT_create()
    kp1, ds1 = sift.detectAndCompute(img1, None)
    img1 = cv2.drawKeypoints(img1, kp1, img1)
    #cv2.imwrite('sift_keypoints1.jpg', img1)
    plt.imshow(img1)
    plt.show()
    sift = cv2.SIFT_create()
    kp2, ds2 = sift.detectAndCompute(img2, None)
    img2 = cv2.drawKeypoints(img2, kp2, img2)
    nearest_neigh = NearestNeighbors(n_neighbors=2).fit(ds2)
    dist, ind = nearest_neigh.kneighbors(ds1)
    #cv2.imwrite('sift_keypoints2.jpg', img2)
    plt.imshow(img2)
    plt.show()
    x1 = []
    x2 = []
    for i in range(len(dist)):
        ratio = dist[i][0]/dist[i][1]
        if ratio<0.7:
            x1.append([kp1[i].pt[0], kp1[i].pt[1]])
            x2.append([kp2[ind[i][0]].pt[0], kp2[ind[i][0]].pt[1]])
    #print(x2)
    #print(x1)
    return np.asarray(x1), np.asarray(x2)

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    max_inlier_count = 0
    A = np.empty((3, 3))
    for itr in range(ransac_iter):
        x1_rand_in = random.sample(range(0, len(x1)), 4)
        x1_rand = []
        x2_rand = []
        #print(x1_rand_in)
        #print(x2)
        for i in x1_rand_in:
            x1_rand.append(x1[i])
        for i in x1_rand_in:
            x2_rand.append(x2[i])
        a = [[x1_rand[0][0], x1_rand[0][1], 1, 0, 0, 0, -1*x1_rand[0][0]*x2_rand[0][0], -1*x1_rand[0][1]*x2_rand[0][0]],
             [0, 0, 0, x1_rand[0][0], x1_rand[0][1], 1, -1*x1_rand[0][0]*x2_rand[0][1], -1*x1_rand[0][1]*x2_rand[0][1]],
             [x1_rand[1][0], x1_rand[1][1], 1, 0, 0, 0, -1*x1_rand[1][0]*x2_rand[1][0], -1*x1_rand[1][1]*x2_rand[1][0]],
             [0, 0, 0, x1_rand[1][0], x1_rand[1][1], 1, -1*x1_rand[1][0]*x2_rand[1][1], -1*x1_rand[1][1]*x2_rand[1][1]],
             [x1_rand[2][0], x1_rand[2][1], 1, 0, 0, 0, -1*x1_rand[2][0]*x2_rand[2][0], -1*x1_rand[2][1]*x2_rand[2][0]],
             [0, 0, 0, x1_rand[2][0], x1_rand[2][1], 1, -1*x1_rand[2][0]*x2_rand[2][1], -1*x1_rand[2][1]*x2_rand[2][1]],
             [x1_rand[3][0], x1_rand[3][1], 1, 0, 0, 0, -1*x1_rand[3][0]*x2_rand[3][0], -1*x1_rand[3][1]*x2_rand[3][0]],
             [0, 0, 0, x1_rand[3][0], x1_rand[3][1], 1, -1*x1_rand[3][0]*x2_rand[3][1], -1*x1_rand[3][1]*x2_rand[3][1]]]
        a = np.array(a)
        b = np.array(x2_rand).reshape(-1)
        try:
            x = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            continue
        x = np.append(x, [1])
        x = x.reshape((3, 3))
        #print(x)
        x1_1 = np.transpose(np.concatenate((x1, np.ones((len(x1), 1))), axis=1))
        #print(x1_1)
        x_mul = np.matmul(x, x1_1)

        x2_1 = np.concatenate((x2, np.ones((len(x2), 1))), axis=1)
        x_mul1 = np.transpose(x_mul - np.transpose(x2_1))
        #print(x_mul1)
        inliers = 0
        for i in range(len(x_mul1)):
            distance = np.sqrt(np.sum(x_mul1[i]**2))
            #print(distance)
            if distance<ransac_thr:
                inliers = inliers+1
        #print(inliers)
        if inliers>max_inlier_count:
            max_inlier_count = inliers
            A = x

    #print(max_inlier_count)
    return A


def warp_image(img, A, output_size):
    # To do
    img_warped = np.zeros(output_size)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            x = np.array([j, i, 1])
            #print(x)
            point = np.matmul(A, x)
            #print(point)
            if point[1] < img.shape[0] and point[0] < img.shape[1]:
                img_warped[i][j] = img[int(point[1])][int(point[0])]

    return img_warped


def get_differential_filter():
    # To do
    filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    filter_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    pad_im = np.zeros((len(im)+2, len(im[0])+2))
    pad_im[1:len(im)+1, 1:len(im[0])+1] = im
    im_filtered = np.zeros((len(im), len(im[0])))
    for i in range(1, len(im)+1):
        for j in range(1, len(im[0])+1):
            t = 0
            for a in range(0, len(filter)):
                for b in range(0, len(filter[0])):
                    t = t+filter[a][b]*pad_im[i+a-1][j+b-1]
            im_filtered[i-1][j-1] = t
    return im_filtered


def align_image(template, target, A):
    # To do
    template = template.astype('float') / 255.0
    target = target.astype('float') / 255.0
    filter_x, filter_y = get_differential_filter()
    #template_dx = filter_image(template, filter_x)
    #template_dy = filter_image(template, filter_y)
    template_dx = cv2.filter2D(src=template, ddepth=-1, kernel=np.asarray(filter_x))
    template_dy = cv2.filter2D(src=template, ddepth=-1, kernel=np.asarray(filter_y))
    hessian = np.zeros((6, 6))
    steepest_descent_image = np.zeros((len(template), len(template[0]), 6, 1))
    for i in range(len(template)):
        for j in range(len(template[0])):
            jacob = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
            steepest_descent_image[i][j] = np.transpose(np.matmul(np.array([template_dx[i][j], template_dy[i][j]]), jacob)).reshape((6, 1))
            hessian = hessian + np.matmul(steepest_descent_image[i][j], np.transpose(steepest_descent_image[i][j]))
    hessian_inv = np.linalg.pinv(hessian)
    #print(steepest_descent_image)
    #print(hessian)
    A_refined = A.copy()
    epsilon = 0.1
    errors = [[], []]
    z=0
    while z<1000:
        warped_img = warp_image(target, A_refined, template.shape)
        error_img = warped_img - template
        F = np.zeros((6, 1))
        for i in range(len(template)):
            for j in range(len(template[0])):
                F = F + steepest_descent_image[i][j] * error_img[i][j]
        delta_p = np.matmul(hessian_inv, F)
        a_p = [[delta_p[0][0]+1, delta_p[1][0], delta_p[2][0]], [delta_p[3][0], delta_p[4][0]+1, delta_p[5][0]], [0, 0, 1]]
        A_refined = np.matmul(A_refined, np.linalg.pinv(a_p))
        delta_p_norm = np.linalg.norm(delta_p)

        error = np.linalg.norm(error_img)
        if z%20==0:
            print(delta_p_norm, z)
        errors[1].append(error)
        errors[0].append(z)
        if delta_p_norm < epsilon:
            break
        z=z+1

    return A_refined, np.asarray(errors)


def track_multi_frames(template, img_list):
    # To do
    A_list = np.zeros((len(img_list), 3, 3))
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    for i in range(0, len(img_list)):
        A_refined, errors = align_image(template, img_list[i], A)

        A = A_refined
        A_list[i]=A_refined
        img_warped = warp_image(img_list[i], A_refined, template.shape)
        template=img_warped
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    #print("visualize", x1)
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors[0], errors[1] * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)
    #print(x2)
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr, ransac_iter)
    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    plt.imshow(np.abs(img_warped-template), cmap='jet')
    plt.title('Error map')
    plt.show()

    A_refined, errors = align_image(img_warped, target_list[1], A)
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)


