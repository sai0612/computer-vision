import random

import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


def compute_F(pts1, pts2):
    # TO DO
    ransac_iterations = 1000
    threshold = 0.05
    max_inliers = 0
    i = 0
    F = None
    while i < ransac_iterations:
        A = np.zeros((8, 9))
        for j in range(8):
            ind = random.randint(0, len(pts1)-1)
            u = pts1[ind]
            v = pts2[ind]
            A[j] = np.asarray([u[0]*v[0], u[1]*v[0], v[0], u[0]*v[1], u[1]*v[1], v[1], u[0], u[1], 1])
        F_temp = null_space(A)
        F_temp = F_temp[:, 0]
        F_temp = F_temp.reshape(3, 3)
        #print(F_temp)
        u, D, vT = np.linalg.svd(F_temp)
        D[2] = 0
        F_temp = np.matmul(np.matmul(u, np.diag(D)), vT)
        inliers = 0
        for j in range(len(pts1)):
            u = np.asarray([pts1[j][0], pts1[j][1], 1]).reshape(3, 1)
            v = np.asarray([pts2[j][0], pts2[j][1], 1]).reshape(1, 3)
            #print(u, v, F_temp)
            error = np.abs(np.matmul(v, np.matmul(F_temp, u)))
            if error < threshold:
                inliers = inliers+1
        if inliers > max_inliers:
            max_inliers = inliers
            F = F_temp
        i = i+1
    print(max_inliers)
    return F


def triangulation(P1, P2, pts1, pts2):
    # TO DO
    pts3D = np.zeros((len(pts1), 3))
    for i in range(len(pts1)):
        u = np.asarray([pts1[i][0], pts1[i][1], 1])
        u_skew = np.asarray([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        v = np.asarray([pts2[i][0], pts2[i][1], 1])
        v_skew = np.asarray([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        p = np.concatenate((np.matmul(u_skew, P1), np.matmul(v_skew, P2)))
        u, D, vT = np.linalg.svd(p)
        v = np.transpose(vT)
        x_1 = v[:, 3]
        x_1 = x_1/x_1[3]
        x = x_1[:3]
        pts3D[i] = x
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    best_pose = 0
    max_valid = 0
    for i in range(len(Rs)):
        nvalid = 0
        r3T = Rs[i][2]
        C = Cs[i].reshape(-1)
        for j in range(len(pts3Ds[i])):
            X = pts3Ds[i][j]
            condition = np.matmul(r3T, np.subtract(X, C))
            if condition > 0:
                nvalid = nvalid+1
        if max_valid < nvalid:
            max_valid = nvalid
            best_pose = i
    R = Rs[best_pose]
    C = Cs[best_pose]
    pts3D = pts3Ds[best_pose]
    #print(best_pose, max_valid, R, C, pts3D)
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    C = C.reshape(-1)
    r1 = C/np.linalg.norm(C)
    temp = np.asarray([-C[1], C[0], 0])
    r2 = temp/np.linalg.norm(temp)
    r3 = np.cross(r1, r2)
    R_rect = np.vstack((np.transpose(r1), np.transpose(r2), np.transpose(r3)))
    H1 = np.matmul(K, np.matmul(R_rect, np.linalg.pinv(K)))
    H2 = np.matmul(K, np.matmul(R_rect, np.matmul(np.transpose(R), np.linalg.pinv(K))))
    return H1, H2


def dense_match(img1, img2, descriptors1, descriptors2):
    # TO DO
    disparity = np.zeros(img1.shape)
    for u in range(len(img1)):
        for v in range(len(img1[0])):
            if img1[u][v] != 0:
                disp_list = []
                for i in range(0, v+1):
                    disp = pow(np.linalg.norm(descriptors1[u][v]-descriptors2[u][i]), 2)
                    disp_list.append(disp)
                d = np.argmin(disp_list)
                disparity[u][v] = np.abs(d-v)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    disparity[disparity > 150] = 150
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 0: get correspondences between image pair
    data = np.load('./correspondence.npz')
    pts1, pts2 = data['pts1'], data['pts2']
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 2: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 5: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    data = np.load('./dsift_descriptor.npz')
    desp1, desp2 = data['descriptors1'], data['descriptors2']
    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    visualize_disparity_map(disparity)
