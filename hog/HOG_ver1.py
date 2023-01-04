import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

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


def get_gradient(im_dx, im_dy):
    # To do
    grad_mag = np.zeros((len(im_dx), len(im_dx[0])))
    grad_angle = np.zeros((len(im_dx), len(im_dx[0])))
    for i in range(0, len(im_dx)):
        for j in range(0, len(im_dx[0])):
            grad_mag[i][j] = math.sqrt(im_dx[i][j]*im_dx[i][j]+im_dy[i][j]*im_dy[i][j])
            if im_dx[i][j]==0:
                grad_angle[i][j]=0
            else:
                grad_angle[i][j] = math.degrees(math.atan(im_dy[i][j]/im_dx[i][j]))
            if grad_angle[i][j]<0:
                grad_angle[i][j] = grad_angle[i][j]+180

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do

    ori_histo = np.zeros((len(grad_mag)//cell_size, len(grad_mag)//cell_size, 6))
    for i in range(0, len(grad_mag)-cell_size, cell_size):
        for j in range(0, len(grad_mag[0])-cell_size, cell_size):
            for a in range(0, cell_size):
                for b in range(0, cell_size):
                    if grad_angle[i+a][j+b]>=165 or grad_angle[i+a][j+b]<15:
                        ori_histo[i//cell_size][j//cell_size][0] += grad_mag[i+a][j+b]
                    elif grad_angle[i+a][j+b]>=15 and grad_angle[i+a][j+b]<45:
                        ori_histo[i // cell_size][j // cell_size][1] += grad_mag[i + a][j + b]
                    elif grad_angle[i+a][j+b]>=45 and grad_angle[i+a][j+b]<75:
                        ori_histo[i // cell_size][j // cell_size][2] += grad_mag[i + a][j + b]
                    elif grad_angle[i+a][j+b]>=75 and grad_angle[i+a][j+b]<105:
                        ori_histo[i // cell_size][j // cell_size][3] += grad_mag[i + a][j + b]
                    elif grad_angle[i+a][j+b]>=105 and grad_angle[i+a][j+b]<135:
                        ori_histo[i // cell_size][j // cell_size][4] += grad_mag[i + a][j + b]
                    elif grad_angle[i+a][j+b]>=135 and grad_angle[i+a][j+b]<165:
                        ori_histo[i // cell_size][j // cell_size][5] += grad_mag[i + a][j + b]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    ori_histo_normalized = np.zeros((len(ori_histo)-1, len(ori_histo[0])-1, len(ori_histo[0][0])*block_size*block_size))
    ep = 0.001
    for i in range(0, len(ori_histo)-block_size+1):
        for j in range(0, len(ori_histo[0])-block_size+1):
            t = 0
            hi_sum = 0
            for a in range(0, block_size):
                for b in range(0, block_size):

                    for c in range(0, len(ori_histo[0][0])):
                        hi_sum = hi_sum + ori_histo[i+a][j+b][c]*ori_histo[i+a][j+b][c]
            de = math.sqrt(hi_sum+ep*ep)
            for a in range(0, block_size):
                for b in range(0, block_size):
                    for c in range(0, len(ori_histo[0][0])):
                        h1 = ori_histo[i+a][j+b][c]/de
                        ori_histo_normalized[i][j][t*len(ori_histo[0][0])+c] = h1

                    t=t+1
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    filter_x, filter_y = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    #plt.imshow(im_dx)
    #plt.show()
    #plt.imshow(im_dy)
    #plt.show()
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    #plt.imshow(grad_mag)
    #plt.show()
    #plt.imshow(grad_angle, cmap='hot', interpolation='nearest')
    #plt.show()
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size=8)
    hog = get_block_descriptor(ori_histo, block_size=2)

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    bounding_boxes = []
    I_template_hog = extract_hog(I_template)
    I_template_hog = I_template_hog.flatten()
    mean =0
    temp_mag = 0
    for i in range(0, len(I_template_hog)):
        mean = mean+I_template_hog[i]

    mean = mean/len(I_template_hog)
    for i in range(0, len(I_template_hog)):
        I_template_hog[i] = I_template_hog[i]-mean
        temp_mag = temp_mag + I_template_hog[i] * I_template_hog[i]
    temp_mag = math.sqrt(temp_mag)


    for i in range(0, len(I_target)-len(I_template)+1, 6):
        for j in range(0, len(I_target[0])-len(I_template[0])+1, 6):
            I_target_hog = extract_hog(I_target[i:i+len(I_template), j:j+len(I_template[0])])
            I_target_hog = I_target_hog.flatten()
            mean = 0
            tar_mag = 0
            for a in range(0, len(I_target_hog)):
                mean = mean + I_target_hog[a]

            mean = mean / len(I_target_hog)
            dot =0
            for a in range(0, len(I_target_hog)):
                I_target_hog[a] = I_target_hog[a] - mean
                tar_mag = tar_mag + I_target_hog[a] * I_target_hog[a]
                dot = dot+I_template_hog[a]*I_target_hog[a]

            tar_mag = math.sqrt(tar_mag)
            dot = dot/(tar_mag*temp_mag)
            if dot>=0.4:
                bounding_boxes.append([j, i, dot])

    #visualize_face_detection(cv2.imread('target.png'), np.asarray(bounding_boxes), I_template.shape[0])
    bbs = bounding_boxes.copy()
    bb_ans = []
    while len(bbs)>0:
        maxi=0
        ma = -sys.maxsize - 1
        for i in range(0, len(bbs)):
            if bbs[i][2]>ma:
                ma = bbs[i][2]
                maxi = i
        bb_ans.append([bbs[maxi][0], bbs[maxi][1], bbs[maxi][2]])
        ax1 = bbs[maxi][0]
        ay1 = bbs[maxi][1]
        ax2 = bbs[maxi][0] + len(I_template[0])
        ay2 = bbs[maxi][1] + len(I_template)

        for i in range(0, len(bbs)):
            bx1 = bbs[i][0]
            by1 = bbs[i][1]
            bx2 = bbs[i][0] + len(I_template[0])
            by2 = bbs[i][1] + len(I_template)
            if not (ax1 == ax2 or ay1 == ay2 or bx1 == bx2 or by1 == by2) and not (ax2 <= bx1 or ay2 <= by1 or ax1 >= bx2 or ay1 >= by2):
                areaa = (ax2 - ax1) * (ay2 - ay1)
                areab = (bx2 - bx1) * (by2 - by1)
                w = min(ax2, bx2) - max(ax1, bx1)
                h = min(ay2, by2) - max(ay1, by1)
                inter_area = w*h
                iou = inter_area/(areab+areaa-inter_area)
                if iou>0.5:
                    bounding_boxes.remove([bbs[i][0], bbs[i][1], bbs[i][2]])
        bbs=bounding_boxes.copy()
    return bb_ans


def visualize_face_detection(I_target_b, bounding_boxes,box_size):
    I_target = np.asarray(I_target_b)
    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    cv2.imwrite("cv.png", fimg)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    plt.imshow(im, cmap='gray')
    plt.show()
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, np.asarray(bounding_boxes), I_template.shape[0])
    #this is visualization code.




