import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img):
    # To do
    stride = 8
    size = 8
    sift = cv2.SIFT_create()
    key_points = [cv2.KeyPoint(x+size/2, y+size/2, size) for x in range(0, len(img[0]), stride)
                  for y in range(0, len(img), stride)]
    kp, dense_feature = sift.compute(img, key_points)
    return dense_feature


def get_tiny_image(img, output_size):
    # To do
    feature = np.zeros(output_size)
    #print(len(img), len(img[0]))
    h = len(img)/output_size[0]
    w = len(img[0])/output_size[1]
    x = 0
    y = 0
    h = int(h)
    w = int(w)
    for i in range(0, h*output_size[0], h):
        for j in range(0, w*output_size[1], w):
            #print(i,j)
            feature[x][y] = np.average(img[i: i+h, j: j+w])
            y = y+1
        y=0
        x = x+1
    feature = feature-np.mean(feature)
    norm = np.linalg.norm(feature)
    feature = feature/norm
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    neigh = NearestNeighbors(n_neighbors=k)
    neigh = neigh.fit(feature_train)
    distances, indices = neigh.kneighbors(feature_test)
    label_test_pred = []
    for neighbours in indices:
        pred = []
        for neighbour in neighbours:
            pred.append(label_train[neighbour])
        pred_max = pred[0]
        pred_max_c = 0
        for i in pred:
            freq = pred.count(i)
            if freq > pred_max_c:
                pred_max_c = freq
                pred_max = i
        label_test_pred.append(pred_max)
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    feature_train = []
    feature_test = []
    img_tiny_h = 16
    img_tiny_w = 16
    for img_train_path in img_train_list:
        img_train = cv2.imread(img_train_path, cv2.IMREAD_GRAYSCALE)
        tiny_img_train = get_tiny_image(img_train, [img_tiny_h, img_tiny_w])
        feature_train.append(tiny_img_train.flatten())
    for img_test_path in img_test_list:
        img_test = cv2.imread(img_test_path, cv2.IMREAD_GRAYSCALE)
        tiny_img_test = get_tiny_image(img_test, [img_tiny_h, img_tiny_w])
        feature_test.append(tiny_img_test.flatten())
    k = 8
    label_test_pred = predict_knn(feature_train, label_train_list, feature_test, k)
    confusion = np.zeros([15, 15])
    accuracy = 0
    for i in range(len(label_test_list)):
        if label_test_pred[i] == label_test_list[i]:
            accuracy = accuracy+1
        pred_test_in = label_classes.index(label_test_pred[i])
        act_test_in = label_classes.index(label_test_list[i])
        confusion[act_test_in][pred_test_in] = confusion[act_test_in][pred_test_in] + 1
    print(accuracy)
    accuracy = accuracy/len(label_test_list)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    print(confusion)

    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dict_size):
    # To do
    dense_feature_list = np.vstack(dense_feature_list)
    kmeans = KMeans(n_clusters=dict_size)
    vocab = kmeans.fit(dense_feature_list)
    vocab = vocab.cluster_centers_
    np.savetxt('vocab.txt', vocab)
    return vocab


def compute_bow(feature, vocab):
    # To do
    neigh = NearestNeighbors(n_neighbors=1)
    neigh = neigh.fit(vocab)
    distances, indices = neigh.kneighbors(feature)
    bow_feature = np.zeros(len(vocab))
    for i in range(len(indices)):
        bow_feature[indices[i]]=bow_feature[indices[i]]+1
    norm = np.linalg.norm(bow_feature)
    bow_feature = bow_feature/norm
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    feature_train = []
    feature_test = []
    dense_feature_list = []
    for img_train_path in img_train_list:
        img_train = cv2.imread(img_train_path, cv2.IMREAD_GRAYSCALE)
        dense_feature = compute_dsift(img_train)
        dense_feature_list.append(dense_feature)

    dict_size = 50
    vocab = build_visual_dictionary(dense_feature_list, dict_size)
    for img_train_path in img_train_list:
        img_train = cv2.imread(img_train_path, cv2.IMREAD_GRAYSCALE)
        dense_feature = compute_dsift(img_train)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_train.append(bow_feature)

    for img_test_path in img_test_list:
        img_test = cv2.imread(img_test_path, cv2.IMREAD_GRAYSCALE)
        dense_feature = compute_dsift(img_test)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_test.append(bow_feature)
    k = 8
    label_test_pred = predict_knn(feature_train, label_train_list, feature_test, k)
    confusion = np.zeros([15, 15])
    accuracy = 0
    for i in range(len(label_test_list)):
        if label_test_pred[i] == label_test_list[i]:
            accuracy = accuracy + 1
        pred_test_in = label_classes.index(label_test_pred[i])
        act_test_in = label_classes.index(label_test_list[i])
        confusion[act_test_in][pred_test_in] = confusion[act_test_in][pred_test_in] + 1
    print(accuracy)
    accuracy = accuracy / len(label_test_list)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    svm_classifier = LinearSVC()
    #models = {}
    predictions = np.zeros([len(n_classes), len(feature_test)])
    for i in range(len(n_classes)):
        binary_label_train = []
        for label in label_train:
            if label == n_classes[i]:
                binary_label_train.append(1)
            else:
                binary_label_train.append(0)
        model = svm_classifier.fit(feature_train, binary_label_train)
        pred = model.decision_function(feature_test)
        print(pred)
        predictions[i] = pred

    predictions = np.transpose(predictions)
    label_test_pred = []
    label_test = np.argmax(predictions, axis=1)
    print(label_test)
    for i in range(len(label_test)):
        label_test_pred.append(n_classes[label_test[i]])
    print(label_test_pred)
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    feature_train = []
    feature_test = []
    dense_feature_list = []
    for img_train_path in img_train_list:
        img_train = cv2.imread(img_train_path, cv2.IMREAD_GRAYSCALE)
        dense_feature = compute_dsift(img_train)
        dense_feature_list.append(dense_feature)

    dict_size = 50
    path_to_vocab = './vocab.txt'
    if os.path.isfile(path_to_vocab):
        vocab = np.loadtxt(path_to_vocab)
    else:
        vocab = build_visual_dictionary(dense_feature_list, dict_size)
    for img_train_path in img_train_list:
        img_train = cv2.imread(img_train_path, cv2.IMREAD_GRAYSCALE)
        dense_feature = compute_dsift(img_train)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_train.append(bow_feature)

    for img_test_path in img_test_list:
        img_test = cv2.imread(img_test_path, cv2.IMREAD_GRAYSCALE)
        dense_feature = compute_dsift(img_test)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_test.append(bow_feature)
    label_test_pred = predict_svm(feature_train, label_train_list, feature_test, label_classes)
    confusion = np.zeros([15, 15])
    accuracy = 0
    for i in range(len(label_test_list)):
        if label_test_pred[i] == label_test_list[i]:
            accuracy = accuracy + 1
        pred_test_in = label_classes.index(label_test_pred[i])
        act_test_in = label_classes.index(label_test_list[i])
        confusion[act_test_in][pred_test_in] = confusion[act_test_in][pred_test_in] + 1
    print(accuracy)
    accuracy = accuracy / len(label_test_list)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




