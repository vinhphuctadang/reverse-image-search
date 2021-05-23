import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from annoy import AnnoyIndex
import random
import time
import matplotlib.pyplot as plt 

DATA_DIR = "../data/small-set/"
KNOWN_DATA_DIR = DATA_DIR + "known-data/"
QUERY_DATA_DIR = DATA_DIR + "query-data/"

def load_image(file_path):
    return image.load_img(file_path, target_size=(224, 224))

# load data within given "directory"
def load_data(directory):
    data = {}
    _, dirs, __ = next(os.walk(directory))
    for d in dirs:
        _, __, files = next(os.walk(directory + d))
        data[d] = []
        for file_name in files:
            file_path = directory + d + "/" + file_name 
            data[d].append(load_image(file_path))
    return data

# def extract_feature(model, src_images, chunk_size=10): # enable chunking to elimate processing time
#     img_input = preprocess_input(img_input)
#     feature_maps = model.predict(img_input).flatten()
#     for index in range(len(feature_maps)):
#         feature_maps[index] = feature_maps[index].flatten()
#     return feature_maps

# extract feature from image 
def extract_feature(model, src_image):
    img_input = image.img_to_array(src_image)
    img_input = np.expand_dims(img_input, axis=0) # == [x]
    img_input = preprocess_input(img_input)
    return model.predict(img_input)[0].flatten()

def show_hits(images, src_image, hits):
    fig=plt.figure(figsize=(9, 10))
    top_k = 8
    # subplot_index = 0
    for i in range(top_k+1):
        fig.add_subplot(3, 3, i+1)
        if i == 0:
            plt.title("query image")
            plt.imshow(src_image)
            continue
        plt.title("match %d" % i)
        plt.imshow(images[hits[i-1]])
    plt.show()

def load_model():
    model = MobileNetV2(
        include_top=False,
        weights="../model/mobilenetv2_notop",
    )
    return model
# known = load_data(KNOWN_DATA_DIR)
# query = load_data(QUERY_DATA_DIR)

# model = load_model()
# features = {}
# marker = time.time()
# count = 0
# # make get features:
# for kind in known:
#     features[kind] = alist = []
#     for i in range(len(known[kind])):
#         count += 1
#         alist.append(extract_feature(model, known[kind][i]))
# time_ellapsed = time.time() - marker
# print("Extract features of %d image(s) takes %.2fs" % (count, time_ellapsed), ", avg =", time_ellapsed / count, " seconds")

# indexer = AnnoyIndex(62720, "angular")
# images = []
# for kind in known:
#     images.extend(known[kind])

# feature_array = []
# for kind in known:
#     feature_array.extend(features[kind])

# marker = time.time()
# for i in range(len(feature_array)):
#     indexer.add_item(i, feature_array[i])
# indexer.build(6) # 10 trees
# indexer.save("test.ann")
# print("Indexing with annoy, dim: %d, done in %f second(s)" % (len(feature_array[0]), time.time()-marker))

# query_image_path = "../data/small-set/query-data/ant/image_0035.jpg"
# src_image = image.load_img(query_image_path, target_size=(224, 224))
# img_feature = extract_feature(model, src_image)

# marker = time.time()
# hits = indexer.get_nns_by_vector(img_feature, 8, search_k=3, include_distances=False)
# print("Searching done in %f second(s)" % (time.time()-marker))

# show_hits(images, src_image, hits)
