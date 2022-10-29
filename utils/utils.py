# ------------------------------------------------------------------------------
# Written by HengWeiBin (hengweibin1898@gmail.com)
# ------------------------------------------------------------------------------
import numpy as np
import scipy.ndimage as ndimage
from sklearn.cluster import AgglomerativeClustering
import cv2

def getBoundingBox(cluster):
    '''
    Get bounding box (top-left point and bot-right point)
    Input:
        Points
    Output:
        2 tuples with min(x, y), max(x, y)
    '''
    zipped_points = np.column_stack(cluster)
    max_x = zipped_points[0].max()
    min_x = zipped_points[0].min()
    max_y = zipped_points[1].max()
    min_y = zipped_points[1].min()
    
    return (min_x, min_y), (max_x, max_y)

def getClusterSubImages(img_origin, dmap_uint8):
    '''
    Find all local maxima points from density map (dmap_uint8),
    then cluster these points with distance,
    finally cut out sub-images and return.
    Input:
        img_origin - original color image with 3 channels
        dmap_uint8 - density map corresponding to img_origin with single channel
    Output:
        list with sub images of img_origin
    '''
    #==========================================================================
    #                  Find Local Maxima
    #==========================================================================
    neighborhood_size = 3
    threshold = 2
    
    data_max = ndimage.maximum_filter(dmap_uint8, neighborhood_size)
    maxima = (dmap_uint8 == data_max)
    data_min = ndimage.minimum_filter(dmap_uint8, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, _ = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    #==========================================================================
    #==========================================================================
        
    
    #==========================================================================
    #               Agglomerative Clustering and plot
    #==========================================================================
    
    points = np.array(list(zip(x, y))).astype(np.int32)
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=max(img_origin.shape), compute_full_tree=True).fit(points)
    clusters = {}
    for i, cluster in enumerate(clustering.labels_):
        if cluster in clusters.keys():
            clusters[cluster].append(points[i])
        else:
            clusters[cluster] = [points[i]]

    sub_images = []
    for cluster in clusters.values():
        (min_x, min_y), (max_x, max_y) = getBoundingBox(cluster)
        dilate_height = int((max_y - min_y) * 0.05)
        dilate_width = int((max_x - min_x) * 0.05)
        x1 = 0 if min_x - dilate_width < 0 else min_x - dilate_width
        x2 = img_origin.shape[1] if max_x + dilate_width > img_origin.shape[1] else max_x + dilate_width
        y1 = 0 if min_y - dilate_height < 0 else min_y - dilate_height
        y2 = img_origin.shape[0] if max_y + dilate_height > img_origin.shape[0] else max_y + dilate_height
        
        sub_images.append([img_origin[y1 : y2, x1: x2], (x1, y1), (x2, y2)])
    #==========================================================================
    #==========================================================================
    return sub_images