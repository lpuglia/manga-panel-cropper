import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from rectangle import Rectangle
import random

debug = False
colors = [[0,255,0], [255,0,0], [0,0,255], [0,255,255], [255,255,0], [255,0,255], [0,255,128], [0,128,255], [0,255,255], [128,255,0], [255,128,0], [128,0,255], [255,0,128]]

def get_contours(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = None
    contours = None
    contrast = 10
    # gray = cv2.medianBlur(gray,5)
    # plot_contours(gray,[])
    # gray = gray * (contrast/127+1) - contrast 
    ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if debug:
        print("DEBUG get_contours")
        plot_contours(img, contours)
    return contours


def build_crops_edge_graph(crops, image_shapes):
    crops_after = []
    crops_before = []
    for i,c in enumerate(crops):
        after = []
        before = []
        
        for j,d in enumerate(crops):
            d_before_area = c.intersection_before_area(d)
            c_before_area = d.intersection_before_area(c)
            if d_before_area/d.area > c_before_area/c.area:
                before.append(j)
            
        crops_before.append(before)
    return crops_before

def sort_edge_graph(crops, edge_graph):
    sorted_list = []
    if edge_graph==[]: return []
    edges = deepcopy(edge_graph)
    # print(edges)
    while True:
        # print(edges)
        try:
            idx = edges.index([])
        except:
            valid_edge = {x for l in [e for e in edges if e is not None] for x in l}
            # print(valid_edge)
            idx = sorted(valid_edge, key=lambda x: crops[x].center_x, reverse=True)[0] #righter crop

        sorted_list.append(idx)
        edges[idx] = None
        if not any(edges): break
        for i in range(len(edges)):
            if edges[i] and edges[i].count(idx)>0:
                jdx = edges[i].index(idx)
                edges[i].pop(jdx)
    return_crops = []
    for idx in sorted_list:
        return_crops.append(crops[idx])
    return return_crops

def get_crops(img, contours):
    total_area = img.shape[0]*img.shape[1]
    
    if img.shape[0]/img.shape[1]<1.0:
        minimal_hull_area = 0.02
    else:
        minimal_hull_area = 0.04

    
    hulls = [cv2.convexHull(c) for c in contours]
    
    regular_hulls = []
    for c,h in zip(contours, hulls):
        c_area = cv2.contourArea(c)
        h_area = cv2.contourArea(cv2.approxPolyDP(h,0.001,True))
        # print(c_area,h_area)
        if h_area/total_area>minimal_hull_area or c_area*1.5>h_area:
            regular_hulls.append(h)

    # regular_hulls = [h for c,h in zip(contours, hulls) if cv2.contourArea(c)*2.>cv2.contourArea(cv2.approxPolyDP(h,0.001,True) and cv2.contourArea(c))]
    
    crops = []
    for hull,contour in zip(regular_hulls,contours):
        r = cv2.boundingRect(hull)
        crops.append(Rectangle(*r))

    if debug:
        print("DEBUG hulls")
        plot_contours(img, hulls)
        print("DEBUG regular hulls")
        plot_contours(img, regular_hulls)
        print("DEBUG crops")
        plot_frames(img, crops)

    return crops

def remove_small_crops(img, crops):
    total_area = img.shape[0]*img.shape[1]

    if img.shape[0]/img.shape[1]<1.0:
        minimal_crop_area = 0.01
    else:
        minimal_crop_area = 0.04

    # for c in crops:
    #     print(c.area/total_area)
    big_crops = [c for c in crops if c.area/total_area>minimal_crop_area] # bigger than 10% of total area

    if debug:
        print("DEBUG big crops")
        plot_frames(img, big_crops)

    return big_crops

def merge_overlapping_crops(crops, percentage=0.5):
    crops_pair = [(a, b) for idx, a in enumerate(crops) for b in crops[idx + 1:]]
    to_remove = set()
    for c1,c2 in crops_pair:
        if (c1+c2).area == c2.area: to_remove.add(c1)
        elif (c1+c2).area == c1.area: to_remove.add(c2)

    for c in to_remove:
        del crops[crops.index(c)]

    while True:
        crops_pair = [(a, b) for idx, a in enumerate(crops) for b in crops[idx + 1:]]
        to_merge = []
        for c1,c2 in crops_pair:
            try:
                intersect_area = (c1-c2).area
            except:
                intersect_area = 0

            if intersect_area>c1.area*percentage or intersect_area>c2.area*percentage:
                to_merge.append([c1,c2])
        
        if not to_merge: break

        new_crops = set()
        to_delete = set()
        for i,j in to_merge:
            new_crops.add(i+j)
            to_delete.add(i)
            to_delete.add(j)

        for t in to_delete:
            del crops[crops.index(t)]
        
        crops.extend(new_crops)
    if debug:
        print('merge_overlapping_crops')
        print(crops)
    return crops

def plot_contours(img, contours):
    z = np.copy(img)
    for contour in contours:
        cv2.drawContours(z, [contour],0,random.sample(colors, 1)[0],2)
    plt.figure(figsize=(16,16))
    plt.imshow(z, cmap='gray')
    plt.show()

def plot_frames(img, crops, meta=False):
    z = np.copy(img)
    for i,c in enumerate(crops):
        cv2.rectangle(z,(c.x,c.y),(c.w,c.z),random.sample(colors, 1)[0],2)
        if meta:
            cv2.putText(z, f'{i}', (c.center_x, c.center_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,0), 2)
            cv2.circle(z, (c.center_x, c.center_y), 5, (255, 0, 255), 5)
    plt.figure(figsize=(16,16))
    plt.imshow(z, cmap='gray')
    plt.show()

def plot_crops(img, crops):
    if crops:
        for c in crops:
            cropped_image = img[c.y:c.z, c.x:c.w, :]
            plt.figure(figsize=(8,8))
            plt.imshow(cropped_image)
            plt.show()
    else:
        plt.figure(figsize=(16,16))
        plt.imshow(img)
        plt.show()
        
        
def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img