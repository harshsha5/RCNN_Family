import _init_paths
from datasets.factory import get_imdb
import pdb
import cv2
import visdom
import numpy as np

def visualize_bboxes(im, dets):
    #Visualize image and ground truth boxes
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)

    '''for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(
                im,
                '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 255),
                thickness=1)'''
    return im

imdb = get_imdb('voc_2007_trainval')
annotations = imdb._load_pascal_annotation(imdb._image_index[2020])
#pt = imdb._load_image_set_index()
image_path = imdb.image_path_from_index(imdb._image_index[2020])
im = cv2.imread(str(image_path))
im = visualize_bboxes(im,annotations['boxes'])
vis = visdom.Visdom(server='http://ec2-3-15-207-181.us-east-2.compute.amazonaws.com/',port='8097')
im = np.swapaxes(im,0,2)
im = np.swapaxes(im,1,2)
vis.image(im)

im_new = cv2.imread(str(image_path))
selective_search_db = imdb.selective_search_roidb()
required_selective_search_db = selective_search_db[2020]

#TODO: Get top 10 bboxes
top_10_boxes = required_selective_search_db['boxes'][:10]
#for i in range(10):
im_new = visualize_bboxes(im_new,top_10_boxes)
im_new = np.swapaxes(im_new,0,2)
im_new = np.swapaxes(im_new,1,2)
vis.image(im_new)



#Get predicted bounded_boxes 
#annotation_list = []
#annotation_list.append(annotations)
#roidb = imdb._load_selective_search_roidb(annotation_list)
#vis_detections(im,imdb._classes[annotations['gt_classes'][0]],roidb['boxes'])
#pdb.set_trace()
