import _init_paths
from datasets.factory import get_imdb
import pdb
import cv2

def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(
                im,
                '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15),
                cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 255),
                thickness=1)
    return im

imdb = get_imdb('voc_2007_trainval')
annotations = imdb._load_pascal_annotation(imdb._image_index[2020])
#pt = imdb._load_image_set_index()
image_path = imdb.image_path_from_index(imdb._image_index[2020])
im = cv2.imread(str(image_path))
#Get predicted bounded_boxes 
roidb = _load_selective_search_roidb(annotations)
pdb.set_trace()
