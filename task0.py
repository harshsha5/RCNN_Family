import _init_paths
from datasets.factory import get_imdb
import pdb

imdb = get_imdb('voc_2007_trainval')
annotations = imdb._load_pascal_annotation(imdb._image_index[2020])
#pt = imdb._load_image_set_index()
image_path = imdb.image_path_from_index(imdb._image_index[2020])
pdb.set_trace()
