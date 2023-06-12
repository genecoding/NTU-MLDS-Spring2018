import re
import math
import pandas
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_images(images):
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    fig = plt.figure(figsize=(sqrtn*1.1, sqrtn*1.1))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.2, hspace=0.2)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.permute(1, 2, 0))
    return


def get_tag_dict(tag_file):
    df = pandas.read_csv(tag_file, header=None)  # no header in this homework datasets
    tags = df[1].to_numpy()
    
    if 'extra' in tag_file:
        # for extra_data/tags.csv
        
        # split from the space after 'hair'
        # tag_set = set([t for tag in tags for t in re.split(r'(?<=hair) ', tag)])
        
        # no split
        tag_set = set([tag for tag in tags])
    else:
        # for tags_clean.csv
        tag_set = set([t.split(':')[0].rstrip() for tag in tags for t in tag.split('\t') if t != ''])
    
    tag_dict = {tag: idx for idx, tag in enumerate(tag_set)}
    return tag_dict
    

# tag_dict = get_tag_dict(tag_file)    
tag_dict = {
    'aqua hair green eyes': 0,
    'blue hair yellow eyes': 1,
    'orange hair green eyes': 2,
    'blonde hair purple eyes': 3,
    'brown hair black eyes': 4,
    'blue hair blue eyes': 5,
    'brown hair yellow eyes': 6,
    'orange hair aqua eyes': 7,
    'green hair orange eyes': 8,
    'gray hair aqua eyes': 9,
    'blue hair brown eyes': 10,
    'red hair yellow eyes': 11,
    'pink hair aqua eyes': 12,
    'red hair brown eyes': 13,
    'red hair aqua eyes': 14,
    'orange hair red eyes': 15,
    'black hair yellow eyes': 16,
    'blonde hair black eyes': 17,
    'red hair purple eyes': 18,
    'orange hair pink eyes': 19,
    'blue hair purple eyes': 20,
    'gray hair orange eyes': 21,
    'purple hair yellow eyes': 22,
    'aqua hair purple eyes': 23,
    'pink hair brown eyes': 24,
    'gray hair blue eyes': 25,
    'gray hair red eyes': 26,
    'pink hair blue eyes': 27,
    'blue hair orange eyes': 28,
    'pink hair black eyes': 29,
    'blonde hair blue eyes': 30,
    'gray hair brown eyes': 31,
    'purple hair green eyes': 32,
    'purple hair orange eyes': 33,
    'green hair black eyes': 34,
    'blonde hair yellow eyes': 35,
    'green hair blue eyes': 36,
    'pink hair red eyes': 37,
    'red hair black eyes': 38,
    'white hair blue eyes': 39,
    'brown hair aqua eyes': 40,
    'brown hair blue eyes': 41,
    'brown hair pink eyes': 42,
    'purple hair brown eyes': 43,
    'gray hair green eyes': 44,
    'brown hair brown eyes': 45,
    'orange hair purple eyes': 46,
    'blue hair pink eyes': 47,
    'purple hair red eyes': 48,
    'green hair red eyes': 49,
    'gray hair black eyes': 50,
    'orange hair yellow eyes': 51,
    'white hair purple eyes': 52,
    'blonde hair red eyes': 53,
    'black hair brown eyes': 54,
    'white hair pink eyes': 55,
    'white hair orange eyes': 56,
    'pink hair green eyes': 57,
    'aqua hair aqua eyes': 58,
    'aqua hair pink eyes': 59,
    'brown hair orange eyes': 60,
    'red hair pink eyes': 61,
    'red hair green eyes': 62,
    'orange hair brown eyes': 63,
    'orange hair orange eyes': 64,
    'purple hair purple eyes': 65,
    'white hair yellow eyes': 66,
    'green hair yellow eyes': 67,
    'blue hair green eyes': 68,
    'pink hair yellow eyes': 69,
    'white hair brown eyes': 70,
    'black hair aqua eyes': 71,
    'blonde hair aqua eyes': 72,
    'pink hair pink eyes': 73,
    'gray hair pink eyes': 74,
    'green hair pink eyes': 75,
    'red hair blue eyes': 76,
    'pink hair orange eyes': 77,
    'blue hair aqua eyes': 78,
    'aqua hair blue eyes': 79,
    'black hair purple eyes': 80,
    'aqua hair yellow eyes': 81,
    'black hair pink eyes': 82,
    'aqua hair brown eyes': 83,
    'green hair aqua eyes': 84,
    'gray hair purple eyes': 85,
    'aqua hair orange eyes': 86,
    'blonde hair orange eyes': 87,
    'purple hair blue eyes': 88,
    'black hair black eyes': 89,
    'green hair purple eyes': 90,
    'purple hair aqua eyes': 91,
    'aqua hair red eyes': 92,
    'orange hair black eyes': 93,
    'brown hair green eyes': 94,
    'white hair aqua eyes': 95,
    'blue hair black eyes': 96,
    'blonde hair pink eyes': 97,
    'brown hair red eyes': 98,
    'blue hair red eyes': 99,
    'black hair orange eyes': 100,
    'green hair brown eyes': 101,
    'red hair red eyes': 102,
    'white hair black eyes': 103,
    'white hair red eyes': 104,
    'blonde hair brown eyes': 105,
    'black hair blue eyes': 106,
    'black hair red eyes': 107,
    'gray hair yellow eyes': 108,
    'brown hair purple eyes': 109,
    'green hair green eyes': 110,
    'white hair green eyes': 111,
    'blonde hair green eyes': 112,
    'aqua hair black eyes': 113,
    'orange hair blue eyes': 114,
    'pink hair purple eyes': 115,
    'black hair green eyes': 116,
    'red hair orange eyes': 117,
    'purple hair pink eyes': 118
}