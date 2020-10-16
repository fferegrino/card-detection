#!/usr/bin/env python
# coding: utf-8

# # Yu-Gi-Oh! Card Recogniser Scene Generation

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pathlib import Path
from skimage import io
import numpy as np
import math
import random
from yugioh.card_dataset import CardDataset

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from shapely.geometry import Polygon, Point
from shapely.geometry.multipolygon import MultiPolygon


# In[2]:


np.random.seed(42)
random.seed(42)


# ## Download the datasets  
# 
#  - [Yu-Gi-Oh! Cards](https://www.kaggle.com/ioexception/yugioh-cards) from Kaggle.  
#     - `kaggle datasets download -d ioexception/yugioh-cards -p input/yugioh-cards --unzip`
#  - [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html) from https://www.robots.ox.ac.uk/
#     - `wget -O input/dtd-r1.0.1.tar.gz https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz`
#     - `tar xvfz input/dtd-r1.0.1.tar.gz --directory input`
#     - `rm input/dtd-r1.0.1.tar.gz`

# ## Read the cards dataset

# In[3]:


card_dataset = CardDataset("input/yugioh-cards/")


# In[4]:


card = random.sample(card_dataset,1)[0]
card


# In[5]:


plt.imshow(card.image())


# In[6]:


card_type_map = {
    'Effect Monster': "monster",
    'Flip Effect Monster': "monster",
    'Fusion Monster': "fusion_monster",
    'Gemini Monster': "monster",
    'Link Monster': "link_monster",
    'Normal Monster': "monster",
    'Normal Tuner Monster': "monster",
    'Pendulum Effect Fusion Monster': "pendulum_monster",
    'Pendulum Effect Monster': "pendulum_monster",
    'Pendulum Flip Effect Monster': "pendulum_monster",
    'Pendulum Normal Monster': "pendulum_monster",
    'Pendulum Tuner Effect Monster': "pendulum_monster",
    'Ritual Effect Monster': "ritual_monster",
    'Ritual Monster': "ritual_monster",
    #'Skill Card': "skill",
    'Spell Card': "spell",
    'Spirit Monster': "monster",
    'Synchro Monster': "synchro_monster",
    'Synchro Pendulum Effect Monster': "pendulum_monster",
    'Synchro Tuner Monster': "synchro_monster",
    #'Token': "token",
    'Toon Monster': "monster",
    'Trap Card': "trap",
    'Tuner Monster': "monster",
    'Union Effect Monster': "monster",
    'XYZ Monster': "xyz_monster",
    'XYZ Pendulum Effect Monster': "pendulum_monster"
}

all_available_cards, classes = [], []
for card in card_dataset:
    if card.type not in card_type_map:
        continue
    card.type = card_type_map[card.type]
    classes.append(card.type)
    all_available_cards.append(card)
    


# In[7]:


import imblearn
from collections import Counter

original_values = Counter(classes)

def oversample(value):
    return {
        class_:max(value, 600) for class_, value in original_values.items()
    }
def undersample(value):
    return {
        class_:min(value, 3500) for class_, value in original_values.items()
    }


rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=undersample)
ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy=oversample)
all_available_cards, b = ros.fit_resample(np.array(all_available_cards).reshape(len(all_available_cards), -1), classes)
all_available_cards, b = rus.fit_resample(all_available_cards, b)
all_available_cards = all_available_cards.squeeze().tolist()
most_common = Counter(b).most_common()
print(most_common)
print(len(most_common))


# ## Read the backgrounds

# In[8]:


dtd_dir="input/dtd/"
background_paths = list(Path(dtd_dir).glob("**/*.jpg"))
bg_images=[]
for image_path in random.sample(background_paths, 5000):
    bg_images.append(io.imread(image_path))


# In[9]:


SCALE_CARDS = 0.4

image_transformations = iaa.Sequential([
    iaa.Multiply((0.8, 1.3)),
    iaa.Sometimes(0.5, 
                  iaa.GammaContrast((0.6, 1.5), per_channel=True),
                  iaa.Multiply((0.5, 1.5),per_channel=0.5), 
                 ),
    iaa.AddElementwise((-20, 20), per_channel=0.5),
])

spatial_transformations = iaa.Sequential([
    
    iaa.Affine(
        scale=SCALE_CARDS,
        translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
        rotate=(-90, 90),
    ),
])

deep_fry_whole = iaa.Sometimes(0.5,iaa.Sequential([
    iaa.AddElementwise((-10, 10), per_channel=0.5),
      iaa.Multiply((0.7, 1),per_channel=0.5), 
]))

# plt.imshow(image_transformations(image=all_available_cards[100].image()))


# In[10]:


class Scenes:
    
    def __init__(self, backgrounds, scene_width, scene_height, card_shape = (614, 421, 3), grid = (4, 7), card_padding=20):
        self.backgrounds = backgrounds
        self.scene_width = scene_width
        self.scene_height = scene_height
        
        self.card_key_points = ia.KeypointsOnImage([
            ia.Keypoint(x=0, y= 0),
            ia.Keypoint(x=0, y=card_shape[0]),   
            ia.Keypoint(x=card_shape[1], y=card_shape[0]),
            ia.Keypoint(x=card_shape[1], y= 0)
        ], shape=card_shape)
        
        self.scene_poly = Polygon(((0., 0.), (0., scene_height), (scene_width, scene_height), (scene_width, 0.), (0., 0.)))
        
        self.card_padding =  card_padding
        
        aux_w = scene_width - card_shape[1]
        self.card_w_padding = [aux_w // 2] * 2
        self.card_w_padding[1] += aux_w % 2
        
        aux_w = scene_height - card_shape[0]
        self.card_h_padding = [aux_w // 2] * 2
        self.card_h_padding[1] += aux_w % 2
        
        # Generate grid
        hs = np.linspace(100, scene_height-100, grid[0]+1).astype(int)
        vs = np.linspace(100, scene_width-100, grid[1]+1).astype(int)
        center_x = (vs[1:] + vs[:-1]) // 2 - scene_width // 2
        center_y = (hs[1:] + hs[:-1]) // 2 - scene_height // 2
        self.grid = list(np.transpose([np.tile(center_x, len(center_y)), np.repeat(center_y, len(center_x))]) )
        
                
    def _generate_random_background(self):
        selected_bg = random.choice(self.backgrounds)
        bg_height, bg_width, _ = selected_bg.shape
        repeat_w = math.ceil(self.scene_width/bg_width)
        repeat_h = math.ceil(self.scene_height/bg_height)
        background = np.repeat(selected_bg, repeats=repeat_w, axis=1)
        background = np.repeat(background, repeats=repeat_h, axis=0)
        return background[:self.scene_height,:self.scene_width,:]
    
    @staticmethod
    def kps_to_polygon(kps):
        pts=[(kp.x,kp.y) for kp in kps]
        return Polygon(pts)
    
    def kps_to_bounding(self, kps):
        """
            Determine imgaug bounding box from imgaug keypoints
        """
        extend=3 # To make the bounding box a little bit bigger
        kpsx=[kp.x for kp in kps.keypoints]
        minx=max(0,int(min(kpsx)-extend))
        maxx=min(self.scene_width,int(max(kpsx)+extend))
        kpsy=[kp.y for kp in kps.keypoints]
        miny=max(0,int(min(kpsy)-extend))
        maxy=min(self.scene_height,int(max(kpsy)+extend))
        if minx==maxx or miny==maxy:
            return None
        else:
            return ia.BoundingBox(x1=minx,y1=miny,x2=maxx,y2=maxy)
    
    def augment_card(self, card):
        transformed_img = image_transformations(image=card)
        transformed_img = np.pad(transformed_img, [
            self.card_h_padding,
            self.card_w_padding, 
            (0,0)
        ])
        kps = self.card_key_points.shift(self.card_w_padding[0], self.card_h_padding[0])
        kps.shape = transformed_img.shape
        transformed_img, transformed_kps = spatial_transformations(image=transformed_img, keypoints=kps)
        return transformed_img[:self.scene_height,:self.scene_width,:], transformed_kps
    
    def move_card_to(self, card, kps, xy):
        return np.roll(card, (xy[1], xy[0]),(0,1)), kps.shift(*xy)
    
    def resolve_overlaps(self, modified_polys, current_poly):
         for idx, old_poly in enumerate(modified_polys):
            if old_poly.overlaps(current_poly):
                new_poly = old_poly - current_poly
                if isinstance(new_poly, MultiPolygon):
                    new_poly = sorted(new_poly, key=lambda pol: pol.area)[1]
                modified_polys[idx] = new_poly
        
    def generate(self, cards):
        full_image = self._generate_random_background()
        keypoint_list = []
        original_polys = []
        modified_polys = []
        bounding_boxes = []
        sample_positions = random.sample(self.grid, len(cards))
        for position, card in zip(sample_positions, cards):
            
            try:
                card_image, keypoints = self.augment_card(card.image())
                card_image, keypoints = self.move_card_to(card_image, keypoints, position)
                current_poly = Scenes.kps_to_polygon(keypoints)
            
                full_image = deep_fry_whole(image= np.where(card_image, card_image, full_image))
            except:
                continue
            self.resolve_overlaps(modified_polys, current_poly)
            
            keypoint_list.append(keypoints)
            original_polys.append(current_poly.intersection(self.scene_poly))
            modified_polys.append(current_poly.intersection(self.scene_poly))
            bounding_boxes.append(self.kps_to_bounding(keypoints))
            
            
        return (
            full_image,
            bounding_boxes,
            original_polys,
            modified_polys,
            keypoint_list,
        )
            


# In[11]:


# scenes = Scenes(bg_images, scene_width=1280, scene_height=720) 
# cards = random.sample(all_available_cards,3)
# board, bounding_boxes, polys, mod, keypoints = scenes.generate(cards)
# 
# fig = plt.figure(dpi=300)
# ax = fig.gca()
# ax.imshow(board)
# rects = []
# for idx, (card, bb, poly) in enumerate(zip(cards,bounding_boxes, mod)):
#     rect = Rectangle((bb.x1, bb.y1),bb.x2-bb.x1, bb.y2-bb.y1,facecolor='red', fill=False, alpha=1)
#     ax.text(bb.x1, bb.y1, f"{idx} - {card.type} - {poly.area:.0f}", size=7)
#     try:
#         ax.plot(*poly.exterior.xy)
#     except:
#         pass
#     rects.append(rect)
# 
# pc = PatchCollection(rects, match_original=True)
# ax.set_aspect('equal')
# ax.add_collection(pc)


# In[13]:


generated_datasets_path = Path("generated")
images_folder = "images"
bounding_boxes_folder = "bounding_boxes"
original_polys_folder = "original_polys"
overlapping_polys_folder = "overlapping_polys"
keypoints_folder = "keypoints_folder"

for folder in [images_folder, bounding_boxes_folder, original_polys_folder, overlapping_polys_folder, keypoints_folder]:
    Path(generated_datasets_path, folder).mkdir(exist_ok=True, parents=True)

scenes = Scenes(bg_images, scene_width=1280, scene_height=720)    
    
def generate_sample(example):
    cards_in_scene = random.randint(3, 10)
    cards = random.sample(all_available_cards, cards_in_scene)
    board, bounding_boxes, original_polys, overlapping_polys, keypoints = scenes.generate(cards)

    sample_id = f"{example:06}"

    io.imsave(Path(generated_datasets_path, images_folder, f"{sample_id}.jpg"), board)

    # Write boundingboxes in YOLO format
    with open(Path(generated_datasets_path, bounding_boxes_folder, f"{sample_id}.txt"), "w") as writable:
        for card, bounding_box in zip(cards, bounding_boxes):
            writable.write(f"{card.type} ")
            writable.write(f"{bounding_box.x1} ")
            writable.write(f"{bounding_box.y1} ")
            writable.write(f"{bounding_box.x2 - bounding_box.x1} ")
            writable.write(f"{bounding_box.y2 - bounding_box.y1}\n")

    # Write overlapping polys in custom format
    # <class> <area> x: <series_of_points,> y: <series_of_points,> 
    with open(Path(generated_datasets_path, overlapping_polys_folder, f"{sample_id}.txt"), "w") as writable:
        for card, poly in zip(cards, overlapping_polys):
            writable.write(f"{card.type} ")
            writable.write(f"{poly.area:.04f} ")


            xs, ys = poly.exterior.xy
            writable.write(f"x ")
            writable.write(" ".join([f"{x:.04f}" for x in xs]))
            writable.write(f" y ")
            writable.write(" ".join([f"{y:.04f}" for y in ys]))

            writable.write("\n")

#    # Write original polys in custom format
#    # <class> <area> x: <series_of_points,> y: <series_of_points,> 
#    with open(Path(generated_datasets_path, original_polys_folder, f"{sample_id}.txt"), "w") as writable:
#        for card, poly in zip(cards, original_polys):
#            writable.write(f"{card.type} ")
#            writable.write(f"{poly.area:.04f} ")
#
#
#            xs, ys = poly.exterior.xy
#            writable.write(f"x ")
#            writable.write(" ".join([f"{x:.04f}" for x in xs]))
#            writable.write(f" y ")
#            writable.write(" ".join([f"{y:.04f}" for y in ys]))
#
#            writable.write("\n")
#
#    # Write keypoints in custom format
#    # <class> <(x, y),> 
#    with open(Path(generated_datasets_path, keypoints_folder, f"{sample_id}.txt"), "w") as writable:
#        for card, kp in zip(cards, keypoints):
#            writable.write(f"{card.type} ")
#            keypoints_str = " ".join([
#                f"({kps.x:.04f},  {kps.y:.04f})"
#                for kps 
#                in kp
#            ])
#            writable.write(keypoints_str)
#
#            writable.write("\n")


# In[ ]:


import time
import os
from joblib import Parallel, delayed

t0 = time.time()
print("starting dataset generation")
Parallel(n_jobs=12, prefer="threads")(delayed(generate_sample)(args) for args in range(50_000))

t1 = time.time()

print(t1-t0)


# In[ ]:




