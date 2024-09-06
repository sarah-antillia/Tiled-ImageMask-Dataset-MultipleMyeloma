# Copyright 2024 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2024/09/01
# ImageMaskDatasetGenerator.py

import sys
import os
import glob
import random
import shutil
import numpy as np

import traceback
import cv2


class ImageMaskDatasetGenerator:
  def __init__(self):
     pass
  
  def get_image_filepaths(self, images_dir ="./train/x"):
    pattern = images_dir + "/*.bmp"
    print("--- pattern {}".format(pattern))
    all_files  = glob.glob(pattern)
    image_filepaths = []
    for file in all_files:
      basename = os.path.basename(file)
      if basename.find("_") == -1:
        image_filepaths.append(file)
    return image_filepaths

  def get_mask_filepaths(self, image_filepath, mask_dir):
    basename = os.path.basename(image_filepath)
    name     = basename.split(".")[0]
    mask_filepattern  = mask_dir + "/" + name + "_*.bmp"
    mask_filepaths    = glob.glob(mask_filepattern)
    return mask_filepaths

  def generate(self, input_dir, output_images_dir, output_masks_dir):
    images_dir = input_dir + "/x/"
    masks_dir  = input_dir + "/y/"
    image_filepaths  = self.get_image_filepaths(images_dir)
 
    
    for image_filepath in image_filepaths:
      basename = os.path.basename(image_filepath)
      name     = basename.split(".")[0]
      if name == "610":
        print("Skipping unmatched {}".format(image_filepath))
        continue

      img         = cv2.imread(image_filepath)
      h, w, c     = img.shape
      output_img_filepath = os.path.join(output_images_dir, name + ".jpg")
      print("=== Saved image_filepath {} as {}".format(image_filepath, output_img_filepath))
      cv2.imwrite(output_img_filepath, img)
       
      # 3 Get some mask_filepaths corresponding to the image_filepath
      mask_filepaths = self.get_mask_filepaths(image_filepath, masks_dir)

      background = np.zeros((h, w, 3), dtype =np.uint8)
      output_mask_filepath =  os.path.join(output_masks_dir, name + ".jpg")

      for mask_filepath in mask_filepaths:
        mask_basename = os.path.basename(mask_filepath)
        print(mask_basename)
        mask_filename   = mask_basename.split(".")[0]
        mask   = cv2.imread(mask_filepath)
        background += mask
      background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
      # convert backround to black-white mask
      #background[background>10]=255
      r, background = cv2.threshold(background, 128, 255, cv2.THRESH_OTSU)

      cv2.imwrite(output_mask_filepath, background)
      
      print("=== Save mask_image     {}".format(output_mask_filepath))

  

"""
INPUT:

./TCIA_SegPC_dataset
├─test
├─train
└─valid


Output:
./MultipleMyeloma-master-Dataset
├─train
└─valid
 
"""

if __name__ == "__main__":
  try:      

    input_dir   = "./TCIA_SegPC_dataset"
    # For simplicity, we have renamed the folder name from the original "validation" to "valid" 
    datasets    = ["valid", "test", "train"]
    # Exclude test dataset, because it does not contain annotations data.
    datasets    = ["valid", "train"]

    output_dir  = "./MultipleMyeloma-master"
    output_images_dir = "./MultipleMyeloma-master/images"
    output_masks_dir  = "./MultipleMyeloma-master/masks"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    generator = ImageMaskDatasetGenerator()
    debug = True
    crop_ellipse = False
    for dataset in datasets:
      input_subdir  = os.path.join(input_dir, dataset)
      output_subdir = os.path.join(output_dir, dataset)

      generator.generate(input_subdir, output_images_dir, output_masks_dir)
  except:
    traceback.print_exc()

      