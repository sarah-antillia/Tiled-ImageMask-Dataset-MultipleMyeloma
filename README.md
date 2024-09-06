<h2>Tiled-ImageMask-Dataset-MultipleMyeloma (Updated: 2024/09/07)</h2>

<br>
This is Tiled-ImageMask Dataset for Multiple Myeloma based on 
<a href="https://segpc-2021.grand-challenge.org/">
Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images (SegPC-2021) 
</a>

<br>
<br>
The dataset used here has been taken from kaggle web-site 
<a href="https://www.kaggle.com/datasets/sbilab/segpc2021dataset">
<b>SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images</b>
</a>
<br>
<br>

<b>Download Tiled-ImageMask-Dataset</b><br>
You can download our datasets from the google drive:<br>

<a href="https://drive.google.com/file/d/105Ppwc5X92_qJhreS1NWUx1-DaCuQd6I/view?usp=sharing">
Tiled-MultipleMyeloma-ImageMask-Dataset.zip</a>
<br>

<br>

<br>
<h3>1. Dataset Citation</h3>
We used the following dataset in kaggle web-site:<br>
<b>SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images</b><br>
<a href="https://www.kaggle.com/datasets/sbilab/segpc2021dataset">
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</a>
<br><br>

<b>Citation:</b><br>
Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: <br>
 Segmentation of Multiple Myeloma Plasma Cells <br>
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.<br>
BibTex<br>
@data{segpc2021,<br>
doi = {10.21227/7np1-2q42},<br>
url = {https://dx.doi.org/10.21227/7np1-2q42},<br>
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },<br>
publisher = {IEEE Dataport},<br>
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},<br>
year = {2021} }<br>
IMPORTANT:<br>
If you use this dataset, please cite below publications-<br>
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy,<br> 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," <br>
 Medical Image Analysis, vol. 65, Oct 2020. DOI: <br>
 (2020 IF: 11.148)<br>
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, <br>
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"<br>
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), <br>
 Barcelona, Spain, 2020, pp. 1389-1393.<br>
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal,<br> 
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," <br>
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908<br>
<br>
<b>License: </b>CC BY-NC-SA 4.0
<br>
<br>

<h3>2. Create master dataset</h3>
Please download <a href="https://www.kaggle.com/datasets/sbilab/segpc2021dataset"><b>SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images</b></a>, and expand it 
in your working directory. It contains three subsets as shown below.
<br>

<pre>
./TCIA_SegPC_dataset
├─test
│  └─x
├─train
│  ├─x
│  └─y
└─valid
    ├─x
    └─y
</pre>
Each x in those subsets contains Multiple Myeloma Plasma Cells images files, and y annotations (mask) files
of 2K pixels.
<br>
<b> train x samples</b>:<br>
<img src="./asset/train_x_samples.png" width="1024" height="auto"><br><br>
<b> train y samples</b>:<br>
<img src="./asset/train_y_samples.png" width="1024" height="auto"><br><br>

Please run the following command for Python script <a href="./ImageMaskDatasetGenerator.py">
ImageMaskDatasetGenerator.py
</a><br>
<pre>
>python ImageMaskDatasetGenerator.py
</pre>
This generates jpg MultipleMyeloma-master dataset.
<pre>
./MultipleMyeloma-master
├─images
└─masks
</pre>
<br>
<hr>
<b>MultipleMyeloma-master/images</b><br>
<img src="./asset/master-images.png" width="1024" height="auto"><br><br>
<b>MultipleMyeloma-master/masks</b><br>
<img src="./asset/master-masks.png" width="1024" height="auto"><br><br>

<hr>

<h3>3. Create tiled dataset</h3>
Please run the following command for Python script <a href="./TiledImageMaskDatasetGenerator.py">
TiledImageMaskDatasetGenerator.py
</a><br>
<pre>
>python TiledImageMaskDatasetGenerator.py
</pre>

This command generates tiledly-splitted 512x512 image and mask files, and size-reduced 512x512 image and mask files 
from MultipleMyeloma-master dataset.<br>
<pre>
./Tiled-MultipleMyeloma-master
├─images
│  ├─10001.jpg
...
│  └─10497_2x2.jpg
└─masks
    ├─10001.jpg
...    
    └─10497_2x2.jpg
</pre>


For example, an image and mask files of 2560x1920 pixels can be split into a lot of 512x512 tiles as shown below:<br>
<hr>
<table>
<tr>
<th>
Image
</th>
<th>
Mask
</th>
</tr>
<tr>
<td>
<img src="./asset/102_image.jpg"  width="480" height="auto">
</td>
<td>
<img src="./asset/102_mask.jpg"  width="480" height="auto">

</td>
</tr>
</table>

<table>
<tr>
<th>
Splitted images
</th>
<th>
Splitted masks
</th>
</tr>
<tr>
<td>
<img src="./asset/tiled-images.png"  width="480" height="auto">

</td>
<td>
<img src="./asset/tiled-mask.png"  width="480" height="auto">

</td>
</tr>

</table>
<br>
<b>However, since all black only masks are irrelevant annotations, 
we excluded those empty mask tiles and corresponding image tiles to generate our tiled-dataset.</b>
In this case, the splitted images and mask will become the following dataset by applying the exclusion operation.
<table>
<tr>
<th>
Splitted images
</th>
<th>
Splitted masks
</th>
</tr>
<tr>
<td>
<img src="./asset/empty_mask_excluded_tiled-images.png"  width="480" height="auto">

</td>
<td>
<img src="./asset/excluded_,mask_tiled-masks.png"  width="480" height="auto">

</td>
</tr>

</table>
 
<hr>


<h3>4. Split tiled dataset</h3>
Please run the following command for Python script <a href="./split_tiled_master.py">
split_tiled_master.py
</a><br>
<pre>
>python split_tiled_master.py
</pre>
This command generates Tiled-MultipleMyeloma-ImageMask-Dataset.<br>
<pre>
./Tiled-MultipleMyeloma-ImageMask-Dataset
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<hr>
<b>train images: </b><br>
<img src="./asset/train_images_sample.png"  width="1024" height="auto">
<br>
<b>train masks: </b><br>
<img src="./asset/train_masks_sample.png"  width="1024" height="auto">
<hr>

<b>Tiled-MultipleMyeloma-ImageMask-Dataset Statistics</b><br>
<img src="./Tiled-MultipleMyeloma-ImageMask-Dataset_Statistics.png" width="512" height="auto"><br>


