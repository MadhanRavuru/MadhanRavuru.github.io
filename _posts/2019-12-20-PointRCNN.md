---
layout: post
title: PointRCNN: 3D Object Proposal generation and detection from Point cloud 
---

# PointRCNN: 3D Object Proposal generation and detection from Point cloud

## Introduction

Detecting and localizing objects in images and videos is a key component of many real-world applications, such as autonomous 
driving and domestic robots. The CNN machinery for 2D object detection with 4 Degrees-of-Freedom (DoF) is mature to handle 
large variations of viewpoints and background clutters in images. In autonomous driving, LiDAR sensors are mostly used to 
generate 3D point clouds. The detection of 3D objects with point clouds still faces great challenges because of the
irregular data format and sparse representation of point cloud with large search space. In the below figure, we can see the
7 DoF of 3D object detection (position and dimensions of the bounding box along with box orientation).

| ![boundingbox]({{ site.baseurl }}/images/boundbox.png) |
|:--:| 
| *2D and 3D object detection. Images are adapted from 3D Bounding Box Estimation Using Deep Learning and Geometry (Arsalan Mousavian et. at. CVPR 2017)* |

The [KITTI vision benchmark](http://www.cvlibs.net/datasets/kitti/) provides a standardized dataset for training and evaluating the performance of different 3D object detectors. The proposed PointRCNN outperforms
state-of-the-art methods with remarkable margins by using only point cloud as input on KITTI dataset.

## Related Work

Many state-of-the-art 3D detection methods make use of the mature 2D object detection frameworks by projection of point cloud to Bird's-eye view(BEV) or regular 3D voxels for feature learning.

### Aggregate View Object Detection (AVOD)
AVOD uses LIDAR point clouds and RGB images to generate features that are shared by two subnetworks: a region proposal network (RPN) and a second stage detector network. The RPN places 80-100k anchor boxes in the 3D space and for each anchor box, features are pooled in multiple views for generating proposals. The second stage detection network uses the generated proposals to predict the accurate extents, orientation and classification of objects in 3D space. However, transforming point cloud to BEV loses geometric information and is not optimal.

| ![avod]({{ site.baseurl }}/images/avod.png) |
|:--:| 
| *Jason Ku, Melissa Mozifian, Jungwook Lee, Ali Harakeh and Steven Lake Waslander. Joint 3d proposal generation and object detection from view aggregation. CoRR, 2017.* |

### Frustum-Pointnet
This method estimates 3D bounding boxes based on 3D points cropped from 2D regions. Given RGB-D data, 2D object region proposals in the RGB image are generated using a CNN. Then from the depth data, each 2D region is extruded to a 3D viewing frustum to get a point cloud. Finally, our frustum PointNet predicts a 3D bounding box for the object from the points in frustum. But, perfomance of this method heavily relies on 2D detection without taking the advantages of 3D information.

| ![frustum]({{ site.baseurl }}/images/frustum.png) |
|:--:| 
| *Charles Ruizhongtai Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J. Guibas. Frustum pointnets for 3d object detection from RGB-D data. CoRR, 2017* |

### VoxelNet
VoxelNet is comprised of feature learning network, convolutional middle layers and RPN. The feature learning network divides raw point cloud into equally spaced 3D voxels. The points within each voxel are transformed into a unified feature representation through the voxel feature encoding (VFE) layer. Then, 3D convolution is applied to get aggregate spatial context. Finally, a RPN generates the 3D detection. However, information loss occurs during quantization. Also, 3D convolution suffers from greater computational cost and thus higher latency in comparison to 2D convolution.


| ![voxel]({{ site.baseurl }}/images/voxel.png) |
|:--:| 
| *Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. CoRR, 2017.* |


## Our Method

In contrast to the mentioned related work, our bottom-to-up 3D proposal generation method directly generates robust 3D proposals from raw point clouds, which is optimal, efficient and free from quantization. The proposed method comprises of 2 stages.

1. Bottom-up 3D box proposal generation
* PointNet++ with multi-scale grouping as our backbone network
* Segmenting the point cloud of the whole scene into foreground and background points 
* Generates high quality 3D box proposals in bottom-up manner 

2.	Refining the proposals in the canonical coordinates
* Point cloud region pooling (proposed by authors)
* Transforming the pooled points of each proposal to canonical coordinates (helps to learn better local spatial features) 

An h1 header
============

Paragraphs are separated by a blank line.

2nd paragraph. *Italic*, **bold**, and `monospace`. Itemized lists
look like:

  * this one
  * that one
  * the other one

Note that --- not considering the asterisk --- the actual text
content starts at 4-columns in.

> Block quotes are
> written like so.
>
> They can span multiple paragraphs,
> if you like.

Use 3 dashes for an em-dash. Use 2 dashes for ranges (ex., "it's all
in chapters 12--14"). Three dots ... will be converted to an ellipsis.
Unicode is supported. ☺



An h2 header
------------

Here's a numbered list:

 1. first item
 2. second item
 3. third item

Note again how the actual text starts at 4 columns in (4 characters
from the left side). Here's a code sample:

    # Let me re-iterate ...
    for i in 1 .. 10 { do-something(i) }

As you probably guessed, indented 4 spaces. By the way, instead of
indenting the block, you can use delimited blocks, if you like:

~~~
define foobar() {
    print "Welcome to flavor country!";
}
~~~

(which makes copying & pasting easier). You can optionally mark the
delimited block for Pandoc to syntax highlight it:

~~~python
import time
# Quick, count to ten!
for i in range(10):
    # (but not *too* quick)
    time.sleep(0.5)
    print i
~~~



### An h3 header ###

Now a nested list:

 1. First, get these ingredients:

      * carrots
      * celery
      * lentils

 2. Boil some water.

 3. Dump everything in the pot and follow
    this algorithm:

        find wooden spoon
        uncover pot
        stir
        cover pot
        balance wooden spoon precariously on pot handle
        wait 10 minutes
        goto first step (or shut off burner when done)

    Do not bump wooden spoon or it will fall.

Notice again how text always lines up on 4-space indents (including
that last line which continues item 3 above).

Here's a link to [a website](http://foo.bar), to a [local
doc](local-doc.html), and to a [section heading in the current
doc](#an-h2-header). Here's a footnote [^1].

[^1]: Footnote text goes here.

Tables can look like this:

size  material      color
----  ------------  ------------
9     leather       brown
10    hemp canvas   natural
11    glass         transparent

Table: Shoes, their sizes, and what they're made of

(The above is the caption for the table.) Pandoc also supports
multi-line tables:

--------  -----------------------
keyword   text
--------  -----------------------
red       Sunsets, apples, and
          other red or reddish
          things.

green     Leaves, grass, frogs
          and other things it's
          not easy being.
--------  -----------------------

A horizontal rule follows.

***

Here's a definition list:

apples
  : Good for making applesauce.
oranges
  : Citrus!
tomatoes
  : There's no "e" in tomatoe.

Again, text is indented 4 spaces. (Put a blank line between each
term/definition pair to spread things out more.)

Here's a "line block":

| Line one
|   Line too
| Line tree

and images can be specified like so:

![example image](example-image.jpg "An exemplary image")

Inline math equations go in like so: $\omega = d\phi / dt$. Display
math should get its own line and be put in in double-dollarsigns:

$$I = \int \rho R^{2} dV$$

And note that you can backslash-escape any punctuation characters
which you wish to be displayed literally, ex.: \`foo\`, \*bar\*, etc.

