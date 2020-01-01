---
layout: post
title: PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud 
mathjax: true
---

# PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud

## Introduction

Detecting and localizing objects in images and videos is a key component for many real-world applications, such as autonomous driving and domestic robots. The CNN machinery for 2D object detection with 4 Degrees-of-Freedom (DoF) is mature to handle large variations of viewpoints and background clutters in images. Beyond understanding of 2D scenes, 3D object detection is in-dispensable and crucial for many applications. The remarkable progress achieved by CNN on 2D object detection in recent years has not transferred well to 3D object detection. Unlike 2D object detection, the 3D object detection requires also the estimation of orientation of bounding boxes. In the below figure, we can see the 7 DoF of 3D object detection (position and dimensions of the bounding box along with box orientation).

| ![boundingbox]({{ site.baseurl }}/images/boundbox.png) |
|:--:| 
| *2D and 3D object detection. Images are adapted from 3D Bounding Box Estimation Using Deep Learning and Geometry (Arsalan Mousavian, Dragomir Anguelov, John Flynn, Jana Kosecka, CVPR 2017)* |

In autonomous driving, LiDAR sensors are mostly used to generate 3D point clouds. Using point clouds as input, the detection of 3D objects still faces great challenges because of the irregular data format and sparse representation of point cloud with large search space. Many state-of-the-art methods project point cloud to Bird's-Eye View(BEV) or to regular 3D voxels, and detect 3D objects using 2D object detection frameworks. However, these are not optimal strategies and suffer from informtion loss during quantization. The bounding boxes could provide only weak supervisions for semantic segmentation in 2D object detection. But, the training data for 3D object detection directly provides 3D annotated bounding boxes. This implies that the semantic masks for 3D object segmentation are given by the 3D ground-truth boxes. This is a key difference between 2D and 3D object detection training data. Based on this observation, a novel two-stage 3D object detection framework ([PointRCNN](https://arxiv.org/abs/1812.04244)) is introduced by the authors. This method directly uses 3D point clouds as input and achieves accurate and robust 3D detection performance. The whole framework of this method is composed of two stages: stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in the canonical coordinates to obtain the final detection results. The [KITTI vision benchmark](http://www.cvlibs.net/datasets/kitti/) provides a standardized dataset for training and evaluating the performance of different 3D object detectors. The proposed method outperforms state-of-the-art methods with remarkable margins by using only point cloud as input on KITTI dataset.

## Related Work

Many state-of-the-art 3D detection methods make use of the mature 2D object detection frameworks by projection of point cloud to BEV or by quantization to regular 3D voxels for feature learning. However, the projection of point cloud to BEV loses geometric information. Transformation of point clouds to volumetric grids suffer from information loss during quantization. Such kind of data transformation might often obscure natural 3D patterns and invariances of the 3D data. We also discuss about Frustum-Pointnet, which uses mature 2D CNN framework and advanced 3D deep learning for object localization. This method relies completely on 2D detection performance, without taking advantage of 3D information for robust 3D bounding box proposal generation. 

### Aggregate View Object Detection (AVOD)
[AVOD](https://arxiv.org/abs/1712.02294) uses LIDAR point clouds and RGB images to generate features that are shared by two subnetworks: a region proposal network (RPN) and a second stage detector network. The RPN places 80-100k anchor boxes in the 3D space and for each anchor box, features are pooled in multiple views for generating proposals. A novel architecture, which is capable of performing multimodal feature fusion to generate reliable 3D object proposals on high resolution feature maps is used in RPN. The second stage detection network uses the generated proposals to predict the accurate extents, orientation and classification of objects in 3D space. However, transforming point cloud to BEV loses geometric information. Also, this method uses large number of anchor boxes for proposal generation, which is not optimal.

| ![avod]({{ site.baseurl }}/images/avod.png) |
|:--:| 
| *Jason Ku, Melissa Mozifian, Jungwook Lee, Ali Harakeh and Steven Lake Waslander. Joint 3d proposal generation and object detection from view aggregation. CoRR, 2017.* |

### Frustum-Pointnet
[Frustum-Pointnet](https://arxiv.org/abs/1711.08488) estimates 3D bounding boxes based on 3D points cropped from 2D regions. 
The method directly operates on raw point clouds by popping up RGB-D scans, and avoids obscuring natural 3D patterns and invariances of 3D data. Instead of relying completely on 3D proposals, Frustum-Pointnet uses mature 2D object detectors and advanced 3D deep learning for localization of objects. Given RGB-D data, 2D object region proposals in the RGB image are generated using a 2D CNN. Then from the depth data, each 2D region is extruded to a 3D viewing frustum to get a point cloud. Finally, Frustum-PointNet predicts a 3D bounding box for the object from the points in frustum. This might miss difficult objects, which are only clearly visible from 3D space. Also, perfomance of this method heavily relies on 2D detection without taking the advantages of 3D information.

| ![frustum]({{ site.baseurl }}/images/frustum.png) |
|:--:| 
| *Charles Ruizhongtai Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J. Guibas. Frustum pointnets for 3d object detection from RGB-D data. CoRR, 2017* |

### VoxelNet
[VoxelNet](https://arxiv.org/abs/1711.06396) is a generic 3D object detection network, which combines extraction of feature and prediction of bounding box into a single stage, end-to-end trainable deep network. It is comprised of feature learning network, convolutional middle layers and RPN. The feature learning network divides raw point cloud into equally spaced 3D voxels. The points within each voxel are transformed into a unified feature representation through the voxel feature encoding (VFE) layer. Then, 3D convolution is applied to get aggregate spatial context. Finally, a RPN generates the 3D detection. However, information loss occurs during quantization of point cloud to volumetric representation. Also, 3D convolution suffers from greater computational cost and thus higher latency in comparison to 2D convolution. 3D CNN is
both memory and computation inefficient, and is not optimal.

| ![voxel]({{ site.baseurl }}/images/voxel.png) |
|:--:| 
| *Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. CoRR, 2017.* |


## PointRCNN

In contrast to the mentioned related work, PointRCNN achieves robust 3D object detection performance from raw point clouds, which is optimal, efficient and free from quantization. The proposed method comprises of 2 stages:

**a. Bottom-up 3D box proposal generation**
* PointNet++ with multi-scale grouping as our backbone network
* Segmenting the point cloud of the whole scene into foreground and background points 
* Generates high quality 3D box proposals in bottom-up manner 

**b. Refining the proposals in the canonical coordinates**
* Point cloud region pooling (proposed by authors)
* Transforming the pooled points of each proposal to canonical coordinates (helps to learn better local spatial features) 

| ![overview]({{ site.baseurl }}/images/overview.png) |
|:--:| 
| *Overview of PointRCNN* |

### Bottom-up 3D box proposal generation (Stage 1)

[PointNet++](https://arxiv.org/abs/1706.02413) with multi-scale grouping is used as our backbone network to learn discriminative point-wise features for describing the raw point clouds. An alternative point-cloud network structures, such as VoxelNet with [sparse convolutions](https://arxiv.org/abs/1711.10275), could also be adopted as our backbone network.
Point-wise features for the segmentation of the raw point cloud of the whole scene into foreground and background points are learned. Also, simultaneously 3D proposals are generated from the segmented foreground points. Thus, PointRCNN avoids using a large set of predefined 3D boxes in the 3D space and limits the search space for 3D object proposal generation.

#### Foreground point segmentation

As objects in 3D scenes are naturally well-separated without overlapping each other, 3D points inside 3D boxes are considered as foreground points. Thus, the training data for 3D object detection directly provides the ground-truth segmentation mask for 3D object segmentation. From the point-wise features encoded by the PointNet++, one segmentation head is appended for the estimation of the foreground mask and one box regression head for the generation of 3D proposals. The foreground segmentation and 3D box proposal generation are done simultaneously. As the number of foreground
points is usually much smaller than background points for a general large-scale outdoor scene, focal loss is used to handle the class imbalance problem.

The formula for [focal loss](https://arxiv.org/abs/1708.02002) is given by,

$$\begin{aligned}
L_{\mathrm{focal}}(p_{t}) &= -\alpha_{t}(1-p_{t})^{\gamma} \mathrm{log}(p_{t}), 
\\ \\ \text{where} \  p_{t} &= 
    \begin{cases}
      p & \text{for foreground point} \\
      1-p & \text{otherwise}
    \end{cases} 
\end{aligned}$$    

While training point cloud segmentation, the parameters $$\alpha_{t} \text{and} \gamma $$ are chosen as 0.25 and 2 respectively.

| ![FocalLossgraph]({{ site.baseurl }}/images/FocalLossgraph.png) |
|:--:| 
| *Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar. Focal Loss for Dense Object Detection. CVPR, 2018* |

Focal Loss adds a factor $$(1-p_{t})^{\gamma}$$ to standard cross entropy loss. By setting $$\gamma > 0$$, we put more focus on hard, misclassified examples and reduce the relative loss for well-classified examples $$(p_{t} > 0.5).$$


#### Bin based 3D bounding box proposal generation

As mentioned above, a box regression head is also appended for simultaneously generating 3D proposals with the foreground point segmentation. During training, for each foreground point, 3D bounding box location is regressed from the box regression head. Although, the background points are not used for regressing the boxes, these points provide supporting information for the box proposal generation because of the receptive field of the point-cloud network.

In the LiDAR coordinate system, a 3D bounding box is represented as $$(x, y, z, h, w, l, θ)$$, where $$(x, y, z)$$ is the center location of object, $$(h, w, l)$$ is the size of object, and $$ θ $$ is the orientation of object from the BEV.
[Direct regression is presumably harder task and can introduce instability](https://arxiv.org/abs/1901.02970) during training. To limit the generated 3D box proposals, bin-based regression loss is introduced for estimation of 3D bounding boxes. For estimating object center location, each foreground point surrounding area is split into a series of discrete bins along the $$X$$ and $$Z$$ axes. Along $$X$$ and $$Z$$ axis of current foreground point, 1D search range $$\mathcal{S}$$ is set and divided into bins of uniform length $$\delta$$ for representing different centers of object $$(x, z)$$ on the *X*-*Z* plane. Now, bin classification along eadetection performance. The whole framework of this method is composed of two stages: stage-1 for the bottom-up 3D proposal generation and stage-2 for refining proposals in the canonical coordinates to obtain the final detection results. The KITTI vision benchmark ch $$X$$ and $$Z$$ axis is performed along with residual regression within the classified bin. Thus, localization loss for the $$X$$ or $$Z$$ axis consists of two terms. Bin-based loss formulation instead of direct regression with smooth $$L$$1 loss results in more robust and accurate center localization. For estimating box orientation, the orientation $$2\pi$$ is divided into *n* bins and bin classification target
and residual regression target is found similar to $$x$$ or $$z$$ prediction.

| ![xzestimate]({{ site.baseurl }}/images/xzestimate.png) |
|:--:| 
| *Illustration of bin-based localization* |

The localization target is formulated as,

$$\begin{aligned}
\text{bin}_{u}^{(p)} &=\left\lfloor\frac{u^{p}-u^{(p)}+\mathcal{S}}{\delta}\right\rfloor \ \ \forall \ u\in{\{x,z,\theta\}}\ \\
\text{res}_{u}^{(p)} &= \dfrac{1}{C}\left ( u^{p}-u^{(p)}+\mathcal{S}- \left ( \text{bin}_{u}^{(p)}\cdot \delta + \dfrac{\delta}{2} \right )\right )\  \ \forall \ u\in{\{x,z,\theta\}}\ ,\\
\end{aligned}$$


To calculate the center location $$y$$ along the vertical $$Y$$ axis, [smooth L1 loss](https://arxiv.org/pdf/1711.06753.pdf) is directly used for regression as most object's $$y$$ values are within a very limited range. So, using the smooth $$L$$1 loss is sufficient to obtain accurate $$y$$ values. Also, for the object size $$(h, w, l)$$ estimation, smooth $$L$$1  loss is used to directly regress by calculating residuals w.r.t. the mean object size of each class in the whole training set. Thus, the localization target for $$y,h,w,l$$ is given by,


$$\begin{aligned}
\text{res}_{v}^{(p)} &= v^{p}-v^{(p)}\ \ \ \forall \ v\in{\{y,h,w,l\}}\
\end{aligned}$$

In the above formulation of localization targets, $$(x^{(p)},y^{(p)},z^{(p)})$$ denote the coordinate of a interested foreground point, $$(x^{p},y^{p},z^{p})$$ is the center coordinate of its corresponding object and $$C$$ is the bin length for normalization. $$\text{bin}_{u}^{(p)}$$ and $$\text{res}_{u}^{(p)}$$ are the bin classification target and residual regression target respectively. $$\text{res}_{v}^{(p)}$$ is the regression target with smooth $$L$$1 loss formulation.

In the inference stage, the predicted residual is added to their initial values, for directly regressed parameters ($$y,h,w,l$$). To find the bin-based predicted parameters ($$x,z,θ$$), the bin center with high predicted confidence is chosen and later, the predicted residual is added for obtaining the refined parameters.

For training, the overall 3D bounding box regression loss $$\mathcal{L}_{\mathrm{reg}}$$ could then be formulated as,

$$\begin{aligned}
\mathcal{L}_{\mathrm{bin}}^{(p)} &=\sum_{u \in\{x,z,\theta\}} (\mathcal{F}_{\mathrm{cls}}(\widehat{\mathrm{bin}}_{u}^{(p)}, \mathrm{bin}_{u}^{(p)})\ +\ \mathcal{F}_{\mathrm{reg}}(\widehat{\mathrm{res}}_{u}^{(p)}, \mathrm{res}_{u}^{(p)})) , \\
\mathcal{L}_{\mathrm{res}}^{(p)} &=\sum_{v \in\{y, h, w, l\}} \mathcal{F}_{\mathrm{reg}}(\widehat{\mathrm{res}}_{v}^{(p)}, \mathrm{res}_{v}^{(p)}), \\
\mathcal{L}_{\mathrm{reg}} &=\frac{1}{N_{\mathrm{pos}}} \sum_{p \in \mathrm{pos}}\left(\mathcal{L}_{\mathrm{bin}}^{(p)}+\mathcal{L}_{\mathrm{res}}^{(p)}\right)
\end{aligned}$$

where $$N_{\mathrm{pos}}$$ is the number of foreground points, $$\text{bin}_{u}^{(p)}$$ and $$\text{res}_{u}^{(p)}$$ are the ground-truth targets, $$\widehat{\mathrm{bin}}_{u}^{(p)}$$ and $$\widehat{\mathrm{res}}_{u}^{(p)}$$ are the predicted bin assignments and residuals of the foreground point $$p$$ respectively. $$\mathcal{F}_{\mathrm{cls}}$$ represents the cross-entropy loss for classification, and $$\mathcal{F}_{\mathrm{reg}}$$ represents the smooth $$L$$1 loss for regression.

For removal of the redundant proposals, non maximum suppression (NMS) based on the oriented IoU from BEV is performed to generate a small number of robust and accurate proposals. These proposals are then fed to stage-2 network.


### Refining the proposals in the canonical coordinates (Stage 2)
Stage-2 network performs refinement of 3D generated proposals in the canonical coordinates. Point cloud region pooling operation is performed for pooling learned point representations from stage-1. This operation helps us to learn more specific local features of each proposal. Then, the pooled points of each proposal are transformed to canonical coordinates to learn better local spatial features. These local features are combined with global semantic features of each point, which is learned in stage-1 for accurate and robust box refinement and confidence prediction.

#### Point cloud region pooling
In general, for occluded objects or distant objects from the sensor, the number of points are less inside the generated proposals. More contextual information should be included for better classification and proposal refinement. Also, for all other generated proposals in stage-1, more points should be included. So, each 3D box proposal, $$\mathbf{b}_{i} = (x_i, y_i, z_i, h_i, w_i, l_i , \theta_i)$$ from stage-1 is enlarged to get new 3D box, $$\mathbf{b}_{i}^{e} = (x_i, y_i, z_i, h_i+\eta, w_i+\eta, l_i+\eta , \theta_i)$$. This helps us in encoding additional contextual information, where $$\eta$$ is some constant value for enlarging the bounding box. The 3D bounding box might also include some background points, while enlarging it. Because of the receptive field of the point-cloud network, these background points will provide supporting information for better refinement of proposals.

| ![RegionPool]({{ site.baseurl }}/images/RegionPool.png) |
|:--:| 
| *Point cloud region pooling* |

An inside/outside test is performed for each point $$p = (x^{(p)},y^{(p)},z^{(p)})$$, to find whether the point p lies inside $$\mathbf{b}_{i}^{e}$$. If the point lies inside, the point and its features are kept for refining the box $$\mathbf{b}_{i}$$. The box proposals, which does not have inside points are removed. The associated features with the inside point *p* include its coordinates $$(x^{(p)},y^{(p)},z^{(p)}) \in\mathbb{R}^{3}$$ and laser reflection intensity $$r^{(p)} \in \mathbb{R}$$, its 'C'-dimensional learned point feature $$\mathbf{f}^{(p)} \in \mathbb{R}^{C}$$ and predicted segmentation mask $$m^{(p)} \in \{0, 1\}$$ from stage-1. The segmentation mask $$m^{(p)}$$ is included to differentiate between predicted foreground/background points within the enlarged bounding box $$\mathbf{b}_{i}^{e}$$. The learned point feature $$\mathbf{f}^{(p)}$$ encodes valuable information regarding learning for segmentation and proposal generation and is therefore included.

#### Canonical 3D bounding box refinement

The pooled points and their corresponding features for each proposal are fed to our stage-2 sub-network for refining the 3D box locations and foreground object confidence. The refinement of the proposals are performed in the canonical coordinate system by combining global semantic features from stage-1 and local spatial features. The transformation into canonical coordinate system eliminates rotation and location variations and helps in better learning of features for the stage-2 sub-network.

##### Canonical transformation
For better learning of local spatial features, the pooled points of each proposal are transformed to canonical coordinate system of the corresponding 3D proposal. The canonical coordinate system for each 3D box proposal implies that (1) the origin is located at the box proposal centre; (2) the local $${X}'$$ and $${Z}'$$ axes are parallel to the ground
with $${X}'$$ pointing in the head direction of proposal with $${Z}'$$ axis perpendicular to $${X}'$$; (3) the $${Y}'$$ axis remains same as the LiDAR coordinate system. By applying proper rotation and translation, coordinates *p* of all pooled points of the box proposal in the LiDAR coordinate system are transformed to the canonical coordinate system as $$\tilde{p}$$.

| ![canonical]({{ site.baseurl }}/images/canonical.png) |
|:--:| 
| *Illustration of canonical transformation* |

##### Feature learning for box proposal refinement
The global semantic features $$\mathbf{f}^{(p)}$$ from stage-1 are combined with the transformed local spatial features $$\tilde{p}$$ for further box and confidence refinement. The canonical transformation inevitably loses depth information of each object. For the compensation of the lost depth information, the features of point *p* are included with distance to the sensor, *i.e.*, $$d^{(p)} = \sqrt{({x^{(p)}})^{2} + ({y^{(p)}})^{2} + ({z^{(p)}})^{2}}$$. The local spatial features $$\tilde{p}$$, predicted segmentation mask $$m^{(p)}$$, laser reflection intensity $$r^{(p)}$$, and distance $$d^{(p)}$$ of associated points for each proposal are first concatenated and fed to several fully-connected layers to get same dimension as the global features $$\mathbf{f}^{(p)}$$. Then, the local and global features are merged to get discriminative feature vector for confidence classification and box refinement.

##### Losses for box proposal refinement
The similar bin-based regression losses, as used in box proposal generation, are adopted for refinement of the box proposal.
For learning box refinement, a ground-truth box is assigned to a 3D box proposal, if their 3D IoU is greater than 0.55. The refinement of proposals are performed in the canonical coordinate system. So, the 3D proposals and their respective 3D ground-truth boxes are transformed into canonical coordinate system. This means that 3D proposal $$\mathbf{b}_{i} = (x_i, y_i, z_i, h_i, w_i, l_i , \theta_i)$$ and 3D ground-truth box $$\mathbf{b}_{i}^{\mathrm{gt}} = (x_i^{\mathrm{gt}}, y_i^{\mathrm{gt}}, z_i^{\mathrm{gt}}, h_i^{\mathrm{gt}}, w_i^{\mathrm{gt}}, l_i^{\mathrm{gt}} , \theta_i^{\mathrm{gt}})$$ are converted to $$\tilde{\mathbf{b}}_{i} = (0,0,0, h_i, w_i, l_i,0)$$ and $$\tilde{\mathbf{b}}_{i}^{\mathrm{gt}} = (x_i^{\mathrm{gt}}-x_i, y_i^{\mathrm{gt}}-y_i, z_i^{\mathrm{gt}}-z_i, h_i^{\mathrm{gt}}, w_i^{\mathrm{gt}}, l_i^{\mathrm{gt}} , \theta_i^{\mathrm{gt}}-\theta_i)$$ respectively. Now, smaller search range $$\mathcal{S}$$ is set for refining the locations of 3D proposals with training targets for the *i*th box proposal’s center location, $$\mathrm{bin}_{\Delta{x}}^i,\mathrm{bin}_{\Delta{z}}^i,\mathrm{res}_{\Delta{x}}^i,\mathrm{res}_{\Delta{z}}^i,\mathrm{res}_{\Delta{y}}^i$$. The box size residuals, $$\mathrm{res}_{\Delta{h}}^i,\mathrm{res}_{\Delta{w}}^i,\mathrm{res}_{\Delta{l}}^i$$ are directly regressed w.r.t to average size of object of each class in the whole training set. As the 3D IoU between 3D proposal and its ground-truth box is atleast 0.55, refinement of box orientation is performed, assuming that the angular difference w.r.t. the ground-truth orientation, $$\theta_i^{\mathrm{gt}}-\theta_i$$ lies in the interval $$\left [ -\frac{\pi}{4},\frac{\pi}{4} \right ]$$. So, $$\frac{\pi}{2}$$ is divided into uniform bins of size $$\omega$$ and the bin-based orientation targets are set as,

$$\begin{aligned}
\text{bin}_{\Delta{\theta}}^{i} &=\left\lfloor\frac{\theta_i^{\mathrm{gt}}-\theta_i +\frac{\pi}{4}}{\omega}\right\rfloor , \\
\text{res}_{\Delta{\theta}}^{i} &=\dfrac{2}{\omega}\left ( {\theta_i^{\mathrm{gt}}-\theta_i +\frac{\pi}{4}} - \left ( \text{bin}_{\Delta{\theta}}^{i} \cdot \omega + \dfrac{\omega}{2} \right ) \right )
\end{aligned}$$

Thus, the overall loss for the stage-2 network is formulated as,
$$\mathcal{L}_{\mathrm{refine}} =\dfrac{1}{\left \| \mathcal{B} \right \|} \sum_{i \in\mathcal{B}} \mathcal{F}_{\mathrm{cls}}(\mathrm{prob}_{i}, \mathrm{label}_{i}) + \dfrac{1}{\left \| \mathcal{B}_\mathrm{pos} \right \|} \sum_{i \in\mathcal{B}_\mathrm{pos}}(\mathcal{\tilde{L}}_{\mathrm{bin}}^{(i)} + \mathcal{\tilde{L}}_{\mathrm{res}}^{(i)})$$

where $$\mathcal{B}$$ is the 3D proposals set from stage-1 and $$\mathcal{B}_\mathrm{pos}$$ denote the positive proposals for regression. The estimated confidence of $$\tilde{\mathbf{b}}_{i}$$ and corresponding label are represented by $$\mathrm{prob}_{i}$$ and $$\mathrm{label}_{i}$$ respectively. The cross entropy loss for supervision of the predicted confidence is denoted by $$\mathcal{F}_{\mathrm{cls}}$$. $$\mathcal{\tilde{L}}_{\mathrm{bin}}^{(i)}$$ and $$\mathcal{\tilde{L}}_{\mathrm{res}}^{(i)}$$ are classification and regression losses similar to $$\mathcal{L}_{\mathrm{bin}}^{(p)}$$  and $$\mathcal{L}_{\mathrm{res}}^{(p)}$$ respectively with new targets calculated by 
$$\tilde{\mathbf{b}}_{i}$$ and $$\tilde{\mathbf{b}}_{i}^{\mathrm{gt}}$$. Finally, oriented NMS with BEV IoU threshold 0.01 is applied to remove the overlapping bounding boxes and generate the 3D refined bounding boxes.

## Experiments and Results

The qualitative results for car detection using PointRCNN method are shown in the below figure. For each sample, the upper part is the RGB image and lower part is the corresponding point cloud representation. Note that the image in the upper part for each sample is just for better visualization. PointRCNN takes only the point cloud as input for the generation of 3D object detection.  Detected car objects are enclosed with green 3D bounding boxes in both upper and lower parts. The driving direction(orientation) for each object is shown by a X mark in the upper part and a red tube in the lower part.

| ![experiment]({{ site.baseurl }}/images/experiment.png) |
|:--:| 
| *Qualitative results of PointRCNN on the KITTI test split* |

The evaluation of PointRCNN on the challenging KITTI dataset and comparison with state-of-the-art 3D object detection methods is performed. Many ablation studies for the analysis of PointRCNN are also conducted.

### Results of official KITTI test split

Average Precision(AP) is used as the evaluation metric with IoU threshold 0.7 for car and 0.5 for pedestrian and cyclist. Results are tabulated for all three difficulty level, i.e. , easy, moderate and hard. We can observe that PointRCNN outperforms other methods with good margins for the 3D car detection. Other methods include [MV3D](https://arxiv.org/abs/1611.07759), [UberATG-ContFuse](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf), [AVOD-FPN](https://arxiv.org/abs/1712.02294), [F-PointNet](https://arxiv.org/abs/1711.08488), [VoxelNet](https://arxiv.org/abs/1711.06396) and [SECOND](https://www.mdpi.com/1424-8220/18/10/3337). MV3D, UberATG-ContFuse, AVOD-FPN and F-PointNet uses both RGB image and point cloud as input, while VoxelNet, SECOND and PointRCNN(our method) uses only point cloud as input. 

| ![testsplitcar]({{ site.baseurl }}/images/testsplitcar.png) |
|:--:| 
| *Performance comparison of 3D car detection with different methods on KITTI test split* |

For pedestrian detection, PointRCNN does not perform well in comparison to methods with multiple sensors. But, in comparison with previous LiDAR-only methods, PointRCNN achieves comparable results. As pedestrians have small size, more details of pedestrian can be captured by image rather than point cloud, and thus, performance of our method is less. Also, the 3D annotated boxes for pedestrian class might not be well separated from each other for a normal 3D scene in comparison with car or cyclist class. Eventually, the foreground points might overlap between the 3D annotated boxes belonging to pedestrian class. As a result, the performance for pedestrian detection using LiDAR only methods will be less. For the cyclist detection, our method performs the best on all three difficulties. 

| ![testsplitother]({{ site.baseurl }}/images/testsplitother.png) |
|:--:| 
| *Performance comparison of 3D pedestrian and cyclist detection with different methods on KITTI test split* |

### Evaluation of 3D proposal generation

The performance of bottom-up proposal generation in PointRCNN is evaluated from recall with different number of Region of Interests(RoIs) and 3D IoU threshold for the car class at moderate difficulty on the val split. Our method achieves remarkable recall in comparison with MV3D and AVOD. Only these two methods reported the number of recall. Though the recall of proposal generations are not directly related to the final 3D detection performance, the higher recall of PointRCNN method conveys accuracy and robustness of bottom-up proposal generation network.

| ![recalltable]({{ site.baseurl }}/images/recalltabulated.png) |
|:--:| 
| *Recall of proposal generation network for car class at moderate difficulty on val split* |

### Ablation Study
For analyzing the effectiveness of various components of PointRCNN, extensive ablation experimens were conducted. All experiments were evaluated on *val* split with car class.

#### Different inputs for the refinement sub-network
The inputs of the refinement sub-network include the merged features, which include pooled features of each pooled point and canonically transformed coordinates. Different combinations of the features are made and fed to the refinement sub-network by removing one feature and keeping all others unchanged. Now, the effect of each feature can be studied. Stage-1 sub-network is same for all experiments to have fair comparison. The results are tabulated below.  $$AP_E , AP_M$$ and $$AP_H$$ represent the average precision for easy, moderate, hard difficulty respectively and CT denotes canonical transformation. We can see that without the CT of pooled points, the refinement sub-network performance drops significantly. This means that transformation into canonical coordinate system helps in learning better local spatial features by eliminating rotation and translation variations and thereby, improves the efficiency of feature learning in stage-2. Slight decrease in performances are observed with removal of stage-1 features $$\mathbf{f}^{(p)}$$ learned from point cloud segmentation, camera depth information
$$d^{(p)}$$ and segmentation mask $$m^{(p)}$$ one at a time. $$\mathbf{f}^{(p)}$$ is needed to take advantages of learning for semantic segmentation in the stage-1. $$d^{(p)}$$ helps in compensation of distance information eliminated during CT and $$m^{(p)}$$ is used to differentiate between foreground and background points in the pooled regions. Thus, in order to achieve good final performance, all features must be considered for the refinement sub-network.

| ![ablation]({{ site.baseurl }}/images/ablation.png) |
|:--:| 
| *Performance with different input combinations of refinement network* |

#### Context-aware point cloud pooling
Each box proposal $$\mathbf{b}_{i}$$ is enlarged by a margin $$\eta$$ to get new 3D box $$\mathbf{b}_{i}^{e}$$ for storing additional contextual information. The below table shows the performance for different pooled context widths $$\eta$$. For $$\eta=1.0m$$, the best performance for all difficulty levels is obtained. $$AP_H$$ drops significantly, when no pooling of contextual information is performed. The hard cases are those, where the object is occluded or far away from sensor. The number of points in generated proposals for such cases are usually smaller and eventually, they need more context information for classification and proposal refinement. Also, when $$\eta$$ is too large, the performance decreases. This may be due to inclusion of noisy foreground points of other objects for the pooled region of current proposals. Therefore, optimal value of $$\eta$$ must be chosen, while enlarging the box, to have good performance.

| ![contextwidth]({{ site.baseurl }}/images/contextwidth.png) |
|:--:| 
| *Performance with different value of context width $$\eta$$* |

#### Losses of 3D bounding box regression
The authors have proposed the bin-based localization for the generation of 3D box proposals. The performances with different types of 3D box regression loss used for stage-1 sub-network are comapred. The other 3D box regression loss include the residual-based loss ([RB-loss](https://arxiv.org/abs/1711.06396)), residual-cos-based loss (RCB-loss), corner loss ([CN-loss](https://arxiv.org/abs/1712.02294)), partial-bin-based loss ([PBB-loss](https://arxiv.org/abs/1711.08488)), and our full bin-based loss (BB-loss). RCB-loss encodes $$\Delta\theta$$ of RB-loss by $$(cos(\Delta\theta), sin(\Delta\theta))$$ to remove the ambiguity of angle regression. Using 100 proposals from stage-1, recall curves with IoU thresholds 0.5 and 0.7 are shown below. Full bin-based loss function has higher recall and converges faster with both IoU thresholds in comparison to all other loss functions. The PBB-loss achieves similar recall as BB-loss with slow convergence speed. Both PBB-loss and BB-loss have significantly higher recall than other losses. By improving the angle regression targets, the improved RCB-loss shows good recall than RB-loss.


| ![recallcurve]({{ site.baseurl }}/images/recallcurve.png) |
|:--:| 
| *Recall curves with different bounding box regression loss function* |

## Conclusion
PointRCNN is a novel two-stage 3D object detection framework, which uses only raw 3D point cloud as input and achieves accurate and robust 3D object detection performance. 3D proposals are generated directly from point cloud in a bottom-up manner in our proposed stage-1 sub-network. The proposed bin-based localization loss achieves significantly higher recall than other 3D box regression losses. For stage-2 sub-network, point cloud region pooling with optimal $$\eta$$ results in good performance. The refinement of proposals are performed in canonical coordinates by combining global features and local spatial features. The experiments show that PointRCNN performs well in comparison to previous state-of-the-art methods with remarkable margins on KITTI dataset. Many state-of-the-art methods uses multiple sensors to perform 3D object detection, but PointRCNN uses only raw point cloud as input, and performs well in comparison with other methods.

### My own
The method does not perform well for pedestrian detection. The authors of PointRCNN assumes that in autonomous driving scenes, the 3D objects are naturally well-separated. This might not be true for pedestrian class. Generally, due to small size of pedestrians, they have less foreground points. Also, the 3D grouth truth boxes for pedestrian class might be overlapping, as they are not well separated in general. Eventually, PointRCNN could not get good performance for 3D pedestrian detection. Also, PointRCNN was never tested for multi-class 3D detection. The problem with multi-class detection is that, when objects are close-by, the 3D ground-truth boxes might be overlapping and include noisy foreground points of other objects. So, PointRCNN might not perform well for multi-class 3D detection.


## Practical Challenges

Detecting multiple classes for a autonomous driving scene is essential rather than detecting single class. But, for multi-class 3D detection, we encounter class imbalance problem. As cars are most prevalent on roads, the model might train well for cars, when compared to other 3D objects. Focal loss should be used to remove class imbalance for multi-class 3D object detection. 

For achieving accurate and robust 3D object detections, autonomous vehicles are usually equipped with different sensors such as cameras, LiDARs and Radars. In order to exploit their complementary properties, multiple sensing modalities can be fused.
For example, camera images can provide detailed texture information of a scene, but cannot directly provide depth information. On the other hand, LiDARs provide accurate depth information of the surroundings through 3D points. But, they cannot capture the fine texture information of objects, and 3D points become sparse with far-away objects. Radars are good, but they have low resolution and classifying objects becomes a challenge. We need to fuse data from different sensors to achieve reliable autonomous driving. But, fusing of different data from sensors is a big challenge. [Deep Multi-modal approach](https://arxiv.org/abs/1902.07830) has been developed to carry out 3D object detection by fusing camera images, LIDAR point clouds and Radars. Even if one of the sensor fails, the 3D objects can be detected using other sensors. Thus, reliable autonomous driving can be achieved by fusing different data from sensors through the deep multi-modal approach.

> written on $$1^{st}$$ January, 2020.
