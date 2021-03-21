# PP-LinkNet: Improving Semantic Segmentation of High Resolution Satellite Imagery with Multi-stage Training

The repository contains codes of [PP-LinkNet](https://arxiv.org/abs/2010.06932). If you feel this repository useful, please cite our paper:

```
@inproceedings{AnTran_ACMMM_2020,
author = {Tran, An and Zonoozi, Ali and Varadarajan, Jagannadan and Kruppa, Hannes},
title = {PP-LinkNet: Improving Semantic Segmentation of High Resolution Satellite Imagery with Multi-Stage Training},
year = {2020},
isbn = {9781450381550},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3423323.3423407},
doi = {10.1145/3423323.3423407},
abstract = {Road network and building footprint extraction is essential for many applications such as updating maps, traffic regulations, city planning, ride-hailing, disaster response etc. Mapping road networks is currently both expensive and labor-intensive. Recently, improvements in image segmentation through the application of deep neural networks has shown promising results in extracting road segments from large scale, high resolution satellite imagery. However, significant challenges remain due to lack of enough labeled training data needed to build models for industry grade applications. In this paper, we propose a two-stage transfer learning technique to improve robustness of semantic segmentation for satellite images that leverages noisy pseudo ground truth masks obtained automatically (without human labor) from crowd-sourced OpenStreetMap (OSM) data. We further propose Pyramid Pooling-LinkNet (PP-LinkNet), an improved deep neural network for segmentation that uses focal loss, poly learning rate, and context module. We demonstrate the strengths of our approach through evaluations done on three popular datasets over two tasks, namely, road extraction and building foot-print detection. Specifically, we obtain 78.19% meanIoU on SpaceNet building footprint dataset, 67.03% and 77.11% on the road topology metric on SpaceNet and DeepGlobe road extraction dataset, respectively.},
booktitle = {Proceedings of the 2nd Workshop on Structuring and Understanding of Multimedia HeritAge Contents},
pages = {57–64},
numpages = {8},
keywords = {pp-linknet, remote sensing, building footprint, mapping application, multi-stage training, road network, hyperspectral imaging, transfer learning},
location = {Seattle, WA, USA},
series = {SUMAC'20}
}
```

## News & Updates

#### Mar 20, 2021
- [x] Commit the codes of rasterization using OpenCV
- [x] Commit the codes of ALPS scores reimplemented from SpaceNet competitions

-------------
# Usage Guide

## ALPS scores
The Python ALPS scores from [CosmiQ](https://github.com/CosmiQ/apls) does not work well. We decide that we port the Java implementation from SpaceNet competitions website into the following repo:
```
https://github.com/antran89/road_visualizer
```

## Rasterization of OSM data into pseudo (noisy) ground truth image

The code to rasterize OSM road networks into images using OpenCV simple like this. For more use cases, please consult the attached notebook.

```
def rasterize_osm_using_opencv(input_path, osm_out_folder):
    """Rasterize OSM road network using opencv, and write it to the result folder.

    Parameters
    ----------
    input_image: string
        Jpg input image
    osm_out_folder : string
        OSM output folder

    Returns
    -------
    None
    """
    img_name = os.path.basename(input_path)
    out_file_name = '%s_mask.png' % (img_name[:-8])
    out_file = os.path.join(osm_out_folder, out_file_name)
    if os.path.exists(out_file):
        return
    left, top = get_lon_lat_from_image_name(img_name, transform_folder)
    right, bottom = get_lon_lat_from_image_name(img_name, transform_folder, r=1024, c=1024)
    epsilon = 1e-3
    try:
        G = ox.graph_from_bbox(north=top+epsilon, south=bottom-epsilon, west=left-epsilon, east=right+epsilon, network_type='all_private',
                    simplify=False, retain_all=True, truncate_by_edge=False)
    except ox.EmptyOverpassResponse:
        cv2.imwrite(out_file, np.zeros((1024, 1024)))
        return
    mask = rasterize_osm_road_network(img_name, G, num_lanes_profile, lane_width=8)
    cv2.imwrite(out_file, mask)
    return
```
