# HDMapNet_devkit

Devkit for HDMapNet.

**HDMapNet: A Local Semantic Map Learning and Evaluation Framework**

[Qi Li](https://liqi17thu.github.io/), [Yue Wang](https://people.csail.mit.edu/yuewang/), [Yilun Wang](https://scholar.google.com.hk/citations?user=nUyTDosAAAAJ&hl=en/), [Hang Zhao](http://people.csail.mit.edu/hangzhao/)

**[[Paper](https://arxiv.org/abs/2107.06307)] [[Project Page](https://tsinghua-mars-lab.github.io/HDMapNet/)] [[5-min video](https://www.youtube.com/watch?v=AJ-rToTN8y8)]**

**Abstract:**
Estimating local semantics from sensory inputs is a central component for high-definition map constructions in autonomous driving. However, traditional pipelines require a vast amount of human efforts and resources in annotating and maintaining the semantics in the map, which limits its scalability. In this paper, we introduce the problem of local semantic map learning, which dynamically constructs the vectorized semantics based on onboard sensor observations. Meanwhile, we introduce a local semantic map learning method, dubbed HDMapNet. HDMapNet encodes image features from surrounding cameras and/or point clouds from LiDAR, and predicts vectorized map elements in the bird's-eye view. We benchmark HDMapNet on nuScenes dataset and show that in all settings, it performs better than baseline methods. Of note, our fusion-based HDMapNet outperforms existing methods by more than 50% in all metrics. In addition, we develop semantic-level and instance-level metrics to evaluate the map learning performance. Finally, we showcase our method is capable of predicting a locally consistent map. By introducing the method and metrics, we invite the community to study this novel map learning problem. Code and evaluation kit will be released to facilitate future development.

**Questions/Requests:** 
Please file an [issue](https://github.com/Tsinghua-MARS-Lab/HDMapNet-dev/issues) or email me at liqi17thu@gmail.com.

### Preparation
1. Download  [nuScenes dataset](https://www.nuscenes.org/) and put it to `dataset/` folder.

2. Install dependencies by running
```
pip install -r requirement.txt
```

### Vectorization
Run `python vis_label.py ` for demo of vectorized labels. The visualizations are in `dataset/nuScenes/samples/GT`.

### Evaluation
Run `python evaluate.py --result_path [submission file]` for evaluation. The script accepts vectorized or rasterized maps as input. For vectorized map, We firstly rasterize the vectors to map to do evaluation. For rasterized map, you should make sure the line width=1.

Below is the format for vectorized submission:

```
vectorized_submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
        "vector":        true   -- Whether this submission uses vector format.
    },
    "results": {
        sample_token <str>: List[vectorized_line]  -- Maps each sample_token to a list of vectorized lines.
    }
}

vectorized_line {
    "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
    "pts_num":           <int>,            -- Number of points in this line.
    "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
    "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
}
```

For rasterized submission, the format is:

```
rasterized_submisson {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
        "vector":       false   -- Whether this submission uses vector format.
    },
    "results": {
        sample_token <str>: {  -- Maps each sample_token to a list of vectorized lines.
            "map": [<float, (C, H, W)>],         -- Raster map of prediction (C=0: ped; 1: divider 2: boundary). The value indicates the line idx (start from 1).
    	    "confidence_level": Array[float],    -- confidence_level[i] stands for confidence level for i^th line (start from 1). 
        }
    }
}
```

Run `python export_to_json.py` to get a demo of vectorized submission. Run `python export_to_json.py --raster` for rasterized submission.


### Citation
If you found this useful in your research, please consider citing
```
@misc{li2021hdmapnet,
      title={HDMapNet: A Local Semantic Map Learning and Evaluation Framework}, 
      author={Qi Li and Yue Wang and Yilun Wang and Hang Zhao},
      year={2021},
      eprint={2107.06307},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```