## Why DOT Tracking?

The DOT approach defines objects as pairs of coordinates and uses an encoder-decoder architecture to generate precise object masks and coordinates. The tracking process is designed to maintain the continuity of object trajectories over time, addressing challenges such as occlusions and varying object appearances, dealing with large crowds and small objects in drone-captured videos.

## Results

The repository is a supplement to the paper "Improving trajectory continuity in drone-based crowd monitoring using a set of minimal-cost techniques and deep discriminative correlation filters". The method achieves state-of-the-art results on the DroneCrowd and newly introduced UP-COUNT-TRACK datasets. The results are summarized in the table below.

-----

**Table 1: UP-COUNT-TRACK Dataset**

*Tracking results for the UP-COUNT-TRACK dataset, considering proposed improvements. \*Globally-optimal greedy (GOG) algorithm is an offline method.*

| **Method** | **HOTA (↑)** | **T-mAP (↑)** | **T-AP@10 (↑)** | **ID-SW (↓)** | **Tr-MAE (↓)** | **Tr-nMAE (↓)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 0.63 | 37.04 | 39.34 | 3305 | 49.13 ± 117.22 | 0.37 ± 0.34 |
| Baseline + CMC | 0.63 | 37.36 | 39.59 | 3180 | 48.84 ± 114.43 | 0.38 ± 0.32 |
| Baseline + CMC + ALT | 0.64 | 38.68 | 40.90 | 3126 | 44.97 ± 103.25 | 0.33 ± 0.29 |
| Baseline + CMC + ALT + CLS | 0.63 | 38.72 | 40.95 | 2943 | 41.81 ± 96.76 | 0.30 ± 0.26 |
| Baseline + CMC + ALT + CLS + DDCF | **0.63** | **44.35** | **45.88** | **287** | **20.45** ± **44.81** | **0.15** ± **0.11** |
| GOG\* | 0.42 | 36.21 | 37.63 | 1868 | 64.19 ± 110.24 | 0.57 ± 0.36 |

-----

**Table 2: DroneCrowd Dataset**

*Tracking results for the DroneCrowd dataset, considering proposed improvements. \*Globally-optimal greedy (GOG) algorithm is an offline method.*

| **Method** | **HOTA (↑)** | **T-mAP (↑)** | **T-AP@10 (↑)** | **ID-SW (↓)** | **Tr-MAE (↓)** | **Tr-nMAE (↓)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 0.53 | 46.27 | 49.99 | 6290 | 57.47 ± 37.13 | 0.32 ± 0.15 |
| Baseline + CMC | 0.53 | 46.87 | 50.37 | 6231 | 55.83 ± 35.53 | 0.31 ± 0.14 |
| Baseline + CMC + ALT | 0.52 | 47.44 | 51.13 | 6771 | 52.90 ± 31.37 | 0.30 ± 0.13 |
| Baseline + CMC + ALT + CLS | 0.52 | 47.32 | 50.88 | 6645 | 52.10 ± 30.36 | 0.30 ± 0.14 |
| Baseline + CMC + ALT + CLS + DDCF | **0.54** | **54.59** | **57.04** | **388** | **37.60** ± **25.78** | **0.23** ± **0.16** |
| GOG\* | 0.51 | 50.47 | 53.21 | 1326 | 38.10 ± 31.14 | 0.24 ± 0.17 |


## Dataset

* The DroneCrowd dataset can be downloaded from [here](https://github.com/VisDrone/DroneCrowd/tree/master#dronecrowd-full-version)

* The UP-COUNT-TRACK dataset is available [here](https://up-count.github.io/tracking)


## Usage

### Clone the repository (including submodules)

```
git clone --recurse-submodules https://github.com/up-count/uav-dot-track.git
```

### Requirements

Install requirements with: 

```bash
pip install -r requirements.txt
```

```bash
pip install -r ./uavdot/requirements.txt
```

```bash
pip install -r ./eval/TrackEval/requirements.txt
```


### Checkpoints

Download the checkpoints from the links below and place them in the `./checkpoints` directory.

| **Model** | **Dataset** | **L-mAP** | **L-AP@10** |   **Link**   |
|:---------:|:-----------:|:---------:|:-----------:|:------------:|
|    DOT    |  Dronecrowd |   47.63   |   53.37     | [download](https://drive.google.com/file/d/1jHZ2_85kS4tdG5Qbq3Jn0Xpjs8So6mwK/view?usp=sharing) |
|  DOT + PD |  Dronecrowd |   51.00   |   57.06     | [download](https://drive.google.com/file/d/1wYa01jGYfrAun3SfxuWzcKAni3hzBOMV/view?usp=sharing) |
|    DOT    |   UP-COUNT  |   60.66   |   69.07     | [download](https://drive.google.com/file/d/16MghcySpCxS0OxJzTJyRLr3AZKZ7cP0w/view?usp=sharing) |
|  DOT + PD |   UP-COUNT  |   66.49   |   75.46     | [download](https://drive.google.com/file/d/1K-SkfIPbivnOw7atjRQHW11Bt0-bhcKi/view?usp=sharing) |

Alternatively, you can find checkpoints [here](https://chmura.put.poznan.pl/s/SGlQ3OFfgRJr86g).



### Usage

To run the algorithm, use the `main.py` script. 

```bash
Usage: main.py [OPTIONS]

Options:
  --name TEXT                     Name of the experiment
  --dataset [dronecrowd|upcount]  Dataset to use  [required]
  --video TEXT                    Path to the video file  [required]
  --task [vid|viz|pred|none|det]  Task to perform, can be multiple  [required]
  --device [cpu|cuda]             Device to run the model on
  --cache-det                     Cache detections
  --cache-dir TEXT                Path to the cache directory
  --track-max-age INTEGER         Max age of the track
  --track-min-hits INTEGER        Min hits of the track
  --track-cmc-flow                Use optical flow
  --track-use-alt                 Use altitude information
  --track-use-pointflow           Use pointflow for tracking
  --track-use-add-cls             Use additional classification for tracking
  --track-use-cutoff              Use dynamic cut-off for tracking
  --help                          Show this message and exit.
```

Five tasks are available:

- `vid` - process the video and save the output video with tracking results
- `viz` - process the video, displaying the results in a window
- `pred` - process the video and save the tracking results in a text file (required for evaluation)
- `none` - process the video without any output (useful for debugging)
- `det` - run only the detection model

Dataset options:

- `dronecrowd` - use the model trained on the DroneCrowd dataset
- `upcount` - use the model trained on the UP-COUNT-TRACK dataset


Example usage:

```bash
python3 main.py --name test_run --dataset upcount --video path/to/video.mp4
```

### Evaluation

Before running the evaluation, check scripts at './scripts' to run all videos in the dataset and save the results in the required format.

1. Run the tracking algorithm with the `pred` task to generate tracking results in a text file.

```bash
python3 main.py --name eval_run --dataset upcount --video path/to/video.mp4 --task pred
```

2. Use the `TrackEval` library to evaluate the tracking results against the ground truth.

```bash
python3 eval/track_eval.py --dataset <dataset_name>
```

