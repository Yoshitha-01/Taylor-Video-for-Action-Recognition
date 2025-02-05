# Taylor-Video-for-Action-Recognition

# Application to Action Recognition

To use Taylor videos for action recognition, follow these simple steps:

## 1. Extract Taylor Video from an RGB Video

To extract Taylor videos from an RGB video, run the `index.py` script.  
**Note:** The Taylor skeleton sequence computation uses only the displacement concept for extraction.

## 2. Action Recognition with Taylor Videos

To utilize Taylor videos for action recognition, follow the setup instructions provided by the [MMACTION2 documentation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html).

## 3. Datasets

You can download the following datasets:

- **CATER Dataset**:  
  [CATER Dataset](https://cmu.app.box.com/s/w1baekogh29fgu3zg7gr6k446xdalgf2/file/593481427220)
  
- **HMDB-51 Dataset**:  
  [HMDB-51 Dataset](https://www.kaggle.com/datasets/easonlll/hmdb51)



## 4. Trained Models

We have trained 6 models using 3 different datasets:

- `i3d.py`
- `tsm_model.py`
- `tsn_model.py`
- `timesformer_model.py`
- `swim_transformer_model.py`
- `r2plus1d_model.py`

## 5. Visualisation

For visualizations in the form of videos (Taylor videos with different numbers of terms), please refer to the folders provided in the repository. The pixel values are scaled for better visualization purposes. Below, we present some visualizations and comparisons of Taylor frames.

### 5.1 Motion Strengths and Directions

Taylor frames indicate motion strengths and directions. Each channel of the Taylor frame represents a motion concept with positive and negative values indicating motion directions (0 for static pixels). Velocity and acceleration channels are computed per video temporal block, capturing relative motion directions from the initial frame.

### 5.2 Redundancy Removal

Taylor videos remove redundancy, such as static backgrounds, unstable pixels, watermarks, and captions.

