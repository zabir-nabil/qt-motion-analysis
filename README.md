# qt-motion-analysis
Pose estimation with descriptive analysis

### Download pre-trained weight

https://drive.google.com/file/d/1n-H_cvTHNldZuz08EE62WiVtqqXzemKq/view?usp=sharing

### Install

`pip install -r requirements.txt`

### Running

If you have only one webcam - 

`python main.py`

If you have more than one webcam - 

`python main.py i`

Here, `i` is the webcam index.

### Logs

Processed frames are saved in logs folder. The groundtruth data should be saved at groundtruth folder from logs folder (.npy files). Similarity between two poses can be estimated with error (l1 distance for coordinates).

### Demo

#### Window 1 (after running main.py)

<p align="center">
  <img src="gui_1.png"/>
</p>

#### Window 2 (after clicking start webcam)
 
<p align="center">
  <img src="gui_2.png"/>
</p>

#### Window 3 (after processing frame)

<p align="center">
  <img src="gui_3.png"/>
</p>
