import cv2
import os
import glob
import pandas as pd
import numpy as np

annotations = pd.read_csv('/home/akhan/annotations.txt',
                          delim_whitespace=True,
                          names=['start', 'end', 'duration', 'class', 'filename'])

annotations['duration'] = 0

renamer = lambda x: x.split('.')[0] + '_crop.mp4'

annotations.filename = annotations.filename.apply(renamer)

for path, subdirs, files in os.walk('/home/akhan/Videos/data/videos'):
    for name in files:
        if name.endswith('crop.mp4'):
            cap = cv2.VideoCapture(os.path.join(path, name))
            fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = np.round(frame_count / fps, 3)
            annotations.loc[annotations['filename'] == name, 'duration'] = np.round(duration, 3)


annotations.to_csv('/home/akhan/ActionDetection-AFSD/thumos_annotations/custom_annotations.csv', header=False)