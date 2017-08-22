
** Vehicle Detection Project **

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output/car_non_car.png
[image2]: ./output/car_no_car_hog.png
[image3]: ./output/all_size_search_window.png
[image4]: ./output/all_size_with_classfier.png
[image41]: ./output/all_size_with_classfier1.png
[image42]: ./output/all_size_with_classfier2.png
[image43]: ./output/all_size_with_classfier3.png
[image51]: ./output/vehicle_boxes_with_heatmap1.png
[image52]: ./output/vehicle_boxes_with_heatmap2.png
[image53]: ./output/vehicle_boxes_with_heatmap3.png
[image54]: ./output/vehicle_boxes_with_heatmap4.png
[image6]: ./examples/labels_map.png
[image7]: ./output/last_frame_with_boxes.jpg
[video1]: ./output.mp4

Here is the link to my [project code](https://github.com/rnaidu02/CarND-Vehicle-Detect-Tracking/blob/master/Vehicle-Detection-Tracking.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook within a function called `get_hog_features()` line # 16 - 34. (https://github.com/rnaidu02/CarND-Vehicle-Detect-Tracking/blob/master/Vehicle-Detection-Tracking.ipynb) .  

I started by reading in all the `vehicle` and `non-vehicle` images from the KITTI image data set that is provided in the lesson.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented with 8, 9 orientation for the HOG parameters and didn't see much difference. I also experimented with RGB, LUV, and YUV color space parameters and settled with RGB. Also I have tried with HOG color space of 0, and ALL and settled with 0, as it gave me the less number of parameters for the SVM classifier (Too many parameters may cause over fitting with the training data).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I have trained the linear SVM using `LinearSVC() with default parameters. The code for the training is available in cell #4 of the Python notebook.

First, I have extracted features for both car and non-car images with the following types
* color histogram features: extracted histogram for each color channel separately and combined them into a list
* spatial bin features: bin the three channels of RGB
* hog features for channel 0: Extract hog features channel 0

Then normalize the values of these three types to make sure that the one type of data is not dominating the input to the classifier

Then the combined data is split into 80% training and 20% test/validation data.

The resulting classifier has about 2628 features vectors in it and it resulted in a test/validation accuracy of 0.9761

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After looking at the video and the test video, the road is occupying half of the screen (split horizontal) and part of the with towards right only. For this reason I've identified the search windows for detecting the
vehicles within 400 to 550 pixels in y direction and 400 to the end in horizontal direction. Here are the identified windows for window sizes of 64x64, 96x96, 128x128, 144x144 with a overlap of 0.75. I chose these values based on the sizes of objects I want to identify.

Here is the image that resulted with the identified boxes.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using R 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4] ![alt text][image41]
![alt text][image42] ![alt text][image43]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)

The cars are identified most of the time within the video. At couple of frames it showed false positives (May be in less than 10 frames out of 1260).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Mo code for filtering out the false positives out of the search windows from frame to frame within the video is done in two stages:

* Heat Map and Labels: With each window sizes of search windows find the resulting windows from the classifier and create a heatmap of the windows overlap for each of the pixels within the image. Using a threshold value filter out the windows with less overlap/heatmap as false positives and ignore them. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code for this logic is in cell #13 of the python notebook.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are four frames and their corresponding heatmaps:

![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]

* History of last 6 frames with the video: After applying the heatmap and labels to the bounding boxes to filter out the false positive detection, I found many instances of bounding boxes appear with the video where there is no vehicles present. To overcome this issue, I have kept a track of last 6 frames heat map within the buffer to see how the next frame is similar to the last 6 frames that are stored. Each Stored frame comparison to the current frame is given a weightage based on how recent the stored frame is. The frame will get a weightage factor of 6 as opposed to last 6th frame will get a weightage of 0.5. The similarity between the frames is determined by the similarity of the size of the boxes (both width and height within a tolerance of 20 pixels), distance between the center of the boxes (to determine which box correspond to which box in the next frame). The code for this logic is available in function `check_against_prev_boxes()` within code cell #13 inside python notebook.

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially I was focusing on using `find_windows()` also for the bounding box identification along with `search_windows()` function. However I had doubts on the identified bounding boxes and there are many false positives in it.

I have tried to combine the advanced lane detection and vehicle detection, and this resulted in many false positive bounding box identification as opposed to very little false positives when only the `project_video.mp4` is provided as input.

Here are the things that can be done to improve the robustness of the pipeline.
* Accuracy improvements: To improve the classifier accuracy, data augmentation can be done to get more training data. Experimenting with `linearSVC()` params can also improve the accuracy. On top of this, having more experimentation on the search window sizes, HOG channel selection, color space selection and find optimum params based on the least false positives. Also the logic to eliminate the false positives using the heat maps of past 6 frames of the current frame logic also can improved quite a bit. I used very basic logic to eliminate the boxes that are not relevant.

* Performance of algorithm: Currently the pipeline takes long time to process the 50 sec video. May places the algorithms can be improved to reduce the time. Current pipeline is not implemented focusing on this.

* Structural improvements for the code: Current python notebook code doesn't store the svm params in persistent form. Having the params stored in a pickle file will result in the cleanliness of the code and usability of the dependent functions.

I would like to use the time in between now and the next term start to play with the Accuracy improvements so that I get better understanding of the CV features that can be used in autonomous driving use cases.
