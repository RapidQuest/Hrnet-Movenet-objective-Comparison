# Hrnet-Movenet-objective-Comparison
## Quick Run :
To get the objective comparison run:
```
   !python compare.py /path/to/test_directory /path/to/movenet_lite_model
   ```
Sample Output:
```
   Movenet lags hr_net by 10%
   ```
Test directory tree should look like this:

   ```
  Test 
   ├── images
   ├── labels
  
   ```
Labels should be in json format. Sample below:
- It should consist 17 coco keypoints.
- The first item in list should be x numpy co-ordinate and second; y numpy co-ordinate.

  ```
  {'0':[23,44],'1':[34,66].......'16':[98,45]}
   ```
## Metrics used for comparison:
- percentage ratio of normalised euclidean distance across all keypoints and samples with a distance threshold(default = 0.5).
## Model :
- By default it compares hrnet w32 with heatmap size [64,64] trained on coco dataset with movenet lightning. 
## Scripts used from [official hrnet repository](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch):
- [pose_hrnet.py](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/lib/models/pose_hrnet.py) to build hrnet model.
  ```
  Modifications:
  Applied sigmoid function to output of hrnet model to make use of keypoint threshold.
  ```
- [transforms.py](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/lib/utils/transforms.py) to perform affine transformations.
- [inference.py](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/lib/core/inference.py) to derive x,y coordinatesv w.r.t to original resolution of detected keypoints above threshold.
