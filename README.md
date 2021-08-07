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
  ```
  Modifications:
  Implemented below function to process movenet output :
  ```
  ```python
  def get_processed_predictions(output,keypoint_thr,frame_batch):
   batch_size = output.shape[0]
   num_of_keypoints = output.shape[2]
   output[:,:,:,[0,1]] = output[:,:,:,[1,0]]
   processed_coordinates = output[:,0,:,:2]*np.tile(frame_batch.reshape((frame_batch.shape[0],1,frame_batch.shape[1])),(1,17,1))
   max_vals = output[:,0,:,2].reshape((batch_size,num_of_keypoints,1))
   pred_mask = np.tile(np.greater(maxvals, keypoint_thr), (1, 1, 2))
   pred_mask = pred_mask.astype(np.float32)
   processed_coordinates *= pred_mask
   return processed_coordinates
  ```
- [evaluate.py](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/lib/core/evaluate.py) to calculate accuracy of both models.
  ```
  Modifications:
  1. Added utilities to enable movenet evaluation.
  2. replaced get_max_predictions() function with get_final_predictions() for hrnet evaluation to obtain coordinates w.r.t orignal image  resolution. The former method obtained x,y coordinates limited to heatmap range only. For real time evaluation this may not be important. So latter method seems to be useful.
  ```
