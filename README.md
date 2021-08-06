# Hrnet-Movenet-objective-Comparison
## Quick Run :
To get the objective comparison run:
```
   !python compare.py /path/to/test_directory /path/to/movenet_lite_model
   ```
Sample Output:
```
   Movenet leads hr_net by 10%
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
