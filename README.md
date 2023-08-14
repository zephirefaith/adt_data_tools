Output of dataloader is a dictionary with following keys:  
- "rgb": Numpy array of the image from RGB camera
- "segmentation": Numpy array of the ground-truth segmentation image
- "pose": Sophus SE3 object specifying left IMU's position wrt world
- "transformed pose": 4D pose matrix of RGB camera wrt world
- "2dbbox", "3dbbox": Annotated 2D and 3D bounding boxes, respectively, of all objects in current frame indexed by object's instance ID
- "intrinsics": A [CameraCalibration](https://github.com/facebookresearch/projectaria_tools/blob/2daefbe31345bda88147b4a59e9d162910c915d4/core/calibration/CameraCalibration.h) object with helpful member functions for projecting/unprojecting between pixels and 3D positions 
