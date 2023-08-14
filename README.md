Output of dataloader is a dictionary with following keys and results in a tuple. The first element is data as described below and 2nd element is dt b/w queried timestamp and data's timestamp:  
- "rgb": Numpy array of the image from RGB camera
- "segmentation": Numpy array of the ground-truth segmentation image
- "pose": [Aria 3D Pose object](https://github.com/facebookresearch/projectaria_tools/blob/caa23d4bb14a6107d9c0f46d76b7a84e6a53cc71/projects/AriaDigitalTwinDatasetTools/data_provider/AriaDigitalTwinDataTypes.h#L90) specifying left IMU's position wrt world
- "transformed pose": 4D pose matrix of RGB camera wrt world
- "2dbbox", "3dbbox": Annotated 2D and 3D bounding boxes, respectively, of all objects in current frame indexed by object's instance ID
- "intrinsics": A [CameraCalibration](https://github.com/facebookresearch/projectaria_tools/blob/2daefbe31345bda88147b4a59e9d162910c915d4/core/calibration/CameraCalibration.h) object associated with RGB camera with helpful member functions for projecting/unprojecting between pixels and 3D positions 
