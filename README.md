ofco
====

This package provides a pure python implementation of the optical flow component of the motion correction algorithm described in Chen\*, Hermans\* et al., *Nature Communications*, 2018.

Installation
------------
This package can be installed using pip.
Navigate to this directory in the command line using `cd` and enter the flowing command:
```
pip install -e .
```

Usage
-----
The preferred way to use ofco is via the command line or in a python script.

It can be launched in the command line with the following command.
```
ofco stack1.tif stack2.tif output
```
where `stack1.tif` is the path to the stack used for motion estimation and `stack2.tif`
is the path to a second stack that is warped with the same displacement vector field.
The output is saved in the directory `output` as files `warped1.tif` and `warped2.tif`.
For more command line options see the output of `ofco -h`.

Alternatively, ofco can be incorporated in a python analysis pipeline as follows.
```
from skimage import io
from ofco import motion_compensate
from ofco.utils import default_parameters

stack1 = io.imread("path_to_stack1.tif")
stack2 = io.imread("path_to_stack2.tif")

param = default_parameters()
frames = range(2)

motion_compensate(stack1, stack2, 'warped1.tif', 'warped2.tif', frames, param)
```
For more information check the documentation.

The last possibility is to run the main python file directly with the following command:
```
python -m ofco.main stack1.tif stack2.tif output
```
This provides the same options as the command line interface mentioned above.
