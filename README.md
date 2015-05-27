## gSLIC real-time super-pixel segmentation software
# Note of 2.0
- Fully re-factored code.
- Works for any size / number of super pixels
- 4.5ms@640x480, 13ms@1280x960, 25ms@1920x1080 image
- Multi-platform supported
# Requirements:
- CUDA REQIRED
- OpenCV optional (only if you want to run the demo, opencv is used for reading camera input) 
# To run demo:
```
mkdir build
cd build
cmake ../
make
```

MIT licence
