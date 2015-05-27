# gSLIC real-time super-pixel segmentation software
## Notes of 2.0
- Fully re-factored code.
- Works for any size / number of super pixels
- 4.5ms@640x480, 13ms@1280x960, 25ms@1920x1080 image
- Multi-platform supported

## Requirements:
- CUDA: required
- OpenCV: optional (only if you want to run the demo, opencv is used for reading camera input) 

## To run demo:
```
mkdir build
cd build
cmake ../
make
```
## Paper to cite:
If you use this code for your research, please kindly cite:
```
@article{gSLIC_2011,
	author = {Carl Yuheng Ren and Ian Reid},
	title = "{gSLIC: a real-time implementation of SLIC superpixel segmentation}",
	journal = {Oxford University Technical Report},
	year = 2011
}
```
(this is a old version of the repot for now, will be updated a bit later)

MIT licence
