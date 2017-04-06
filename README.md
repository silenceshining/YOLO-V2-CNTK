# YOLO-V2-CNTK
YOLO V2 CNTK

## Gernal Things

This is my current try to implement YOLO V2 with CNTK. It doesn't work yet.

Some of the concepts i used are not very efficent and need to be changed.
Furthermore, not all of the concepts the reach YOLO V2 mAP have been implemented yet.

## Dataset and Encoding

To test what i'm doing i use VOC2007. I converted the encoding for the boxes to an format for the CTFReader of cntk.
There are two input files:

### images_map_voc

which looks like this

    0	_VOC2007\JPEGImages\000005.jpg	0
    1	_VOC2007\JPEGImages\000007.jpg	0
    2	_VOC2007\JPEGImages\000009.jpg	0
    .
    .
    .
    
and maps the pictures.


### rois_map_voc

    0 |rois 15 0.587 0.7333333 0.122 0.3413333 15 0.418 0.848 0.176 0.288 ...
    1 |rois 11 0.641 0.5705706 0.718 0.8408408 -1 0 0 0 0 ...
    .
    . 
    .

which encodes the rois and classes. In one row there 100 entries. This is due to the fact that a box/roi is described through
    
    (class number, x, y, w, h) => 5 entries per box
    (x,y) is the center point of each box and NOT the top left corner

and for one image there should be no more that 20 boxes. So we have 5 * 20 = 100. I did this because as far as i know the CTF reader needs a fixed row length.

If for example there only exists one box, like in image index 1 above, all other 19 class numbers are set to -1 to indicate that. 

### Class Numbers

The class numbers are encoded from 0 to 19 in the order:

     person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, diningtable, pottedplant, sofa, tvmonitor
     
     
## Info

In time i will update and impove the descriptions of what's happening. As it is right know a lot of information about what i'm doing can be found in the code docs. I'm new to Python and Deep Learning so don't kill me if something's not as it should be. As i wrote above. I'm well aware that there are things i can do better and i will in time. ;D
