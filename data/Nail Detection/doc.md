# This part of the code serves as the nail detection. The default YOLO dataset contains labels of the nail. We don't want YOLO to classify nail feature. We just want YOlO for object detection. CNN Models will handle the classification.

Ran this linux script to change all the labels to nail only

```
find labels -type f -name "*.txt" -exec sed -i 's/^[0-9]\+/0/' {} \;
```
