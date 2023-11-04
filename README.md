## Improving the precision and accuracy of animal population estimates with aerial image object detection

Aerial images with 4305 bounding boxes on zebra, giraffe, and elephants

Eikelboom JA, Wind J, van de Ven E, Kenana LM, Schroder B, de Knegt HJ, van Langevelde F, Prins HH. Improving the precision and accuracy of animal population estimates with aerial image object detection. Methods in Ecology and Evolution. 2019 Nov;10(11):1875-87.

**The dataset is available at** [https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1](https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1)

## Reference

If you use this dataset, please cite the following paper:
https://doi.org/10.1111/2041-210X.13277

## Preprocessing

### Tiling images

The authors of the paper have provided [ImageMagick](https://imagemagick.org/script/download.php) scripts  to preprocess the training images. They should be run on the `train` folder in the following order:
1. `tiles.bat` - cut the raw images into training tiles. Each images is divided into $7 \times 6$ tiles with $200$ pixels overlap.
    > [!Note]
    > The raw images will be still inside the train folder. To delete those, run `remove_raw.py`
2. `transform.bat` - mirror the training tiles. The transformed images are saved inside the `train\MVER` folder. If it doesn't work, try to create the `MVER` folder manually
    > [!Note]
    > If we save transformed images directly under the `train` folder the script will loop infinitely. Hence we have save them in a distinct subfolder `MVER` then move it back to `train` folder
3. `cutoff.bat` - script to cut off the training tiles, so that the partially covered bounding boxes were removed as much as possible.
    > [!Important]
    > Make sure to move all images inside `MVER` folder to directly inside `train` folder (from `train\MVER` to `train`)

### Convert CSV annotations to YOLOv8 format

The annotations of the dataset are in RetinaNet CSV format:
- `annotation_images.csv` is the raw annotations file.
- `annotation_test.csv, annotation_train.csv, annotation_train.csv` are the annotations that were used for testing, training, and validation respectively.

**To convert them to YOLOv8 format, run the `convertformat.py` script.**
The script will create a new folder `labels` inside each `train`, `test`, and `val` folder and save the annotations in YOLOv8 format
```
# run the script with both train, test, and validation annotations in 1 command
py convertformat.py annotations_test.csv annotations_train.csv annotations_val.csv

# default class names are Zebra, Giraffe, and Elephant or specify the class names by a list
py convertformat.py annotations_test.csv annotations_train.csv annotations_val.csv Zebra,Giraffe,Elephant

# or run the script separately for each annotation file
py convertformat.py annotations_test.csv
py convertformat.py annotations_train.csv
py convertformat.py annotations_val.csv

```
> [!IMPORTANT] 
> YOLO locates labels automatically for each image by replacing the last instance of `/images` in each images path with  `/labels`. Organize the dataset structure as follows (`images` folder contains the images and `labels` folder contains the annotations files):
```
data
├── train
│   ├── images
│   ├── labels
├── test
│   ├── images
│   └── labels
└── val
    ├── images
    └── labels
``` 
### Add YAML file
YOLOv8 requires a YAML file to train the model. The file should contain the following:
```
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: 
  0: Zebra
  1: Giraffe
  2: Elephant
```
Save the file as `data.yaml` and place it in the `data` folder. 

Or you can run the `generate_yaml.py` script to generate the YAML file by specifying the number of classes and the class names:
```
py generate_yaml.py Zebra,Giraffe,Elephant
```

## Training with YOLOv8
Training the model with YOLOv8 is straightforward.
1. Install Ultralytics via pip
```pip install ultralytics```
or clone the repository
```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
python setup.py install
```
2. Run the training script, refer to [YOLOv8 Docs](https://docs.ultralytics.com/modes/train/) for more details.
```
yolo task=detect mode=train model=yolov8n.pt data={path to data.yaml} epochs=100 imgsz=640 batch=42 device=-1 plots=True
```

