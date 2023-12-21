# EEG_Image_decode
Using vision-language models to decode natural image perception from non-invasive brain recordings.

## Dataset
You can get the data set used in this project through the Baidu web disk link [here](https://pan.baidu.com/s/1-1hgpoi4nereLVqE4ylE_g?pwd=nid5):
Here we provide our dataset used in the source of paper:</br>
"[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)]".</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.



Another similar data set provides a reference:</br>"[Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams](https://www.nature.com/articles/s41597-021-01102-7)".</br>
Tijl Grootswagers, Ivy Zhou, Amanda K. Robinson, Martin N. Hebart & Thomas A. Carlson .


## Data availability
The raw and preprocessed EEG dataset, the training and test images and the DNN feature maps are available on [osf](https://osf.io/3jk45/). The ILSVRC-2012 validation and test images can be found on [ImageNet](https://www.image-net.org/download.php). To run the code, the data must be downloaded and placed into the following directories:

- **Raw EEG data:** `../project_directory/eeg_dataset/raw_data/`.
- **Preprocessed EEG data:** `../project_directory/eeg_dataset/preprocessed_data/`.
- **Training and test images; ILSVRC-2012 validation and test images:** `../project_directory/image_set/`.
- **DNN feature maps:** `../project_directory/dnn_feature_maps/pca_feature_maps`.



## Environment setup
1. Cloning and building from source
```
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install .
```
2.If you would like to develop on LAVIS, it is recommended to install in editable mode:
```
pip install -e .
```

## Train 
Please modify your data set path and run:
```
python train_mask_img.py
```
## Test 
Please modify your data set path and run:
```
python sp2.py
```
