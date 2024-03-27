# Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion
Using vision-language models to decode and reconstruct natural image perception from non-invasive brain recordings.



<img src="framework.png" alt="Framework" style="max-width: 100%; height: auto;"/>
Framework of the proposed method.

<img src="encoder.png" alt="Encoder" style="max-width: 100%; height: auto;"/>
Encoder structure.



## Environment setup
You can create a new conda environment and install the required dependencies
```
# use conda environment.yml:
conda env create -f environment.yml
conda activate BCI
#or pip install -U -r requirements.txt

pip install wandb
pip install einops
```
Additional environments needed to run all the code:
```
pip install open_clip_torch
# LAVIS makes it possible to use any published CLIP base model.
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .

pip install transformers==0.27.0
pip install diffusers==0.24.0

#Below are the braindecode installation commands for the most common use cases.
pip install braindecode==0.8.1
```
## Quick training and test 
1.- **ATM_S_insubject_retrieval.py** is provided to learn the training strategy of EEG Encoder and verify it during training. Please modify your data set path and run:
```
cd Retrieval
python ATM_S_insubject_retrieval.py
```
2.- **Generation_measure.ipynb** is provided to learn the Reconstruction strategy of EEG Encoder. Please modify your data set path and run zero-shot on 200 classes test dataset:
```
# 1 step: reconstruct images
Generation_metrics_sub<index>.ipynb
# 2 step: compute metrics
cd Generation/fMRI-reconstruction-NSD/src
Reconstruction_Metrics_ATM.ipynb
```


## Data availability
The raw and preprocessed EEG dataset, the training and test images and the DNN feature maps are available on [osf](https://osf.io/3jk45/).
- **Raw EEG data:** `../project_directory/eeg_dataset/raw_data/`.
- **Preprocessed EEG data:** `../project_directory/eeg_dataset/preprocessed_data/`.
- **Training and test images; ILSVRC-2012 validation and test images:** `../project_directory/image_set/`.
- **DNN feature maps:** `../project_directory/dnn_feature_maps/pca_feature_maps`.

We provide you with preprocessed EEG data and magnetoencephalography data, as well as raw image data at the link. Note that the experimental paradigms of the THINGS-EEG and THINGS-MEG datasets themselves are different, so we provide images and data for both datasets separately.

If the Internet speed is not good, you can also choose to run it yourself using the preprocessed code in the warehouse:
## EEG preprocessing
```
python EEG-preprocessing/preprocessing.py
```
## MEG preprocessing
```
MEG-preprocessing/pre_possess.ipynb
```

Also You can get the data set used in this project through the BaiduNetDisk [link](https://pan.baidu.com/s/1-1hgpoi4nereLVqE4ylE_g?pwd=nid5) to run the code.

## Citations

1.Here we provide our THING-EEG dataset cited in the paper:</br>"[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)]".</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.


2.Another used THINGS-MEG data set provides a reference:</br>"[THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior.](https://elifesciences.org/articles/82580.pdf)".</br> Hebart, Martin N., Oliver Contier, Lina Teichmann, Adam H. Rockter, Charles Y. Zheng, Alexis Kidder, Anna Corriveau, Maryam Vaziri-Pashkam, and Chris I. Baker.

3.Thanks to Y Song et al. for their contribution in data set preprocessing and neural network structure, we refer to their work:</br>"[Decoding Natural Images from EEG for Object Recognition](https://arxiv.org/pdf/2308.13234.pdf)".</br> Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. 

