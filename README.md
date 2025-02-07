<div align="center">

<h2 style="border-bottom: 1px solid lightgray;">Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion</h2>

<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">

<p align="center">
  <a href="#">
  <p align="center">
    <a href='https://arxiv.org/pdf/2403.07721'><img src='http://img.shields.io/badge/Paper-arxiv.2403.07721-B31B1B.svg'></a>
    <a href='https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main'><img src='https://img.shields.io/badge/EEG Image decode-%F0%9F%A4%97%20Hugging%20Face-blue'></a>
  </p>
</p>


</div>

<br/>

</div>

<!-- 
<img src="bs=16_test_acc.png" alt="Framework" style="max-width: 90%; height: auto;"/> -->
<!-- 
<img src="test_acc.png" alt="Framework" style="max-width: 90%; height: auto;"/> -->

<!-- As the training epochs increases, the test set accuracy of different methods. (Top: batchsize is 16. Bottom: batchsize is 1024) -->

<!-- 
<img src="temporal_analysis.png" alt="Framework" style="max-width: 90%; height: auto;"/>
Examples of growing window image reconstruction with 5 different random seeds. -->


<img src="fig-framework.png" alt="Framework" style="max-width: 100%; height: auto;"/>
Framework of our proposed method.




<!--  -->
<img src="fig-genexample.png" alt="fig-genexample" style="max-width: 90%; height: auto;"/>  

Some examples of using EEG to reconstruct stimulus images.


## News:
- [2024/09/26] Our paper is accepted to **NeurIPS 2024**.
- [2024/09/25] We have updated the [arxiv](https://arxiv.org/abs/2403.07721) paper.
- [2024/08/01] Update scripts for training and inference in different tasks.
- [2024/05/19] Update the dataset loading scripts.
- [2024/03/12] The [arxiv](https://arxiv.org/abs/2403.07721) paper is available.


<!-- ## Environment setup -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Environment setup</h2>

Run ``setup.sh`` to quickly create a conda environment that contains the packages necessary to run our scripts; activate the environment with conda activate BCI.




```
. setup.sh
```
You can also create a new conda environment and install the required dependencies by running
```
conda env create -f environment.yml
conda activate BCI

pip install wandb
pip install einops
```
Additional environments needed to run all the code:
```
pip install open_clip_torch

pip install transformers==4.28.0.dev0
pip install diffusers==0.24.0

#Below are the braindecode installation commands for the most common use cases.
pip install braindecode==0.8.1
```
<!-- ## Quick training and test  -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Quick training and test</h2>

If you want to quickly reproduce the results in the paper, please download the relevant ``preprocessed data`` and ``model weights`` from [Hugging Face](https://huggingface.co/datasets/LidongYang/EEG_Image_decode) first.
#### 1.Image Retrieval
We provide the script to learn the training strategy of EEG Encoder and verify it during training. Please modify your data set path and run:
```
cd Retrieval/
python ATMS_retrieval.py --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```
We also provide the script for ``joint subject training``, which aims to train all subjects jointly and test on a specific subject:
```
cd Retrieval/
python ATMS_retrieval_joint_train.py --joint_train --sub sub-01 True --logger True --gpu cuda:0  --output_dir ./outputs/contrast
```

Additionally, replicating the results of other methods (e.g. EEGNetV4) by run
```
cd Retrieval/
contrast_retrieval.py --encoder_type EEGNetv4_Encoder --epochs 30 --batch_size 1024
```

#### 2.Image Reconstruction
We provide quick training and inference scripts for ``clip pipeline`` of visual reconstruction. Please modify your data set path and run zero-shot on 200 classes test dataset:
```
# Train and generate eeg features in Subject 8
cd Generation/
python ATMS_reconstruction.py --insubject True --subjects sub-08 --logger True \
--gpu cuda:0  --output_dir ./outputs/contrast
```

```
# Reconstruct images in Subject 8
Generation_metrics_sub8.ipynb
```

We also provide scripts for image reconstruction combined ``with the low level pipeline``.
```
cd Generation/

# step 1: train vae encoder and then generate low level images
train_vae_latent_512_low_level_no_average.py

# step 2: load low level images and then reconstruct them
1x1024_reconstruct_sdxl.ipynb
```


We provide scripts for caption generation combined ``with the semantic level pipeline``.
```
cd Generation/

# step 1: train feature adapter
image_adapter.ipynb

# step 2: get caption from eeg latent
GIT_caption_batch.ipynb

# step 3: load text prompt and then reconstruct images
1x1024_reconstruct_sdxl.ipynb
```

To evaluate the quality of the reconstructed images, modify the paths of the reconstructed images and the original stimulus images in the notebook and run:
```
#compute metrics, cited from MindEye
Reconstruction_Metrics_ATM.ipynb
```

<!-- ## Data availability -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Data availability</h2>

We provide you with the ``preprocessed EEG`` and ``preprocessed MEG`` data used in our paper at [Hugging Face](https://huggingface.co/datasets/LidongYang/EEG_Image_decode), as well as the raw image data.


Note that the experimental paradigms of the THINGS-EEG and THINGS-MEG datasets themselves are different, so we will provide images and data for the two datasets separately.

You can also download the relevant THINGS-EEG data set and THINGS-MEG data set at osf.io.

The raw and preprocessed EEG dataset, the training and test images are available on [osf](https://osf.io/3jk45/).
- ``Raw EEG data:`` `../project_directory/eeg_dataset/raw_data/`.
- ``Preprocessed EEG data:`` `../project_directory/eeg_dataset/preprocessed_data/`.
- ``Training and test images:`` `../project_directory/image_set/`.


The raw and preprocessed MEG dataset, the training and test images are available on [OpenNEURO](https://openneuro.org/datasets/ds004212/versions/2.0.0).





<!-- ## EEG/MEG preprocessing -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">EEG/MEG preprocessing</h2>


Modify your path and execute the following code to perform the same preprocessing on the raw data as in our experiment:
```
cd EEG-preprocessing/
python EEG-preprocessing/preprocessing.py
```

```
cd MEG-preprocessing/
MEG-preprocessing/pre_possess.ipynb
```
Also You can get the data set used in this project through the BaiduNetDisk [link](https://pan.baidu.com/s/1-1hgpoi4nereLVqE4ylE_g?pwd=nid5) to run the code.

## TODO
- [√] Release retrieval and reconstruction scripts.
- [√] Update training scripts of reconstruction pipeline.
- [ ] Adding validation sets improves performance evaluation accuracy.



<!-- ## Acknowledge -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Acknowledge</h2>

1.Thanks to Y Song et al. for their contribution in data set preprocessing and neural network structure, we refer to their work:</br>"[Decoding Natural Images from EEG for Object Recognition](https://arxiv.org/pdf/2308.13234.pdf)".</br> Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. 

2.We also thank the authors of [SDRecon](https://github.com/yu-takagi/StableDiffusionReconstruction) for providing the codes and the results. Some parts of the training script are based on [MindEye](https://medarc-ai.github.io/mindeye/) and [MindEye2](https://github.com/MedARC-AI/MindEyeV2). Thanks for the awesome research works.

3.Here we provide our THING-EEG dataset cited in the paper:</br>"[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)".</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.


4.Another used THINGS-MEG data set provides a reference:</br>"[THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior.](https://elifesciences.org/articles/82580.pdf)".</br> Hebart, Martin N., Oliver Contier, Lina Teichmann, Adam H. Rockter, Charles Y. Zheng, Alexis Kidder, Anna Corriveau, Maryam Vaziri-Pashkam, and Chris I. Baker.



<!-- ## Citation -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Citation</h2>

```bibtex
@inproceedings{
li2024visual,
title={Visual Decoding and Reconstruction via {EEG} Embeddings with Guided Diffusion},
author={Dongyang Li and Chen Wei and Shiying Li and Jiachen Zou and Quanying Liu},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=RxkcroC8qP}
}

@article{li2024visual,
  title={Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion},
  author={Li, Dongyang and Wei, Chen and Li, Shiying and Zou, Jiachen and Liu, Quanying},
  journal={arXiv preprint arXiv:2403.07721},
  year={2024}
}
```
