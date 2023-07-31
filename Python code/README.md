
## Vocal isolation using U-Net architecture

<p align="center">
  <img src="../images/U-net1.png" width=500px />
</p>

image reference by :
[`[1]`](#1--jansson-andreas-et-al-singing-voice-separation-with-deep-u-net-convolutional-networks-2017)

## What you expect to do with this modeul

1. Generate your own dataset for training

<p align="center">
  <img src="../images/gen_dataset.gif" width=350px />
</p>


2. Build U-Net architecture

<p align="center">
  <img src="../images/gen_unet.gif" width=250px >
</p>

3. Save your memory while training

<p align="center">
  <img src="../images/TrainGenerator2.gif" width=450px >
</p>


4. Isolate multiple musics as vocal with once

<p align="center">
  <img src="../images/convert_pred.gif" width=450px >
</p>


## Module example with `Jupyter Notebook`
1. [`How to use utils`](./util_example.ipynb)
2. [`How to use models`](./model_example.ipynb)

## Module example with `Markdown`
1. [`How to use utils`](./util_example.md)
2. [`How to use models`](./model_example.md)

## Dependency
[![Static Badge](https://img.shields.io/badge/numpy-1.23.5-blue?label=numpy&labelColor=blue&color=black)
](https://github.com/numpy/numpy)
[![Static Badge](https://img.shields.io/badge/ipython-8.12.0-black?label=ipython&labelColor=%23FFA500)
](https://ipython.org/)
[![Static Badge](https://img.shields.io/badge/librosa-0.10.0.post2-red?label=librosa&labelColor=red&color=black)
](https://github.com/librosa/librosa)
[![Static Badge](https://img.shields.io/badge/tensorflow-2.12.0-red?label=tensorflow&labelColor=orange&color=black)
](https://github.com/tensorflow/tensorflow)
[![Static Badge](https://img.shields.io/badge/keras-2.12.0-red?label=keras&labelColor=%23FF0000&color=black)
](https://github.com/keras-team/keras)

## Environment
[![Static Badge](https://img.shields.io/badge/Python-3.9.16-blue?label=Python&labelColor=blue&color=black)
](https://www.python.org/)
![Static Badge](https://img.shields.io/badge/macos-gray?style=flat-square)
![Static Badge](https://img.shields.io/badge/window11-gray?style=flat-square)


## Contributors
[![Static Badge](https://img.shields.io/badge/%40jbw9964-gray?style=flat-square)
](https://github.com/jbw9964)
[![Static Badge](https://img.shields.io/badge/%402jae1-blue?style=flat-square)
](https://github.com/2jae1)

## Reference

###### [`[1]`]() : [Jansson, Andreas, et al. "Singing voice separation with deep u-net convolutional networks." (2017).](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf)

###### [`[2]`]() : [Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.](https://arxiv.org/pdf/1505.04597.pdf)

## Dataset reference
###### [`[1]`]() : [Non-copyright background musics from `AShamaluevMusic`](https://www.ashamaluevmusic.com/no-copyright-music)

###### [`[2]`]() : [Kaggle - Common Voice dataset](https://www.kaggle.com/datasets/mozillaorg/common-voice)
