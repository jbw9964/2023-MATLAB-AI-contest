
## How to use utils in U_Net module

##### 1. [`complex_to_polar`](../U_Net/utils/complex_to_polar.py) and [`polar_to_complex`](../U_Net/utils/polar_to_complex.py)
- When we operate [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) (Short Time Fourier Formation) to our music source, the returned value will be array with complex numbers.

- Since the complex numbers are inappropriate to train model, we can re-value our complex numbers to [polar-coordinated value](https://en.wikipedia.org/wiki/Polar_coordinate_system), $r$ and $\theta$, that represent the data in complex plane.

- Inorder to check model's prediction as music source, we should inverse the polar-coordinated value to complex number. At this point, you can use `polar_to_complex` function.


```python
# import numpy to generate complex-valued array

import numpy as np

real_part = np.round(np.random.random((2, 2)), decimals=3)
imag_part = np.round(np.random.random((2, 2)), decimals=3)
complex_arr = real_part + imag_part * 1j

print(complex_arr.shape)
display(complex_arr)
```

```
(2, 2)
array([[0.416+0.164j, 0.731+0.875j],
       [0.623+0.902j, 0.905+0.483j]])
```

```python
from U_Net.utils import complex_to_polar

polar_arr = complex_to_polar((complex_arr))

print(polar_arr.shape)
display(polar_arr)
```
```
(2, 2, 2)
array([[[0.44715993, 0.37552303],
        [1.14016929, 0.87482279]],

       [[1.09623583, 0.96634784],
        [1.02582357, 0.49024404]]])
```

---
The `complex_arr` with shape (2, 2) has been re-valued to `polar_arr`. The (..., 0) element represent the radius in complex plane, and (..., 1) element represent the radian angle.

```python
r = polar_arr[..., 0]       # radius in complex plane
theta = polar_arr[..., 1]   # radian angle in complex plane
```

You can see `complex_to_polar` function works well.

```python
print("Same radius")
display(np.abs(complex_arr), r, np.isclose(np.abs(complex_arr), r))

print("Same angle")
display(np.arctan(imag_part / real_part), theta, np.isclose(np.arctan(imag_part / real_part), theta))
```
```
Same radius
array([[0.44715993, 1.14016929],
       [1.09623583, 1.02582357]])

array([[0.44715993, 1.14016929],
       [1.09623583, 1.02582357]])

array([[ True,  True],
       [ True,  True]])
       
Same angle
array([[0.37552303, 0.87482279],
       [0.96634784, 0.49024404]])

array([[0.37552303, 0.87482279],
       [0.96634784, 0.49024404]])

array([[ True,  True],
       [ True,  True]])
```

And you can see the output of `polar_to_complex` are same with `complex_arr`.

```python
from U_Net.utils import polar_to_complex

inverse = polar_to_complex(polar_arr)

display(complex_arr, inverse, np.isclose(complex_arr, inverse))
```
```
array([[0.416+0.164j, 0.731+0.875j],
       [0.623+0.902j, 0.905+0.483j]])

array([[0.416+0.164j, 0.731+0.875j],
       [0.623+0.902j, 0.905+0.483j]])

array([[ True,  True],
       [ True,  True]])
```

---

##### 2. [`gen_dataset`](../U_Net/utils/gen_dataset.py)
- In order to train model, we need large dataset. For our model, the training data should be 2 music source.

- One contains musics and vocals, One only contains vocals. However, it is difficult to find proper dataset, specifically, the sync between vocal and muscis should be exact. Further more, the copyrights of muscis can be obstacle.

- So we choosed to make our own datasets, using [`gen_dataset`](../U_Net/utils/gen_dataset.py).

- [`gen_dataset`](../U_Net/utils/gen_dataset.py) can merges the muscis and speech dataset. We used [non-copyright musics](https://www.ashamaluevmusic.com/no-copyright-music) and [Common voice dataset in kaggle](https://www.kaggle.com/datasets/mozillaorg/common-voice).

- So we can make our own dataset, seperated by musics and vocals.

```python
from U_Net.utils import gen_dataset

target_dir = "../Data/target_dir/"      # directory that saves the training data
music_dir = "../Data/music_only/"       # there are 10 muscis in sample
voice_dir = "../Data/voice_only/"       # there are 50 voices in sample

gen_dataset(
    target_dir=target_dir, music_dir=music_dir, voice_dir=voice_dir,
    voice_amp_ratio=0.6, train_test_split=0.4, random_state=None
)
```
```
Processing... [/] : [010/010]	  Done
```

---

##### - [`gen_dataset`](../U_Net/utils/gen_dataset.py) arguments
```python
gen_dataset(
    target_dir=target_dir, music_dir=music_dir, voice_dir=voice_dir,
    voice_amp_ratio=0.6, train_test_split=0.4, random_state=None
)
```
- The `target_dir` is a path that generated datasets will be saved.
- The `music_dir` is the path that every muscis files exists, and `voice_dir` so on.
- `voice_amp_ratio` is a ratio to fit the amplitude of `voice data` with `music data`. If the `voice_amp_ratio=1.0`, the voice will be large as same as the music.
- `train_test_split` is a ratio to split test dataset. If `train_test_split=None`, test dataset won't be generated.
- `random_state` is a random state to shuffles the `voice data` in `voic_dir`. If `random_state=None`, it shuffles data without random state.
- `random_state` uses `np.random.RandomState`.

If you done correctly, `target_dir` will be look like this.


```zsh
$ cd ../Data/target_dir && tree
.
├── test_data
│   ├── merge_007.wav
│   ├── merge_008.wav
│   ├── merge_009.wav
│   ├── merge_010.wav
│   ├── music_007.wav
│   ├── music_008.wav
│   ├── music_009.wav
│   ├── music_010.wav
│   ├── voice_007.wav
│   ├── voice_008.wav
│   ├── voice_009.wav
│   └── voice_010.wav
└── train_data
    ├── merge_001.wav
    ├── merge_002.wav
    ├── merge_003.wav
    ├── merge_004.wav
    ├── merge_005.wav
    ├── merge_006.wav
    ├── music_001.wav
    ├── music_002.wav
    ├── music_003.wav
    ├── music_004.wav
    ├── music_005.wav
    ├── music_006.wav
    ├── voice_001.wav
    ├── voice_002.wav
    ├── voice_003.wav
    ├── voice_004.wav
    ├── voice_005.wav
    └── voice_006.wav

3 directories, 30 files
```

