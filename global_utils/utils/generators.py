
__all__ = [
    "TrainGenerator",
    "ValidGenerator",
    "EvalGenerator"
]

import librosa, warnings
import numpy as np

from sys import getsizeof
from glob import glob1

from keras.utils import Sequence
from numpy import ndarray

from .complex_to_polar import complex_to_polar

class TrainGenerator(Sequence) : 
    def __init__(
            self, src_path : str, bulk_num : int=5, 
            sample_dur : float=5, max_cache_size : float=2, restrict_cache=False,
            n_fft : int=1918, win_length : int=1024, 
            sample_rate=None, shuffle=True
    ) :
        if src_path[-1] != "/" : src_path += "/"

        input_list = [src_path + name for name in glob1(dirname=src_path, pattern="merge*")]
        output_list = [src_path + name for name in glob1(dirname=src_path, pattern="voice*")]
        
        assert len(input_list) == len(output_list)\
            , AssertionError("The number of source sample must be same. : [{}, | {}]".format(
            len(input_list), len(output_list)
            ))
        assert len(input_list), AssertionError("In src_path, no match with pattern [merge*]")
        assert len(output_list), AssertionError("In src_path, no match with pattern [voice*]")
        
        def sort_via_dur(input_list) : 
            var_list = sorted(input_list)
            try : 
                var_list = [[path, librosa.get_duration(path=path)] for path in var_list]
            except : 
                var_list = [[path, librosa.get_duration(filename=path)] for path in var_list]
            var_list = sorted(var_list, key=lambda x : x[1], reverse=True)
            return [var[0] for var in var_list]
        
        self._input_path = sort_via_dur(input_list)
        self._output_path = sort_via_dur(output_list)

        self._bulk_num = bulk_num
        self._sample_rate = sample_rate
        self._sample_dur = sample_dur
        
        self._offset_list = np.zeros_like(self._input_path, dtype=np.float32)
        try : 
            self._max_dur_list = [librosa.get_duration(path=path) for path in self._input_path]
        except : 
            self._max_dur_list = [librosa.get_duration(filename=path) for path in self._input_path]

        self._index_list = [i for i in range(len(self._input_path))]
        if shuffle : np.random.shuffle(self._index_list)
        
        self._n_fft = n_fft
        self._win_len = win_length

        rate = librosa.get_samplerate(path=self._input_path[0]) if not sample_rate else sample_rate
        sample_source = librosa.load(path=self._input_path[0], sr=rate, duration=sample_dur)[0]
        self._sample_arr = librosa.stft(sample_source, n_fft=n_fft, win_length=win_length)
        quotient = len(self._sample_arr[0]) // 64
        self._sample_arr = complex_to_polar(self._sample_arr[:,:quotient * 64])

        self._src_index = 0

        self._max_cache_size = max_cache_size * (1024**3)
        self._restrict_cache = restrict_cache

        self.__cache_warning()
    
    def __cache_warning(self) : 
        if self._max_cache_size <= self.__estimate_cache() : 
            if self._restrict_cache : 
                raise MemoryError("Loaded data exceeded max_cache_size.")
            else : 
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn("Loaded data exceeded max_cache_size.", ResourceWarning)
    
    def __estimate_cache(self) : 
        return getsizeof(self._sample_arr) * self._bulk_num * 2

    def __resource_validation(self, arr : ndarray) : 
        shape1, shape2 = arr.shape, self._sample_arr.shape
        assert shape1 == shape2, AssertionError("The shape of IO is different : [{} | {}]".format(shape1, shape2))
    
    def __get_src_list(self, src_index) : 
        return [i % len(self._input_path) for i in range(src_index, src_index + self._bulk_num)]

    def __load_data(self, src_index, update=True) : 
        x_list = []
        y_list = []
        for index in self.__get_src_list(src_index) : 
            x_list.append(self.__load_single_data(index, self._input_path))
            y_list.append(self.__load_single_data(index, self._output_path))
            if update : self._offset_list[index] += self._sample_dur
        
        return np.array(x_list), np.array(y_list)

    def __load_single_data(self, src_index, path_list) : 
        path = path_list[src_index]
        offset = self._offset_list[src_index]
        sample_dur = self._sample_dur
        max_dur = self._max_dur_list[src_index] - 0.1
        
        if offset + sample_dur >= max_dur : 
            self._offset_list[src_index] = 0
            return self.__load_single_data(src_index, path_list)
        
        else : 
            sample_rate = self._sample_rate if self._sample_rate else librosa.get_samplerate(path) 
            source = librosa.load(path, sr=sample_rate, offset=offset, duration=sample_dur)[0]
            D = librosa.stft(source, n_fft=self._n_fft, win_length=self._win_len)
            quotient = len(D[0]) // 64
            del source
            D = complex_to_polar(D[:,:quotient * 64])
            self.__resource_validation(D)

            return D

    def __len__(self) :
        current_index = self.__get_src_list(self._src_index)
        max = 0
        for index in current_index : 
            try : 
                duration = librosa.get_duration(path=self._input_path[index])
            except : 
                duration = librosa.get_duration(filename=self._input_path[index])
            if duration > max : max = duration
        
        return int(max // self._sample_dur + 1)
    
    def __getitem__(self, index) :
        return self.__load_data(self._src_index, update=True)
    
    def on_epoch_end(self) :
        self._src_index += self._bulk_num
        self._src_index %= len(self._input_path)

        return super().on_epoch_end()

    @property
    def input_shape(self) : 
        return self._sample_arr.shape

class ValidGenerator(TrainGenerator) : 
    pass

class EvalGenerator(TrainGenerator) : 
    def __init__(
            self, src_path: str, eval_num: int = 5, 
            sample_dur: float = 5, max_cache_size: float = 2, restrict_cache=False, 
            n_fft: int = 1918, win_length: int = 1024, 
            sample_rate=None, shuffle=False
    ):
        super().__init__(src_path, eval_num, sample_dur, max_cache_size, restrict_cache, n_fft, win_length, sample_rate, shuffle)


