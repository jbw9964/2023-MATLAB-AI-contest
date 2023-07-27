
__all__ = [
    "convert_pred"
]

import time, sys, librosa
import numpy as np

from threading import Thread
from keras import Model
from numpy import ndarray
from IPython.display import Audio

from . import polar_to_complex
from .generators import PredGenerator

def convert_pred(
            model : Model, src_path : str, pred_dir : str, pattern : str=".mp3",
            n_fft : int=1918, win_length : int=1024, sample_rate=None
    ) : 
        def check_status() :
            nonlocal total_num, count, stdout_list
            while count <= total_num : 
                string = "\rProcessing... [{}] : ".format(stdout_list[0]).ljust(15)
                string += "[{}/{}]".format(str(count).zfill(3), str(total_num).zfill(3))
                sys.stdout.write(string)
                sys.stdout.flush()
                stdout_list.append(stdout_list.pop(0))
                time.sleep(0.15)

        input_shape = model.input_shape
        pred_generator = PredGenerator(input_shape, src_path, pred_dir, pattern, n_fft, win_length, sample_rate)

        total_num = len(pred_generator._input_path_list)
        count = 1

        stdout_list = "/-\|"
        stdout_list = list(stdout_list)
        t1 = Thread(target=check_status)
        t1.start()
        while not pred_generator._is_all_done : 
            # print(count)
            sample_rate, pred_path, n_fft, win_len, sample_shape, total_length = pred_generator.before_pred()
            pred_data = model.predict(pred_generator, verbose=False)

            assert type(pred_data) == ndarray
            
            total_arr = np.zeros(
                (pred_generator.input_shape[0], pred_generator.input_shape[1] * total_length), 
                dtype=np.complex64
            )
            
            pivot = 0
            for bulk_arr in pred_data : 
                bulk_pivot = 0
                bulk_max_pivot = len(bulk_arr[0])
                
                while bulk_pivot < bulk_max_pivot : 
                    alter_pivot = 64
                    total_arr[:,pivot:pivot + alter_pivot] = polar_to_complex(bulk_arr[:,bulk_pivot:bulk_pivot + alter_pivot])

                    bulk_pivot += alter_pivot
                    pivot += alter_pivot

            output = librosa.istft(total_arr, n_fft=n_fft, win_length=win_len)
            output_audio = Audio(output, rate=sample_rate)
            with open(pred_path, mode="wb") as f : 
                f.write(output_audio.data)

            del pred_data, total_arr, output, output_audio
            # gc.collect()
            
            count += 1
        
        t1.join()

        print("\t Done")

