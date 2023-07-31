
import sys

from U_Net.utils import gen_dataset as gen_dataset_py

if __name__ == "__main__" : 
    
    target_dir = sys.argv[1]
    music_dir = sys.argv[2]
    voice_dir = sys.argv[3]
    voice_amp_ratio = float(sys.argv[4])
    train_test_split = bool(sys.argv[5])

    print("Generating data...", end="")
    gen_dataset_py(
        target_dir, music_dir, voice_dir, 
        voice_amp_ratio, train_test_split
    )
    print("\t--> Done")