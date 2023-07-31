
import sys

from U_Net import convert_pred as convert_pred_py

from keras.models import load_model

if __name__ == "__main__" : 
    model_path = sys.argv[1]
    src_path = sys.argv[2]
    pred_dir = sys.argv[3]
    pattern = sys.argv[4]
    sample_rate = float(sys.argv[5])

    model = load_model(model_path)
    print("\n\nLoaded model from [{}]".format(model_path))

    print("Converting data...", end="")
    convert_pred_py(model, src_path, pred_dir, pattern, sample_rate=sample_rate)
    print("\t--> Done")