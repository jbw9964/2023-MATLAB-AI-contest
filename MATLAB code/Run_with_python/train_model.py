
import sys

from U_Net import gen_unet as gen_unet_py
from U_Net import TrainGenerator, ValidGenerator

from keras.callbacks import EarlyStopping

if __name__ == "__main__" : 
    
    model_path = sys.argv[1]
    src_path = sys.argv[2]
    val_src_path = sys.argv[3]
    input_pattern = sys.argv[4]
    output_pattern = sys.argv[5]
    sample_rate = float(sys.argv[6])
    epoch = int(sys.argv[7])

    model_path = model_path + "/" if model_path[-1] != "/" else model_path

    train_gen = TrainGenerator(
        src_path, input_pattern, output_pattern, 
        sample_rate=sample_rate
    )
    val_gen = ValidGenerator(
        val_src_path, input_pattern, output_pattern, 
        sample_rate=sample_rate
    )
    print("Input shape : ", train_gen.input_shape)

    model = gen_unet_py(input_shape=train_gen.input_shape)
    model.compile(optimizer="adam", loss="mae")

    print("========================= Model summary =========================\n")
    print(model.summary())
    print("\n=================================================================\n")
    
    print("Training model may take times...")

    patience = int(epoch / 10)
    if patience == 0 : patience = 1
    callback = EarlyStopping(verbose=False, restore_best_weights=True, patience=patience)

    print("Start training...", end="")
    model.fit(
        x=train_gen, validation_data=val_gen, verbose=False, 
        epochs=epoch, callbacks=callback
    )
    print("\t--> Done")
    model.save(model_path + "model.h5")
    print("Model was saved in [{}]".format(model_path + "model.h5"))
