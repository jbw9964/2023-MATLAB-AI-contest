
__all__ = [
    "gen_separate_unet", 
    "gen_unet"
]

from keras import Model
from keras.layers import InputLayer, Conv2D, Dropout, BatchNormalization
from keras.layers import Concatenate, Conv2DTranspose, LeakyReLU, ReLU
from keras.layers import Multiply, Subtract

def gen_separate_unet(input_shape=(960, 832, 2), encode=5, num_separ=2) -> Model : 
    input_layer = InputLayer(input_shape=input_shape).input

    separation_list = []
    for _ in range(num_separ) : 
        last_layer = input_layer
        concate_list = []
        filter_num = 16 * input_shape[2]                # set initial number of filters as 16
        
        for _ in range(encode) : 
            conv_encoder = Conv2D(filters=filter_num, kernel_size=5, strides=2, padding="same")(last_layer)
            batch_norm = BatchNormalization()(conv_encoder)
            activ_layer = LeakyReLU(alpha=0.2)(batch_norm)
            
            concate_list.insert(0, conv_encoder)
            last_layer = activ_layer
            filter_num *= 2
        
        conv_layer = Conv2D(filters=filter_num, kernel_size=5, strides=2, padding="same")(last_layer)
        activ_layer = LeakyReLU(alpha=0.2)(conv_layer)
        last_layer = activ_layer

        count = 1
        for concate_layer in concate_list : 
            filter_num /= 2

            conv_trans = Conv2DTranspose(filters=filter_num, kernel_size=5, strides=2, padding="same")(last_layer)
            merge_layer = Concatenate(axis=3)([conv_trans, concate_layer])
            conv_decoder = Conv2D(filters=filter_num, kernel_size=5, strides=1, padding="same")(merge_layer)
            batch_norm = BatchNormalization()(conv_decoder)
            activ_layer = ReLU()(batch_norm)

            if count <= 3 : 
                last_layer = Dropout(rate=0.5)(activ_layer)
            
            else : 
                last_layer = activ_layer
            
            count += 1

        conv_layer = Conv2DTranspose(filters=8 * input_shape[2], kernel_size=5, strides=2, padding="same")(last_layer)
        conv_layer = Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=1, padding="same", activation="sigmoid")(conv_layer)
        
        separation_list.append(conv_layer)

    output_layer_list = []
    subtract_list = []
    count = 0
    for layer in separation_list : 
        if not count : 
            vocal_separation_layer = layer
        else : 
            output_layer = Multiply(name="separation_{}".format(count))([input_layer, layer])
            output_layer_list.append(output_layer)
            subtract_list.append(layer)
        count += 1
    
    for iteration, inst_layer in enumerate(subtract_list) : 
        vocal_separation_layer = Subtract(name="sub_{}".format(iteration + 1))([vocal_separation_layer, inst_layer])
    
    vocal_separation_layer = Multiply(name="vocal_separation")([input_layer, vocal_separation_layer])
    output_layer_list.insert(0, vocal_separation_layer)

    return Model(inputs=[input_layer], outputs=output_layer_list)

def gen_unet(input_shape=(960, 832, 2), encode=5) : 
    input_layer = InputLayer(input_shape=input_shape).input
    
    last_layer = input_layer
    concate_list = []
    filter_num = 16 * input_shape[2]                # set initial number of filters as 16
    
    for _ in range(encode) : 
        conv_encoder = Conv2D(filters=filter_num, kernel_size=5, strides=2, padding="same")(last_layer)
        batch_norm = BatchNormalization()(conv_encoder)
        activ_layer = LeakyReLU(alpha=0.2)(batch_norm)
        
        concate_list.insert(0, conv_encoder)
        last_layer = activ_layer
        filter_num *= 2
    
    conv_layer = Conv2D(filters=filter_num, kernel_size=5, strides=2, padding="same")(last_layer)
    activ_layer = LeakyReLU(alpha=0.2)(conv_layer)
    last_layer = activ_layer

    count = 1
    for concate_layer in concate_list : 
        filter_num /= 2

        conv_trans = Conv2DTranspose(filters=filter_num, kernel_size=5, strides=2, padding="same")(last_layer)
        merge_layer = Concatenate(axis=3)([conv_trans, concate_layer])
        conv_decoder = Conv2D(filters=filter_num, kernel_size=5, strides=1, padding="same")(merge_layer)
        batch_norm = BatchNormalization()(conv_decoder)
        activ_layer = ReLU()(batch_norm)

        if count <= 3 : 
            last_layer = Dropout(rate=0.5)(activ_layer)
        
        else : 
            last_layer = activ_layer
        
        count += 1

    conv_layer = Conv2DTranspose(filters=8 * input_shape[2], kernel_size=5, strides=2, padding="same")(last_layer)
    conv_layer = Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=1, padding="same", activation="sigmoid")(conv_layer)
    output_layer = Multiply()([input_layer, conv_layer])
    
    return Model(inputs=[input_layer], outputs=[output_layer])