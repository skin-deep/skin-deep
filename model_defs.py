import sys, os, shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class ExpressionModel(object):
    """Container for ML models."""
    import keras as DLbackend
    DLmodel = DLbackend.models.Model
    
    #import keras.backend.tensorflow_backend as ktf
    #def get_session(gpu_fraction=0.333):
    #    import tensorflow as tf
    #    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    #    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #ktf.set_session(get_session())
    
    @classmethod
    def build(*args, **kwargs):
        """(Re-)builds the model.
        
            Returns one or more models; when not invoked explicitly, the first item is assumed to be 
        the primary, complete model, and the following items, if present, are (functional) subcomponents,
        e.g. Encoder/Decoder layers in autoencoders, a model applied to a single input in a model with many 
        independent inputs etc.
        """
        # separate from new/init to facilitate iteration
        return NotImplemented
    
    def __new__(cls, *args, **kwargs):
        # making the actual model an attribute of an instance would be rather hacky
        main_mdl = cls.build(*args, **kwargs)
        
        # the builder may return multiple items, presumably all models
        whole, aux_parts = main_mdl, None
        try: main_mdl, aux_parts = main_mdl[0], main_mdl[1:]
        except (AttributeError, IndexError): pass
        if main_mdl is NotImplemented: raise NotImplementedError
        
        return whole
        
    

class deep_AE(ExpressionModel):
    @classmethod
    def calculate_sizes(cls, datashape, compression_fac):
        try: compression_fac = int(compression_fac)
        except Exception as E:
            print("Using default compression factor!")
            compression_fac = 128
        try:
            print("DATASHAPE: "+str(datashape))
            uncompr_size = datashape[-1]
            compr_size = max(1, (min(uncompr_size, uncompr_size // compression_fac)))
        except (IndexError, TypeError) as E:
            print(E)
            uncompr_size = None
            compr_size = compression_fac
        return uncompr_size, compr_size

    @classmethod
    def input_preprocessing(input_lay, *args, **kwargs):
        """Subclass hook for adding extra preprocessing before the Autoencoder itself."""
        return input_lay
        
    @classmethod
    def build(cls, datashape, activators=None, compression_fac=None, **kwargs):
        activators = activators or {'deep': 'selu', 'regression': 'linear'}
        uncompr_size, compr_size = cls.calculate_sizes(datashape, compression_fac)
        # deep levels handling:
        deep_lvls = kwargs.get('depth', 1)
        try: deep_lvls = max(1, abs(int(deep_lvls)))
        except Exception: deep_lvls = 1
        
        clamp_size = lambda S: max(1, min(S, uncompr_size-1))
        lay_sizes = [clamp_size(compr_size * (2**lvl)) for lvl in reversed(range(deep_lvls))] # FIFO sizes for encoders!
        print(lay_sizes)

        # layers
        inbound = cls.DLbackend.layers.Input(shape=(datashape or [datashape[0]]))
        last_lay = inbound
        
        last_lay = cls.input_preprocessing(last_lay, **kwargs)
        
        for (i, siz) in enumerate(lay_sizes): #[0]th is already built
            print(siz or "NONE!")
            encoded = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i))(last_lay)
            last_lay = encoded
        Encoder = cls.DLmodel(inbound, encoded, name='Encoder')
        
        representation = cls.DLbackend.layers.Input(Encoder.layers[-1].output_shape[1:])
        
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            print(i, siz)
            decfunc = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decfunc(last_lay)
            last_lay = decoded
            dec_start = dec_start or decfunc
        # let's make sure the last layer is 1:1 to input no matter what
        decfunc = cls.DLbackend.layers.Dense(Encoder.layers[0].input_shape[-1], activation=activators.get('regression', 'linear'), kernel_initializer='lecun_normal', name='decoder_{}'.format(len(lay_sizes)-1))
        decoded = decfunc(last_lay)
        dec_start = dec_start or decfunc
            
        Autoencoder = cls.DLmodel(inbound, decoded, name='Autoencoder')
        
        # dummy input for feeding into Decoder separately
        dummy_in = cls.DLbackend.layers.Input([Encoder.layers[-1].output_shape[-1]])
        Decoder = cls.DLmodel(dummy_in, dec_start(dummy_in), name='Decoder') #this is bullshit rn

        return (Autoencoder, Encoder, Decoder)
        

class deepAE_dropout(deep_AE):
    @classmethod
    def input_preprocessing(cls, input_lay, *args, **kwargs):
        transformed = input_lay
        transformed = cls.DLbackend.layers.AlphaDropout(0.5)(transformed)
        return transformed
    
class labeled_AE(deep_AE):
    @classmethod
    def input_preprocessing(cls, input_lay, *args, **kwargs):
        transformed = input_lay
        transformed = cls.DLbackend.layers.AlphaDropout(0.25)(transformed)
        return transformed
        
    @classmethod
    def build(cls, datashape, activators=None, compression_fac=None, **kwargs):
        labels = kwargs.get('labels')
        activators = activators or {'deep': 'selu', 'regression': 'linear'}
        uncompr_size, compr_size = cls.calculate_sizes(datashape, compression_fac)
        # deep levels handling:
        deep_lvls = kwargs.get('depth', 1)
        try: deep_lvls = max(1, abs(int(deep_lvls)))
        except Exception: deep_lvls = 1
        
        clamp_size = lambda S: max(1, min(S, uncompr_size-1))
        lay_sizes = [clamp_size(compr_size * (2**lvl)) for lvl in reversed(range(deep_lvls))] # FIFO sizes for encoders!
        print(lay_sizes)

        # layers
        # Encoder: vals -> compressed vals
        inbound = cls.DLbackend.layers.Input(shape=([datashape[-1]] or [datashape[0]]), name='expression_in')
        last_lay = inbound
        
        last_lay = cls.input_preprocessing(last_lay, **kwargs)
        
        for (i, siz) in enumerate(lay_sizes):
            print(siz or "NONE!")
            encoder_node = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i))
            last_lay = encoder_node(last_lay)
        else: enc_out_layer = last_lay
        
        Encoder = cls.DLmodel(inbound, enc_out_layer, name='Encoder')
        
        # Diagnostician: compressed vals -> class
        #import numpy as np
        print("MOD LABEL:\n ", labels)
        #labels = np.array(labels)
        # 'interface' layer for inspecting latent spaces as a predictive ensemble
        ensemble_inputs = {enc_out_layer}
        interface_node = cls.DLbackend.layers.Dense(kwargs.get('ens_interface_size', 350), activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='diagger_{}'.format(0))
        for assay_latent_space in ensemble_inputs:
            # corrupt each input separately
            diagger_inp = cls.DLbackend.layers.AlphaDropout(0.15)(assay_latent_space)
            # connect the interface layer to each input
            diagger = interface_node(diagger_inp)
        #diagger = cls.DLbackend.layers.Dense(100, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='diagger_{}'.format(1))(diagger)
        diagnosis = cls.DLbackend.layers.Dense(3, activation=activators.get('classification', 'softmax'), kernel_initializer='lecun_normal', name='diagnosis_out')(diagger)
        
        # Decoder: compressed vals -> regression vals
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            print(i, siz)
            decoder_node = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decoder_node(last_lay)
            last_lay = decoded
            dec_start = dec_start or decoder_node
        # let's make sure the last layer is 1:1 to input no matter what
        decoder_node = cls.DLbackend.layers.Dense(Encoder.layers[0].input_shape[-1], activation=activators.get('regression', 'linear'), kernel_initializer='lecun_normal', name='expression_out')
        decoded = decoder_node(last_lay)
        dec_start = dec_start or decoder_node
            
        Autoencoder = cls.DLmodel(inputs=[inbound], outputs=[decoded, diagnosis], name='Autoencoder')
        
        Diagnostician=cls.DLmodel(inputs=[inbound], outputs=[diagnosis], name='Predictor')
        # dummy input for feeding into Decoder separately
        #dummy_in = cls.DLbackend.layers.Input([Encoder.layers[-1].output_shape[-1]])
        #Decoder = cls.DLmodel(dummy_in, dec_start(dummy_in), name='Decoder') #this is bullshit rn

        return (Autoencoder, Encoder, Diagnostician)
    
def build_models(datashape, which='labeledAE', activators=None, **kwargs):
    if not activators: activators = {'deep': 'selu', 'regression': 'linear', 'classification': 'softmax'}
    model_dict = {'deepAE': deep_AE, 'deepAE-dropout': deepAE_dropout, 'labeledAE': labeled_AE}
    print(kwargs.get('labels'))
    model_to_build = which if isinstance(which, ExpressionModel) else model_dict.get(which, NotImplemented)
    if model_to_build is NotImplemented: raise NotImplementedError
    return model_to_build(datashape, activators, **kwargs)
    
def main(datashape=None, which_model=None):
    datashape = datashape or eval(input('Enter datashape: '))
    #print("Known models: \n{}".format("\n".join(model_dict.keys())))
    which_model = which_model or input('Enter model type: ')
    build_models(datashape=datashape, which=which_model)
    
if __name__ == '__main__':
    main()
