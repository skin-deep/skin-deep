import sys, os, shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class ExpressionModel(object):
    """Container for ML models."""
    import keras as DLbackend
    DLmodel = DLbackend.models.Model
    
    @classmethod
    def build(*args, **kwargs):
        """(Re-)builds the model.
        
            Returns one or more models; when not invoked explicitly, the first item is assumed to be 
        the primary, complete model, and the following items, if present, are (functional) subcomponents,
        e.g. encoder/decoder layers in autoencoders, a model applied to a single input in a model with many 
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
        main_mdl.aux_components = {getattr(part, 'name', str(part)) : part for part in aux_parts}
        print("Yup, done that!")#TRAIN ERRYTHING PER-CLASS
        
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
        """Subclass hook for adding extra preprocessing before the autoencoder itself."""
        return input_lay
        
    @classmethod
    def build(cls, datashape, activators=('selu', 'sigmoid'), compression_fac=None, **kwargs):
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
            encoded = cls.DLbackend.layers.Dense(siz, activation=activators[0], name='encoder_{}'.format(i))(last_lay)
            last_lay = encoded
        encoder = cls.DLmodel(inbound, encoded, name='Encoder')
        
        representation = cls.DLbackend.layers.Input([encoder.layers[-1].output_shape[1:]])
        
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            print(i, siz)
            decfunc = cls.DLbackend.layers.Dense(siz, activation=activators[1], name='decoder_{}'.format(i))
            decoded = decfunc(last_lay)
            last_lay = decoded
            if i==0: dec_start = decfunc
        # let's make sure the last layer is 1:1 to input no matter what
        decfunc = cls.DLbackend.layers.Dense(encoder.layers[0].input_shape[-1], activation=activators[1], name='decoder_{}'.format(len(lay_sizes)-1))
        decoded = decfunc(last_lay)
            
        autoencoder = cls.DLmodel(inbound, decoded, name='Autoencoder')
        
        # dummy input for feeding into decoder separately
        dummy_in = cls.DLbackend.layers.Input([encoder.layers[-1].output_shape[-1]])
        decoder = cls.DLmodel(dummy_in, dec_start(dummy_in), name='Decoder') #this is bullshit rn

        return (autoencoder, encoder, decoder)
    
class deepAE_dropout(deep_AE):
    @classmethod
    def input_preprocessing(cls, input_lay, *args, **kwargs):
        transformed = input_lay
        transformed = cls.DLbackend.layers.Dropout(0.5)(transformed)
        return transformed
    
def build_models(datashape, which='deepAE', activators=('selu', 'sigmoid'), **kwargs):
    model_dict = {'deepAE': deep_AE, 'deepAE-dropout': deepAE_dropout}
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
