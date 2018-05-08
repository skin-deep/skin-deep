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
        
    #def custom_loss(cls, *args, **kwargs):
    #    """Special loss function required for the particular model."""
    #    return None
        
    @staticmethod
    def batchgen(source, catlabels): 
        """Defines data to retrieve from inputs on a per-model basis.
        :param source: an iterable of data
        :param catlabels: category labels
        """
        
        batch = (({
                 'expression_in': np.array(x.T.values),
                 #'expression_in_2': np.array(x.T.values),
                 'diagnosis_in': np.array(catlabels.get(str(x.index.name).upper())),
                },
                {
                 'expression_out': np.array(x.T.values),
                 'diagnosis': np.array(catlabels.get(str(x.index.name).upper())),
                 #'DiagBoost': np.array(catlabels.get(x.index.name))
                })
                for x in source)
        #validgen = itt.cycle([np.asarray([next(sampled[1]).values], dtype='float32') for x in range(self.config.options.get('train_steps', 75))])        
        return batch
    

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
        
    @staticmethod
    def batchgen(source, catlabels): 
        """Defines data to retrieve from inputs on a per-model basis.
        :param source: an iterable of data
        :param catlabels: category labels
        """
        import numpy as np
        from keras.utils import normalize as K_norm
        batch = (
                ({
                 'expression_in': K_norm(np.array(x.T.values), order=1),
                 'diagnosis_in': np.array(catlabels.get(str(x.index.name).upper())),
                },
                {
                 'expression_out': K_norm(np.array(x.T.values), order=1),
                 'diagnosis': np.array(catlabels.get(str(x.index.name).upper())),
                })
                for x in source)
        #print ("BATCH GEN B: ", next(batch))
        return batch

        
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
        transformed = cls.DLbackend.layers.AlphaDropout(0.5)(transformed)
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
            print(str(i)+':', siz or "NONE!")
            encoder_node = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i),
                                                        kernel_regularizer=cls.DLbackend.regularizers.l1_l2(l1=0.01, l2=0.01)
                                                     )
            last_lay = encoder_node(last_lay)
        else: enc_out_layer = last_lay
        
        Encoder = cls.DLmodel(inbound, enc_out_layer, name='Encoder')
        
        # Diagnostician: compressed vals -> class
        # 'interface' layer for inspecting latent spaces as a predictive ensemble
        ensemble_inputs = {enc_out_layer}
        interface_node = cls.DLbackend.layers.Dense(kwargs.get('ens_interface_size', lay_sizes[-1]), activation='linear', activity_regularizer=cls.DLbackend.regularizers.l2(0.01), kernel_initializer='lecun_normal', name='diagger_{}'.format(0))
        for assay_latent_space in ensemble_inputs:
            # corrupt each input separately
            diagger_inp = cls.DLbackend.layers.AlphaDropout(0.1)(assay_latent_space)
            # connect the interface layer to each input
            diagger = interface_node(diagger_inp)
        #diagger = cls.DLbackend.layers.Dense(100, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='diagger_{}'.format(1))(diagger)
        diagnosis = cls.DLbackend.layers.Dense(kwargs.get('num_classes', 3), activation=activators.get('classification', 'softmax'), 
                                                kernel_initializer='lecun_normal', kernel_regularizer=cls.DLbackend.regularizers.l1(0.1),
                                                name='diagnosis')(diagger)
        
        # Decoder: compressed vals -> regression vals
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            #print(i, siz)
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
        
        
class variational_deep_AE(labeled_AE):

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
        # Encoder: vals -> compressed vals
        inbound = cls.DLbackend.layers.Input(shape=([datashape[-1]] or [datashape[0]]), name='expression_in')
        last_lay = inbound
        
        last_lay = cls.input_preprocessing(last_lay, **kwargs)
        
        
        
        for (i, siz) in enumerate(lay_sizes[:-1]):
            print(str(i)+':', siz or "NONE!")
            encoder_node = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i))
            last_lay = encoder_node(last_lay)
        else:
            # VAE stuff:
            latent_dims = 2
            enc_mean = cls.DLbackend.layers.Dense(latent_dims, name='encoded_mean'.format())(last_lay)
            enc_stdev = cls.DLbackend.layers.Dense(latent_dims, name='encoded_log-stdev'.format())(last_lay)
            
            def sampler(mean, log_stddev):
                import keras.backend as K
                std_norm = K.random_normal(shape=(K.shape(mean)[0], latent_dims), mean=0, stddev=1)
                return mean + K.exp(log_stddev) * std_norm
            
            latent_vector = Lambda(sampler)([enc_mean, enc_logstdev])
            
            # encoder output layer:
            enc_out_layer = latent_vector
            
        Encoder = cls.DLmodel(inbound, enc_out_layer, name='Encoder')
        
        # Decoder: compressed vals -> regression vals
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            #print(i, siz)
            decoder_node = cls.DLbackend.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decoder_node(last_lay)
            last_lay = decoded
            dec_start = dec_start or decoder_node
        # let's make sure the last layer is 1:1 to input no matter what
        decoder_node = cls.DLbackend.layers.Dense(Encoder.layers[0].input_shape[-1], activation=activators.get('regression', 'linear'), kernel_initializer='lecun_normal', name='expression_out')
        decoded = decoder_node(last_lay)
        dec_start = dec_start or decoder_node
        
        # Diagnostician: vals -> class
        # 'interface' layer for inspecting latent spaces as a predictive ensemble
        #ensemble_inputs = {enc_out_layer}
        #interface_node = cls.DLbackend.layers.Dense(kwargs.get('ens_interface_size', 650), activation='linear', activity_regularizer=cls.DLbackend.regularizers.l1(0.01), kernel_initializer='lecun_normal', name='diagger_{}'.format(0))
        #for assay_latent_space in ensemble_inputs:
        #    # corrupt each input separately
        #    diagger_inp = cls.DLbackend.layers.AlphaDropout(0.05)(assay_latent_space)
        #    # connect the interface layer to each input
        #    diagger = interface_node(diagger_inp)
            
        diagnosis = cls.DLbackend.layers.Dense(kwargs.get('num_classes', 3), activation=activators.get('classification', 'softmax'), 
                                                kernel_initializer='lecun_normal', activity_regularizer=cls.DLbackend.regularizers.l1(0.01),
                                                name='diagnosis')(enc_out_layer)
            
        Autoencoder = cls.DLmodel(inputs=[inbound], outputs=[decoded, diagnosis], name='Autoencoder')
        Diagnostician=cls.DLmodel(inputs=[inbound], outputs=[diagnosis], name='Predictor')
        
        def VAE_loss(inp, outp):
            """Axis-wise KL-Div + MSE"""
            import keras.backend as K
            reconstruction_loss = K.sum(K.square(outp-inp))
            kl_loss = - 0.5 * K.sum(1 + enc_logstdev - K.square(enc_mean) - K.square(K.exp(enc_logstdev)), axis=-1)
            total_loss = K.mean(reconstruction_loss + kl_loss)    
            return total_loss
        Autoencoder.custom_loss = VAE_loss

        return (Autoencoder, Encoder, Diagnostician)
        
    @staticmethod
    def batchgen(source, catlabels): 
        """Defines data to retrieve from inputs on a per-model basis.
        :param source: an iterable of data
        :param catlabels: category labels
        """
        import numpy as np
        from keras.utils import normalize as K_norm
        batch = (
                ({
                 'expression_in': K_norm(np.array(x.T.values), order=1),
                 'diagnosis_in': np.array(catlabels.get(str(x.index.name).upper())),
                },
                {
                 'expression_out': K_norm(np.array(x.T.values), order=1),
                 'diagnosis': np.array(catlabels.get(str(x.index.name).upper())),
                })
                for x in source)
        return batch
    
def build_models(datashape, which='labeledAE', activators=None, **kwargs):
    if not activators: activators = {'deep': 'selu', 'regression': 'linear', 'classification': 'softmax'}
    model_dict = {'deepAE': deep_AE, 'deepAE-dropout': deepAE_dropout, 'labeledAE': labeled_AE}
    print(kwargs.get('labels'))
    model_to_build = which if isinstance(which, ExpressionModel) else model_dict.get(which, NotImplemented)
    if model_to_build is NotImplemented: raise NotImplementedError
    built = model_to_build(datashape, activators, **kwargs)
    print("Built model: {}".format(built))
    return built
    
def main(datashape=None, which_model=None):
    datashape = datashape or eval(input('Enter datashape: '))
    #print("Known models: \n{}".format("\n".join(model_dict.keys())))
    which_model = which_model or input('Enter model type: ')
    build_models(datashape=datashape, which=which_model)
    
if __name__ == '__main__':
    main()
