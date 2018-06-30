import sys, os, shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import SDutils

class ExpressionModel(object):
    """Container for ML models."""
    import keras
    DLmodel = keras.models.Model
    
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
        inbound = keras.layers.Input(shape=(datashape or [datashape[0]]))
        last_lay = inbound
        
        last_lay = cls.input_preprocessing(last_lay, **kwargs)
        
        for (i, siz) in enumerate(lay_sizes): #[0]th is already built
            print(siz or "NONE!")
            encoded = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i))(last_lay)
            last_lay = encoded
        Encoder = cls.DLmodel(inbound, encoded, name='Encoder')
        
        representation = keras.layers.Input(Encoder.layers[-1].output_shape[1:])
        
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            print(i, siz)
            decfunc = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decfunc(last_lay)
            last_lay = decoded
            dec_start = dec_start or decfunc
        # let's make sure the last layer is 1:1 to input no matter what
        decfunc = keras.layers.Dense(Encoder.layers[0].input_shape[-1], activation=activators.get('regression', 'linear'), kernel_initializer='lecun_normal', name='decoder_{}'.format(len(lay_sizes)-1))
        decoded = decfunc(last_lay)
        dec_start = dec_start or decfunc
            
        Autoencoder = cls.DLmodel(inbound, decoded, name='Autoencoder')
        
        # dummy input for feeding into Decoder separately
        dummy_in = keras.layers.Input([Encoder.layers[-1].output_shape[-1]])
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
        transformed = keras.layers.AlphaDropout(0.5)(transformed)
        return transformed
    
class labeled_AE(deep_AE):
    @classmethod
    def input_preprocessing(cls, input_lay, *args, **kwargs):
        transformed = input_lay
        transformed = keras.layers.AlphaDropout(0.35)(transformed)
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
        inbound = keras.layers.Input(shape=([datashape[-1]] or [datashape[0]]), name='expression_in')
        last_lay = inbound
        
        last_lay = cls.input_preprocessing(last_lay, **kwargs)
        
        for (i, siz) in enumerate(lay_sizes):
            print(str(i)+':', siz or "NONE!")
            encoder_node = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i),
                                                        kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                                                     )
            last_lay = encoder_node(last_lay)
        else: enc_out_layer = last_lay
        
        Encoder = cls.DLmodel(inbound, enc_out_layer, name='Encoder')
        
        # Diagnostician: compressed vals -> class
        # 'interface' layer for inspecting latent spaces as a predictive ensemble
        ensemble_inputs = {enc_out_layer}
        interface_node = keras.layers.Dense(kwargs.get('ens_interface_size', lay_sizes[-1]), activation='linear', activity_regularizer=keras.regularizers.l2(0.01), kernel_initializer='lecun_normal', name='diagger_{}'.format(0))
        for assay_latent_space in ensemble_inputs:
            # corrupt each input separately
            diagger_inp = keras.layers.AlphaDropout(0.1)(assay_latent_space)
            # connect the interface layer to each input
            diagger = interface_node(diagger_inp)
        #diagger = keras.layers.Dense(100, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='diagger_{}'.format(1))(diagger)
        diagnosis = keras.layers.Dense(kwargs.get('num_classes', 3), activation=activators.get('classification', 'softmax'), 
                                                kernel_initializer='lecun_normal', kernel_regularizer=keras.regularizers.l1(0.1),
                                                name='diagnosis')(diagger)
        
        # Decoder: compressed vals -> regression vals
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            #print(i, siz)
            decoder_node = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decoder_node(last_lay)
            last_lay = decoded
            dec_start = dec_start or decoder_node
        # let's make sure the last layer is 1:1 to input no matter what
        decoder_node = keras.layers.Dense(Encoder.layers[0].input_shape[-1], activation=activators.get('regression', 'linear'), kernel_initializer='lecun_normal', name='expression_out')
        decoded = decoder_node(last_lay)
        dec_start = dec_start or decoder_node
            
        Autoencoder = cls.DLmodel(inputs=[inbound], outputs=[decoded, diagnosis], name='Autoencoder')
        
        Diagnostician=cls.DLmodel(inputs=[inbound], outputs=[diagnosis], name='Predictor')
        # dummy input for feeding into Decoder separately
        #dummy_in = keras.layers.Input([Encoder.layers[-1].output_shape[-1]])
        #Decoder = cls.DLmodel(dummy_in, dec_start(dummy_in), name='Decoder') #this is bullshit rn

        return (Autoencoder, Encoder, Diagnostician)
        
        
class coherent_AE(labeled_AE):
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
        inbound = keras.layers.Input(shape=([datashape[-1]] or [datashape[0]]), name='expression_in')
        last_lay = inbound
        
        last_lay = cls.input_preprocessing(last_lay, **kwargs)
        
        for (i, siz) in enumerate(lay_sizes):
            print(str(i)+':', siz or "NONE!")
            encoder_node = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='encoder_{}'.format(i),
                                                        kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                                                     )
            last_lay = encoder_node(last_lay)
        else: enc_out_layer = last_lay
        
        Encoder = cls.DLmodel(inbound, enc_out_layer, name='Encoder')
        
        # Decoder: compressed vals -> regression vals
        dec_start = None
        for (i, siz) in enumerate(reversed(lay_sizes[:-1])):
            #print(i, siz)
            decoder_node = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decoder_node(last_lay)
            last_lay = decoded
            dec_start = dec_start or decoder_node
        # let's make sure the last layer is 1:1 to input no matter what
        decoder_node = keras.layers.Dense(Encoder.layers[0].input_shape[-1], activation=activators.get('regression', 'linear'), kernel_initializer='lecun_normal', name='expression_out')
        decoded = decoder_node(last_lay)
        dec_start = dec_start or decoder_node
        
        # Diagnostician: vals -> class
        diagger = keras.layers.Dense(kwargs.get('num_classes', 3), activation=activators.get('classification', 'softmax'), 
                                                kernel_initializer='lecun_normal', kernel_regularizer=keras.regularizers.l1(0.1),
                                                name='diagnosis')(diagger)
        
        base_diagnosis = diagger(inbound)
        diagnosis = diagger(decoded)
        
        def diagnosis_loss(inp, outp):
            """Axis-wise KL-Div + loss-of-predictor KL-Div"""
            
            import keras.backend as K
            # penalizes loss of predictive information after compression, *NOT* an incorrect prediction (penalized by Decoder loss)
            reconstruction_loss = K.categorical_crossentropy(base_diagnosis, diagnosis)
            
            return reconstruction_loss
            
        Autoencoder.custom_loss = diagnosis_loss
            
        Autoencoder = cls.DLmodel(inputs=[inbound], outputs=[decoded, diagnosis, base_diagnosis], name='Autoencoder')
        
        Diagnostician=cls.DLmodel(inputs=[inbound], outputs=[base_diagnosis], name='Diagnostician')
        # dummy input for feeding into Decoder separately
        #dummy_in = keras.layers.Input([Encoder.layers[-1].output_shape[-1]])
        #Decoder = cls.DLmodel(dummy_in, dec_start(dummy_in), name='Decoder') #this is bullshit rn

        return (Autoencoder, Encoder, Diagnostician)
        
        
class variational_deep_AE(labeled_AE):

    @classmethod
    def input_preprocessing(cls, input_lay, *args, **kwargs):
            transformed = input_lay
            #transformed = keras.layers.BatchNormalization()(transformed)
            #transformed = keras.layers.AlphaDropout(0.15)(transformed)
            #transformed = keras.layers.Softmax()(transformed)
            return transformed

    @classmethod
    def build(cls, datashape, activators=None, compression_fac=None, latent_dims=2, **kwargs):
        import keras
        activators = activators or {'deep': 'selu', 'regression': 'linear'}
        uncompr_size, compr_size = cls.calculate_sizes(datashape, compression_fac)
        
        # deep levels handling:
        deep_lvls = kwargs.get('depth', 1)
        try: deep_lvls = max(1, abs(int(deep_lvls)))
        except Exception: deep_lvls = 1
        
        depth_scaling = kwargs.get('depth_scaling') or 1.2 # layer size increase rate for each deep level between representation and endpoints
        print("Depth scaling: {}\n".format(depth_scaling))
        
        clamp_size = lambda S: max(1, min(S, uncompr_size-1))
        lay_sizes = [clamp_size(round(compr_size * (depth_scaling**lvl))) for lvl in reversed(range(deep_lvls))] # FIFO sizes for encoders!
        print(lay_sizes)

        # layers
        # Encoder: vals -> compressed vals
        inbound = keras.layers.Input(shape=([datashape[-1]] or [datashape[0]]), name='expression_in')
        last_lay = inbound
        
        preprocessed_inp = cls.input_preprocessing(last_lay, **kwargs)
        last_lay = preprocessed_inp
        
        encoding_nodes = []
        decoding_nodes = []
        
        cust_layers = {}#1: {'sizemult':1, 'activation': 'tanh'}}
        
        for (i, siz) in enumerate(lay_sizes[:-1]):
            print(str(i)+':', siz or "NONE!")
            if False or i in cust_layers:
                sizemult = cust_layers.get(i, {}).get('sizemult') or 1
                cust_act = cust_layers.get(i, {}).get('activation')
                
                #split_layers = []
                #for spl in range(splits):
                encoder_node = keras.layers.Dense(siz*sizemult, activation=cust_act or activators.get('deep', 'selu'), 
                                                              kernel_initializer='lecun_normal',
                                                              name='encoder_cust_{}'.format(i),
                                                              #name='encoder_{}-{}'.format(i,spl+1)
                                                              use_bias=False,
                                                             )
                split = encoder_node(last_lay)
                    #split = keras.layers.AlphaDropout(0.1)(split)
                    #split_layers.append(split)
                #merged = keras.layers.concatenate(split_layers)
                last_lay = split
                
            else:
                encoder_node = keras.layers.Dense(siz, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal',
                                                          name='encoder_{}'.format(i)
                                                         )
                last_lay = encoder_node(last_lay)
                
            encoding_nodes.append(encoder_node)
                
        else:
            # VAE stuff:
            enc_mean = keras.layers.Dense(latent_dims, name='encoded_mean'.format())(last_lay)
            inp_mean = keras.layers.Input(shape=(latent_dims,),
                                                  #tensor=keras.backend.zeros(shape=(latent_dims,)), # makes this input optional
                                                  name='inp_mean (optional)',
                                                 )
            enc_mean = keras.layers.add([enc_mean, inp_mean])
            
            enc_logstdev = keras.layers.Dense(latent_dims, name='encoded_log-stdev'.format())(last_lay)
            inp_logstdev = keras.layers.Input(shape=(latent_dims,), 
                                                      #tensor=keras.backend.ones(shape=(latent_dims,)), # makes this input optional
                                                      name='inp_stdev (optional)',
                                                      )
            enc_logstdev = keras.layers.multiply([enc_logstdev, inp_logstdev])
            
            def sampler(distribution_params):
                mean, log_stddev = distribution_params
                import keras.backend as K
                std_norm = K.random_normal(shape=(K.shape(mean)[0], latent_dims), mean=0, stddev=1)
                return mean + K.exp(log_stddev) * std_norm
            
            latent_vector = keras.layers.Lambda(sampler, output_shape=(latent_dims,))([enc_mean, enc_logstdev])
            
            # encoder output layer:
            enc_out_layer = latent_vector
            
        last_lay = enc_out_layer
        
        
        # Decoder: compressed vals -> regression vals
        for (i, siz) in []:#enumerate(reversed(lay_sizes[:-1])):
            #print(i, siz)
            decoder_node = keras.layers.Dense(int(siz), activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
            decoded = decoder_node(last_lay)
            decoding_nodes.append(decoder_node)
            last_lay = decoded
        
        # tiny layers for logical & nonlinear separation of the latent space
        if False:
            for i in range(4):
                decoder_node = keras.layers.Dense(5, activation=activators.get('deep', 'selu'), kernel_initializer='lecun_normal', name='decoder_{}'.format(i))
                decoded = decoder_node(last_lay)
                decoding_nodes.append(decoder_node)
                last_lay = decoded
            
        # let's make sure the last layer is 1:1 to input no matter what
        #last_lay = keras.layers.BatchNormalization()(last_lay)
        decoder_node = keras.layers.Dense(encoding_nodes[0].input_shape[-1], 
                                                    activation=activators.get('regression', 'linear'), 
                                                    kernel_initializer='lecun_normal', 
                                                    name='expression_out',
                                                    use_bias=False,
                                                 )
        decoded = decoder_node(last_lay)
        decoding_nodes.append(decoder_node)
        last_lay = decoded
        
        # Diagnostician: vals -> class; functions as the adversarial component of the model
        diagger = keras.layers.Dense(4, activation='softmax', 
                                                kernel_initializer='lecun_uniform',
                                                use_bias=False,
                                                name='diagnosis'
                                            )
        # raw input diagnosis:
        base_diagnosis = diagger(preprocessed_inp) 
        # corrupted input diagnosis:
        corr_diagnosis = diagger(keras.layers.Dropout(0.05)(preprocessed_inp))
        # generated output diagnosis:
        diagnosis = diagger(decoded)
        
        default_optimizer = keras.optimizers.RMSprop(lr=0.000003)
        def top_2_accuracy(y_true, y_pred): return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
        
        Autoencoder = cls.DLmodel(inputs=[inbound, inp_mean, inp_logstdev], outputs=[decoded, corr_diagnosis, diagnosis], name='Autoencoder')
        Autoencoder.custom_loss = VAE_loss(enc_mean, enc_logstdev, base_diagnosis, diagnosis)
        Autoencoder.compile(
                             optimizer=kwargs.get('AE_optimizer') or default_optimizer, 
                             loss={'diagnosis': 'categorical_crossentropy', 'expression_out': Autoencoder.custom_loss},
                             loss_weights={'diagnosis': 2, 'expression_out': 1,},
                             metrics={'diagnosis': ['binary_accuracy', 'categorical_accuracy', top_2_accuracy]},
                             )
        
        def axiswise_kld_wrapper(*args, **kwargs): return axiswise_kld(enc_mean, enc_logstdev)
        Encoder = cls.DLmodel(inputs=[inbound, inp_mean, inp_logstdev], outputs=[enc_mean, enc_logstdev], name='Encoder')
        Encoder.compile(
                        optimizer=kwargs.get('enc_optimizer') or default_optimizer, 
                        loss=axiswise_kld_wrapper,
                       )
        
        for l in encoding_nodes: l.trainable = False 
        # NO ENCODER TRAINING BELOW:
        
        def recon_loss(*args, **kwargs): return keras.backend.categorical_crossentropy(base_diagnosis, 
                                                                                       diagnosis
                                                                                    )
        Decoder = cls.DLmodel(inputs=[inbound, inp_mean, inp_logstdev], outputs=[decoded, diagnosis], name='Decoder')
        Decoder.compile(
                        optimizer=kwargs.get('enc_optimizer') or default_optimizer, 
                        loss=recon_loss,
                       )
        
        for l in decoding_nodes: l.trainable = False 
        # NO AE TRAINING BELOW:
        
        Diagnostician=cls.DLmodel(inputs=[inbound, inp_mean, inp_logstdev], outputs=[decoded, base_diagnosis], name='Diagnostician')
        Diagnostician.custom_loss = VAE_loss(enc_mean, enc_logstdev, base_diagnosis, diagnosis)
        Diagnostician.compile(
                             optimizer=kwargs.get('AE_optimizer') or default_optimizer, 
                             loss={'diagnosis': 'categorical_crossentropy', 'expression_out': Diagnostician.custom_loss},
                             loss_weights={'diagnosis': 18, 'expression_out': 2,},
                             metrics={'diagnosis': ['binary_accuracy', 'categorical_accuracy', top_2_accuracy]},
                             )

        return Autoencoder, Encoder, Diagnostician, Decoder
        
    @staticmethod
    def batchgen(source, catlabels, batch_size=3, with_name=False, seek_novel=True): 
        """Defines data to retrieve from inputs on a per-model basis.
        :param source: an iterable of data
        :param catlabels: category labels
        :param batch_size: *starting* batch size; you may set new size on the returned generator without reinitializing
        """
        import keras.backend as K
        import numpy as np
        
        def batcher():
            _size = max(1, batch_size)
            x_cache = set()
            sentinel, tries = True, 0
            while source:
                batch_expressions = None
                batch_diagnoses = None
                batch_means = None
                batch_stdevs = None
                batch_names = []
                
                for i in range(_size):
                    sentinel = False
                    x = next(source)
                    curr_batch_name = x.columns[-1]
                    if seek_novel and curr_batch_name in x_cache: 
                        tries += 1
                        if tries < 5: continue
                        else: 
                            x_cache.pop()
                            tries = 0
                    sentinel = True
                    x_cache |= set([curr_batch_name])
                    while len(x_cache) > 20: x_cache.pop()
                    tries = 0
                    

                    expression = K.variable(x.sort_index().T.values)
                    raw_expr = K.get_value(expression)
                    xmax, xmin = raw_expr.max(), raw_expr.min()

                    expression, expr_mean, expr_std = SDutils.inp_batch_norm(raw_expr)
                    expr_mean, expr_std = [[0, 0]], [[1, 1]] # for prediction; not the actual mean/var of sample
                    #print("BATCH XPR: ", expression)
                    expression = K.variable(expression)
                    expr_mean, expr_std = K.variable(np.array(expr_mean)), K.variable(np.array(expr_std))
                    
                    diagnosis = catlabels.get(str(x.index.name).upper(), [0,0,0,0])
                    #print(x.index.name)
                    try: diagnosis = K.variable(diagnosis)
                    except ValueError: diagnosis = K.variable([[0,0,0,0]])
                    
                    batch_diagnoses = K.concatenate([batch_diagnoses, diagnosis], axis=0) if batch_diagnoses is not None else diagnosis
                    batch_expressions = K.concatenate([batch_expressions, expression], axis=0) if batch_expressions is not None else expression
                    batch_means = K.concatenate([batch_means, expr_mean], axis=0) if batch_means is not None else expr_mean
                    batch_stdevs = K.concatenate([batch_stdevs, expr_std], axis=0) if batch_stdevs is not None else expr_std
                    
                    if with_name: batch_names = batch_names + [curr_batch_name]
                    
                if not sentinel: continue
                #print("\n", "BATCH: ", "\n", K.eval(batch_expressions), "\n", K.eval(batch_diagnoses), "\n",)
                #if input('Cont.?'): raise RuntimeError
                #print("Generating a batch...")
                inp_means, inp_stdevs = K.eval(batch_means), K.eval(batch_stdevs) # for DRY purposes
                    
                batch = ((
                            {
                             'expression_in': K.eval(batch_expressions),
                             'diagnosis_in': K.eval(batch_diagnoses),
                             # aliases due to naming issues
                             'inp_mean': inp_means,
                             'inp_mean (optional)': inp_means,
                             'input_1': inp_means,
                             'inp_stdev': inp_stdevs,
                             'inp_stdev (optional)': inp_stdevs,
                             'input_2': inp_stdevs
                            },
                            {
                             'expression_out': K.eval(batch_expressions),
                             'diagnosis': K.eval(batch_diagnoses),
                            }
                        ))
                #print(batch)
                new_size = yield ((batch, batch_names) if with_name else batch)
                try: new_size = abs(int(new_size))
                except Exception as E: new_size = None
                _size = new_size if new_size and new_size > 0 else _size
                
        return batcher()
    
def axiswise_kld(mean, logSTD):
    import keras.backend as K
    return 0.5 * K.sum(K.exp(logSTD) + K.square(mean) - 1. - logSTD, axis=1)
    
def VAE_loss(enc_mean, enc_logstdev, base_diagnosis, diagnosis, *args, **kwargs):
    def _VAElossFunc(inp, outp):
        """Axis-wise KL-Div + loss-of-predictor penalty"""
        import keras.backend as K
        
        # penalizes non-normal distribution of encodings in latent space
        #kl_loss = -0.5 * K.clip(K.sum(1 - K.square(enc_mean) + enc_logstdev - K.square(K.exp(enc_logstdev+K.epsilon())), axis=-1), -200000, -K.epsilon())
        kl_loss = axiswise_kld(enc_mean, enc_logstdev)
        
        # penalizes loss of predictive information after compression, *NOT* an incorrect prediction (penalized by Decoder loss)
        import keras.losses as kL
        #print("DIAGNOSES DEBUG:")
        #print(K.eval(base_diagnosis))
        #print(K.eval(diagnosis))
        #K.print_tensor(base_diagnosis)
        #K.print_tensor(diagnosis)
        reconstruction_loss = (K.categorical_crossentropy(base_diagnosis, diagnosis) + 0.5*kL.mae(inp, outp)
                               )
        
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss
    return _VAElossFunc
    
def build_models(datashape, which='VAE', activators=None, **kwargs):
    if not activators: activators = {'deep': 'selu', 'regression': 'linear', 'classification': 'softmax'}
    model_dict = {'deepAE': deep_AE, 'deepAE-dropout': deepAE_dropout, 'labeledAE': labeled_AE, 'VAE': variational_deep_AE, 'coherentAE': coherent_AE}
    print(kwargs.get('labels'))
    model_to_build = which if isinstance(which, ExpressionModel) else model_dict.get(which, NotImplemented)
    if model_to_build is NotImplemented: raise NotImplementedError
    built = model_to_build(datashape, activators, **kwargs)
    print("Built model: {} {}".format(which, built))
    return built
    
def main(datashape=None, which_model=None):
    datashape = datashape or eval(input('Enter datashape: '))
    #print("Known models: \n{}".format("\n".join(model_dict.keys())))
    which_model = which_model or input('Enter model type: ')
    build_models(datashape=datashape, which=which_model)
    
if __name__ == '__main__':
    main()
