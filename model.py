import sys
import pickle
import dparser as geo
        
def fetch_batch(stream, batch_size=10, aggregator=list, *args, **kwargs):
    """A generic buffer that reads and yields N values from a passed generator as an iterable.
    If the stream ends prematurely, returns however many elements could were read.
    
    :param stream: a generator or any another object supporting the next() protocol
    :param batch_size: optional; how many values should be fetched, defaults to 10. send() arguments can change its value
    :param aggregator: optional; callable constructor for the iterable type to return data in. 
                                 args and kwargs are passed as the arguments of the call. Default: list
    """
    bsize = batch_size
    while stream:
        batch = aggregator(*args, **kwargs)
        for _ in range(batch_size):
            #print (next(stream))
            try: batch += aggregator((next(stream),))
            except StopIteration: 
                stream = False
                break
        bsize = yield batch
        
def feed_data(batch_size=10, compressed=32, xml=None, txt=None):
    datastream = geo.combo_pipeline(xml_path=xml, txt_path=txt)
    batch = fetch_batch(datastream, batch_size)
    for data in batch:
        #print(data)
        if not data: continue
        #input_shape = min(series.shape for series, ptype in data)
        print("\n".join([" - ".join([series.keys()[0], ptype]) for (series, ptype) in data]) )
        #note to self - can use this part to sample the subpopulations proportionately
        
def build_models(datashape, compression_fac=32, activators=('relu', 'sigmoid')):
    import keras
    # A bit more unorthodox than from m import as...
    Model = keras.models.Model
    
    #calculate sizes
    uncompr_size = datashape[0]
    compr_size = min(uncompr_size, uncompr_size // compression_fac)
    
    #layers
    inbound = keras.layers.Input(shape=datashape)
    dummy_in = keras.layers.Input(shape=datashape) # dummy input for feeding into decoder separately
    encoded = keras.layers.Dense(compr_size, activation=activator[0])(inbound)
    decoded = keras.layers.Dense(uncompr_size, activation=activator[1])(encoded)
    
    #models
    encoder = Model(inbound, encoded)
    autoencoder = Model(inbound, decoded)
    decoder = Model(dummy_in, autoencoder.layers[-1](dummy_in))
    
    return (autoencoder, encoder, decoder)   
    

def main():
    #models = build_models()
    #autoencoder = models[0].compile(optimizer='adadelta', loss='binary_crossentropy')
    feed_data(xml='./mag/*.xml', txt='./mag/*.txt')

if __name__ == '__main__':
    sys.exit(main())
