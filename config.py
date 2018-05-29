import collections as coll


LABEL_SAMPLE_SIZE = 'label_sample_size'
SAVE_EVERY = 'save_every'
SAVE_TAILS = 'save_low_epochs'

which_model = 0 # TEMPORARY VAR, REMOVE ME!

class MenuConfiguration(object):
    options = coll.OrderedDict()
    defaults =  (
                    ('train_steps', 30), 
                    ('test_steps', 10), 
                    ('batch_size', 10),
                    (LABEL_SAMPLE_SIZE, None), 
                    
                    ('model_depth', 9), 
                    ('compression_fac', 920),
                    ('depth_scaling', 1.15),
                    

                    (SAVE_EVERY, 10),
                    (SAVE_TAILS, False),
                    
                    ('usemodel', 0),
                    ('list_cwd', False), 
                    ('drop_labels', False),
                )

    def __init__(self, opts=None, **kwargs):
        self.options = coll.OrderedDict()
        # defaults:
        self.options.update(self.defaults)
        # kwargs are options to use:
        if opts: self.options.update(opts)
        self.options.update(kwargs)
        self.options['cmd'] = None
        self.options.pop('cmd')
        # TODO: clean up the kwargs
        
    def get(self, opt_key, fallback=None):
        option_val = self.options.get(opt_key, fallback)
        return option_val
