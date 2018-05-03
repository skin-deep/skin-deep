import collections as coll


LABEL_SAMPLE_SIZE = 'label_sample_size'
SAVE_EVERY = 'save_every'
SAVE_TAILS = 'save_low_epochs'

which_model = 0 # TEMPORARY VAR, REMOVE ME!

class MenuConfiguration(object):
    defaults =  (
                    ('train_steps', 30), 
                    ('test_steps', 15), 
                    (LABEL_SAMPLE_SIZE, None), 
                    ('list_cwd', False), 
                    ('model_depth', 3), 
                    ('drop_labels', False),
                    ('compression_fac', 500), #300/60ep@60st is decent enough
                    (SAVE_EVERY, 10),
                    (SAVE_TAILS, False),
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