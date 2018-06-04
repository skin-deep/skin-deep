"""Stores mappings of sample labels to class names."""
# really could just as well be a JSON...
# separate module for convenience of reimporting

# identity - as a basis for building others and just in case it ever needs modification
minimal = {
           'PP':'PP', 
           'PN':'PN', 
           'NN':'NN',
          }

# most/all all together
primary = {
            '<other>':'NN', 
            'Severe':'PP', 'Mild':'PP', 'Normal':'NN', 
            'Control':'NN', 'Involved':'PP', 
            'Uninvolved':'PN', 'Lesion':'PN', 
            '_Lesional':'PP', '_Non-lesional':'PN', 
            'NL':'PN', 'NL\Z':'NN', 
            'LS':'PP', 'LS\Z':'PP',
            'acne':'NN', 
            'allergic':'NN', 'controlsubject':'NN',
            'Day':'PN', 'Week': 'PP',
            'sun-protected':'NN',
          }
          
fourclass  = {
            #'_PP_':'PP', '_PN_':'PN', '_NN_':'NN',
            '_PP':'PP', '_PN':'PN', '_NN':'NN',
            'Severe':'PP', 'Mild':'PP', 'Normal':'NN', 
            'Control':'NN', 'Involved':'PP', 
            #'Uninvolved':'PN', 'Lesion':'PN', 
            '_Lesional':'PP', '_Non-lesional':'PN', 
            #'NL':'PN', 
            #'\bNL':'PN', 
            '_NL':'PN',
            #'LS':'PP', 
            #'\bLS':'PP', 
            '_LS':'PP',
            'acne.lesion':'IN', 'acne.non':'NN',
            #'allergic':'IN', 'controlsubject':'NN',
            #'Day':'IN', 'Week': 'PP',
            'Brain':'NN', 'Blood': 'NN', 
            #'cell line':'NN', 
            'cell': 'NN',
            #'wound':'NN',
            #'sun-protected':'NN', 'sun-exposed':'NN',
            'AAB-.*?-Normal':'NN', 'AAB-.*?-NL':'NN', 'AAB-.*?-L':'IN', 
          }
          
default = fourclass