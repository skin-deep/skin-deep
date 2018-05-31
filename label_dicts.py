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
            'HEK':'NN', 'KC':'NN', 'LCM':'NN',
          }
          
fourclass  = {
            'Severe':'PP', 'Mild':'PP', 'Normal':'NN', 
            'Control':'NN', 'Involved':'PP', 
            'Uninvolved':'PN', 'Lesion':'PN', 
            '_Lesional':'PP', '_Non-lesional':'PN', 
            'NL':'PN', '\bNL':'PN',
            'NS-':'PN',
            'LS':'PP', '\bLS':'PP',
            'acne':'IN', 
            'allergic':'IN', 'controlsubject':'NN',
            'Day':'IN', 'Week': 'PP',
            'HEK-':'NN', 'KC-':'NN', 'LCM-':'IN',
          }
          
default = fourclass


