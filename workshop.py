from IPython.display import display #Interactive visualization
from IPython.core.display import display, HTML
import ipywidgets as widgets #ditto
from ipywidgets import interact #""


import requests #download files
import tarfile #extracts compressed file


import numpy as np
import os
from collections import deque #high performance list

#Data prep - download

def PDF(filename,page=1,height=300,width=600):
   return HTML('<center><object data="'+filename+'#page='+str(page)+'&navpanes=0&scrollbar=0&toolbar=0" type="application/pdf" width="'+str(width)+'" height="'+str(height)+'"><embed src="'+filename+'" type="application/pdf" /></object></center>')


def Download(address,filename,extract=True):
   '''
   Downloads a file from WWW, stores it as 'filename'.
   If the filenames ends in .tar.gz, and extract is set to True, the file is uncompressed.
   '''
   if not os.path.exists(filename):
      req=requests.get(address,allow_redirects=True)
      open(filename,'wb').write(req.content)

      if filename.endswith("tar.gz") and extract:
          tgz=tarfile.open(filename,"r:gz")
          tgz.extractall()
          tgz.close



#Exercise 2
#You don't need to understand this function. It is just used to convert between the various datatypes
def convert_weights(model,weights):
    structure=model.get_weights()
    #Convert keras weights to numpy's flat weights
    if type(weights)==list:
        return np.concatenate([np.ndarray.flatten(model.get_weights()[i])
                               for i in range(len(model.get_weights()))])
    else:
        #Converts from numpy array to keras array
        new_weights=[]
        data=deque(weights)
        data.reverse()
        for i in range(len(model.get_weights())):
            shape=model.get_weights()[i].shape
            new_weights.append(np.array([data.pop()
                                     for _ in range(np.prod(shape))]).reshape(shape))
        return new_weights

def generate_ground_truth(x,y):
    return np.sin(x)+2*np.cos(x)+3*np.cosh(y)


def optimize_cg(parameters,model,X,Y,values):
    model.set_weights(convert_weights(model,parameters))
    loss=np.sqrt(np.sum(np.square(model.predict(X)-Y.reshape([len(X),1])))/len(X))
    values.append(loss)
    return float(loss)
