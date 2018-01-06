import numpy as np
import scipy.spatial.distance
from tempfile import mkdtemp
import os.path as path

#userprofiles is an nx1 vector. Result is an nxn matrix of pairwise similarity based on cosine similarity
def pwsimcosine_large(usertags, maxusertag,filename):
   numRows = len(usertags)
   #matrixsim = 0.0 * np.arange(numRows*numRows)
   #filename = path.join(mkdtemp(), filename)
   matrixsim = np.memmap(filename, dtype='float32', mode='w+', shape=(28763,28763))

   #print(numRows)

   #matrixsim.shape = (numRows,numRows)   
   for i in range(0,numRows-1):
      print('i' + str(i))
      for j in range(i+1,numRows):
         #print('j' + str(j))
         matrixsim[i,j] = vectorize2compare(usertags[i],usertags[j],maxusertag)
         matrixsim[j,i] = matrixsim[i,j]
      matrixsim[i,i] = 1
   return matrixsim   

#userprofiles is an nx1 vector. Result is an nxn matrix of pairwise similarity based on cosine similarity
def pwsimcosine_user2user_mp(usertags, maxusertag, filename, startcol, endcol):
   #numRows = len(usertags)
   #matrixsim = 0.0 * np.arange(numRows*numRows)
   #filename = path.join(mkdtemp(), filename)
   matrixsim = np.memmap(filename, dtype='float32', mode='w+', shape=(28763,28763))

   #print(numRows)

   #matrixsim.shape = (numRows,numRows)   
   for i in range(startcol,endcol):
      print('i' + str(i))
      for j in range(0,i):
         #print('j' + str(j))
         matrixsim[i,j] = vectorize2compare(usertags[i],usertags[j],maxusertag)
         matrixsim[j,i] = matrixsim[i,j]
      matrixsim[i,i] = 1
   return matrixsim

#userprofiles is an nx1 vector. Result is an nxn matrix of pairwise similarity based on cosine similarity
def pwsimcosinetest(usertags, maxusertag):
   numRows = len(usertags)
   matrixsim = 0.0 * np.arange(numRows*numRows)
   #matrixsim = np.memmap(filename, dtype='int16', mode='w+', shape=(numRows,tagrange+1))
   matrixsim.shape = (numRows,numRows)   
   for i in range(0,numRows):
      for j in range(i+1,numRows):
         matrixsim[i,j] = vectorize2compare(usertags[i],usertags[j],maxusertag)
         matrixsim[j,i] = matrixsim[i,j]
      matrixsim[i,i] = 1
   return matrixsim   

#Takes two inputs, e.g "1/3/4/6/7" and "3/6/7/9/10" and compute cosine similarity between the two
def vectorize2compare(tagA, tagB, maxtag):
   if (tagA == '/' or tagA == ''):
      return 0
   if (tagB == '/' or tagB == ''):
      return 0
   vecA = tag2vector(tagA,maxtag)
   vecB = tag2vector(tagB,maxtag)
   if (np.count_nonzero(vecA) == 0):
      return 0
   if (np.count_nonzero(vecB) == 0):
      return 0
   return 1 - scipy.spatial.distance.cosine(vecA, vecB)

#Takes an input, e.g "1/3/4/6/7" and converts it into sparse vector representation
def tag2vector (tag, maxtag):
   vec = 0 * np.arange(maxtag+1) #if 142 is max tag, you need 143 array size to hold from 0-142
   if (tag == '/' or tag == ''):
      return vec
   tags = tag.split('/')
   tags = map(int, tags)
   tags = np.array(tags)
   for indx in range(len(tags)):
      vec[tags[indx]] = vec[tags[indx]] + 1
   return vec

#usertags is an nx1 vector and each entry is a / separated string of numbers.
#Finds the maximum number or highest value of tag
def maxtagvalue(usertags):
   
   numRows = len(usertags)
   #find range of user tags
   tagrange = 0
   for indx in range(len(usertags)):
      if (usertags[indx] == '/'):
         continue 
      tags = usertags[indx].split('/')
      #print(tags == '/')
      #print(indx)
      tags = map(int, tags)
      #print('max here')
      temp = max(tags)
      if (temp > tagrange):
         tagrange = temp
   return tagrange


def normalize(X_tr, X_te):
	''' Normalize training and test data features
	Args:
		X_tr: Unnormalized training features
		X_te: Unnormalized test features
	Output:
		X_tr: Normalized training features
		X_te: Normalized test features
	'''
	X_mu = np.mean(X_tr, axis=0)
	X_tr = X_tr - X_mu
	X_sig = np.std(X_tr, axis=0)
	X_tr = X_tr/X_sig
	X_te = (X_te - X_mu)/X_sig
	return X_tr, X_te

def tokens(x):
   return x.split('/')


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])



