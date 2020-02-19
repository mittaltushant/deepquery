import pandas as pd
import numpy as np
import torch
import copy
from hessian_eigenthings import compute_hessian_eigenthings
from hvp import *

class TrainDB:
    '''
    -------------------------------------------------------------------------------------
                            Initialization function
    --------------------------------------------------------------------------------------
    '''

    # Initializer / Instance Attributes
    def __init__(self, net, norm=2, storeweight=True, storenorm=True, storediffnorm=True, batchfreq=None, storegrad=True, storetrainloss=True, epochfreq=1 ):

        #Setting Parameters
        self.network = copy.deepcopy(net)
        #self.network.load_state_dict(copy.deepcopy(net.state_dict()))
        self.currnetwork = copy.deepcopy(net)
        self.storegrad = storegrad
        self.storeweight = storeweight
        self.storenorm = storenorm
        self.storediffnorm = storediffnorm
        self.storeepoch = False
        if batchfreq is None:
            self.storeepoch = True #If we don't need to store data for every batch, just store it once for each epoch
        self.epochfreq = epochfreq
        self.batchfreq = batchfreq
        self.storetrainloss = storetrainloss
        self.currentepoch = 0 #Store the current batch number
        self.currentbatch = 0
        self.norm = norm
        self.numiter = 0

        #Initialzing tables and dictionaries
        if storeweight:
            self.tweight = self.createrowfull((self.network).state_dict())
        if storenorm:
            self.tnorm = self.createrownorm((self.network).state_dict(),self.norm)
        if storediffnorm:
            self.tdiffnorm = self.createrownorm((self.network).state_dict(),self.norm)
        if storegrad:
            #self.lgrad = None
            self.lgrad = dict()
        if storetrainloss:
            self.trainloss = dict()
    '''
    -------------------------------------------------------------------------------------
                    A set of constructor functions - to build the tables/dicts
    --------------------------------------------------------------------------------------
    '''
    def createrowfromweights(self,weights,layers):
        if self.storeepoch:
             df = pd.DataFrame([weights],  index=self.currentepoch, columns=layers)
        else:
            ind = [(self.currentepoch,self.currentbatch)]
            df = pd.DataFrame([weights],  index=pd.MultiIndex.from_tuples(ind), columns=layers)
        return df

    def createrowfull(self,n):
        ''' Returns a row consisting of weights in the network n '''
        layers = list(n.keys())
        weights = []
        for l in layers:
            weights.append(np.array(n[l]))
        return self.createrowfromweights(weights,layers)

    def createrownorm(self,n,norm):
        ''' Returns a row consisting of norm of the weights in the network n '''
        layers = list(n.keys())
        weights = []
        for l in layers:
            X = np.array(n[l])
            if np.ndim(X) >2:
                #print('A')
                #print(np.linalg.norm(X))
                weights.append(np.linalg.norm(X))
            elif np.ndim(X) == 1 and isinstance(norm,str) :
                #print('B')
                #print(np.linalg.norm(X,2))
                weights.append(np.linalg.norm(X, 2) )
            else:
                #print('C')
                #print(np.linalg.norm(X,norm))
                weights.append(np.linalg.norm(X, norm))
        return self.createrowfromweights(weights,layers)


    def createrowdiffnorm(self,olddict,newdict,norm):
        layers = list(olddict.keys())
        weights = []
        for l in layers:
            X = np.array(newdict[l])
            Y = np.array(olddict[l])
            if np.ndim(X) > 2:
                #print('D')
                #print(np.linalg.norm(X-Y))
                weights.append(np.linalg.norm(X-Y))
            elif np.ndim(X) == 1 and isinstance(norm,str) :
                #print('E')
                #print(np.linalg.norm(X-Y),2)
                weights.append(np.linalg.norm(X-Y, 2) )
            else:
                #print('F')
                #print(np.linalg.norm(X-Y),2)
                weights.append(np.linalg.norm(X-Y, norm))
        return self.createrowfromweights(weights,layers)


    def createrowgrad(self,grad):
        ''' Very inefficient - Discarded for now'''
        return pd.DataFrame([grad])

    '''
    -------------------------------------------------------------------------------------
                            The main step function
    --------------------------------------------------------------------------------------
    '''

    def step(self,epoch,batch_id,prev_net,net,grad,loss):
        self.numiter +=1
        if self.storeepoch:
            if epoch == self.currentepoch:
                return
        elif batch_id % self.batchfreq != 0 :
            return
        self.currentepoch = epoch
        self.currentbatch = batch_id
        if self.storeweight:
            self.tweight = self.tweight.append(self.createrowfull(net.state_dict()))
        if self.storenorm:
            self.tnorm = self.tnorm.append(self.createrownorm(net.state_dict(),self.norm))
        if self.storediffnorm:
            self.tdiffnorm = self.tdiffnorm.append(self.createrowdiffnorm(prev_net,net.state_dict(),self.norm))
        if self.storegrad:
            if self.storeepoch:
                self.lgrad[epoch] = [g/(self.numiter*1.0) for g in grad] #grad/(self.numiter*1.0)
            else:
                self.lgrad[(epoch,batch_id)] = [g/(self.numiter*1.0) for g in grad] #grad/#Hard-coded
        if self.trainloss:
            if self.storeepoch:
                self.trainloss[epoch] = loss
            else:
                self.trainloss[(epoch,batch_id)] = loss
        self.currnetwork = copy.deepcopy(net)
            #if self.lgrad is None:
            #    self.lgrad = self.createrowgrad(grad)
            #else:
            #    self.lgrad = self.lgrad.append(self.createrowgrad(grad))

    '''
    -------------------------------------------------------------------------------------
                            A set of data query functions
    --------------------------------------------------------------------------------------
    '''


    def genind(self,epoch,batch_id):
        if epoch is None:
            return "Error" # Must do roper error handling
        if batch_id is None:
            if self.storeepoch:
                return epoch  #Not the best way to write this
            else:
                return "Error"
        return (epoch,batch_id)

    def query(self,df,layer, epoch=None, batch_id=None):
        ''' Returns element at a given epoch number and/or batch_id
        '''
        #if iteration is not None:
        #    return df[layer][iteration]
        return df.loc[self.genind(epoch,batch_id),layer]

    def ithweight(self,layer, epoch=None, batch_id=None):
        ''' Returns weight of a specific layer at a specific epoch/batch_id
        '''
        #return self.query(self.tweight,layer,epoch,batch_id)
        return self.tweight.loc[self.genind(epoch,batch_id),layer]

    def ithnorm(self,layer, epoch=None, batch_id=None):
        ''' Returns norm of a specific layer at a specific epoch/batch_id
        '''
        return self.tnorm.loc[self.genind(epoch,batch_id),layer]

    def ithdiffnorm(self,layer, epoch=None, batch_id=None):
        ''' Returns difference of a specific layer at a specific epoch/batch_id # Needs some explaination
        '''
        return self.tnorm.loc[self.genind(epoch,batch_id),layer]

    def ithgrad(self, epoch=None, batch_id=None):
        ''' Returns the (concatenated) gradient at a specific epoch/batch_id
        '''
        return self.lgrad[self.genind(epoch,batch_id)]


    def ithtrain_accuracy(self,epoch,batch_id):
        return self.trainloss[self.genind(epoch,batch_id)]

    #def ithtest_accuracy(self,iteration):
    #    return Should we have it? Too expensive

    def ithhess_eigenval(self,epoch,batch_id,k=1):
        '''Returns top-k eigenvalues of the Hessian of the loss surface at iteration corresponding to the epoch and iteration'''


        #network =  #build network using current weights at iteration i
        #eigenvals, eigenvecs = compute_hessian_eigenthings(self.network, self.lgrad[self.genind(epoch,batch_id)],num_eigenthings=k,power_iter_steps=100)
        hess = HessianOperator(self.currnetwork, self.lgrad[self.genind(epoch,batch_id)] )
        eigenvalue_analysis(hess,k=k,max_iter=20)
        return

    def maxweight(self,layer=None):
        '''Returns row of the table in which the iteration in which norm of the weight was maximum'''
        return

    def maxweightupdate(self,layer):
        return



    '''
    -------------------------------------------------------------------------------------
                            A set of visuzlization functions
                            Plot norm, diffnorm, statistics, loss_landscape ??
    --------------------------------------------------------------------------------------
    '''





    '''
    -------------------------------------------------------------------------------------
                            A set of metadata/statistics functions
                            Like memory footprint, max/min (NOT CLEAR WHAT TO DO HERE)
    --------------------------------------------------------------------------------------
    '''
