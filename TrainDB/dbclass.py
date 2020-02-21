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
    def __init__(self, net,dataloader, criterion,\
        dictf=None, dictg=None, norm=2,\
         storeweight=True, storenorm=True, storediffnorm=True,\
          storegrad=True, storetrainloss=True, batchfreq=None, epochfreq=1 ):
        '''
        dictf is a dictionary of \{ fn_name: function definiton \} such that f is a function of weights,
        i.e we will store f(W) for each layer and each iteration

        dictg is a dictionary of \{ fn_name: function definiton \} such that f is a function of weights,
        i.e we will store f(W_i - W_{i-1}) for each layer and each iteration
        '''


        #Setting Parameters
        self.network = copy.deepcopy(net)
        #self.network.load_state_dict(copy.deepcopy(net.state_dict()))
        self.currnetwork = copy.deepcopy(net)
        self.criterion = criterion
        self.dataloader = dataloader
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
        self.dictf = copy.deepcopy(dictf)
        self.dictg = copy.deepcopy(dictg)


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
        if self.dictf is not None:
            self.tdictf = self.createrowf((self.network).state_dict(),self.dictf)
        if self.dictg is not None:
            self.tdictg = self.createrowf((self.network).state_dict(),self.dictg)

    '''
    -------------------------------------------------------------------------------------
                    A set of constructor functions - to build the tables/dicts
    --------------------------------------------------------------------------------------
    '''
    def createrowfromweights(self,weights,layers):
        if self.storeepoch:
             df = pd.DataFrame([weights],  index=[self.currentepoch], columns=layers)
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

    def createrowf(self,n,dictf):
        ''' Returns a row consisting of f(W) for each f in dictf of the weights of every layer in the network n '''
        layers = list(n.keys())
        functions = list(dictf.keys())
        weights = []
        col = []
        for l in layers:
            X = np.array(n[l])
            for f in functions:
                weights.append(dictf[f](X))
                col.append(str(l)+ '-' +str(f))
        return self.createrowfromweights(weights,col)

    def createrowg(self,oldn,newn,dictg):
        ''' Returns a row consisting of f(W) for each f in dictf of the weights of every layer in the network n '''
        layers = list(newn.keys())
        functions = list(dictg.keys())
        weights = []
        col = []
        for l in layers:
            X = np.array(newn[l])
            Y = np.array(oldn[l])
            for g in functions:
                weights.append(dictg[g](X-Y))
                col.append(str(l)+ '-' +str(g))
        return self.createrowfromweights(weights,col)

    def createrownorm(self,n,norm):
        ''' Returns a row consisting of norm of the weights in the network n '''
        layers = list(n.keys())
        weights = []
        for l in layers:
            X = np.array(n[l])
            if np.ndim(X) >2:
                weights.append(np.linalg.norm(X))
            elif np.ndim(X) == 1 and isinstance(norm,str) :
                weights.append(np.linalg.norm(X, 2))
            else:
                weights.append(np.linalg.norm(X, norm))
        return self.createrowfromweights(weights,layers)


    def createrowdiffnorm(self,olddict,newdict,norm):
        layers = list(olddict.keys())
        weights = []
        for l in layers:
            X = np.array(newdict[l])
            Y = np.array(olddict[l])
            if np.ndim(X) > 2:
                weights.append(np.linalg.norm(X-Y))
            elif np.ndim(X) == 1 and isinstance(norm,str) :
                weights.append(np.linalg.norm(X-Y, 2))
            else:
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

    def step(self,epoch,batch_id,prev_net,net,loss):
        self.numiter +=1
        self.currnetwork = copy.deepcopy(net)
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
        if self.dictf is not None:
            self.tdictf = self.tdictf.append(self.createrowf(net.state_dict(),self.dictf))
        if self.dictg is not None:
            self.tdictg = self.tdictg.append(self.createrowg(prev_net,net.state_dict(),self.dictg))
        if self.storegrad:
            grad = []
            for param in net.parameters():
                grad.append(param.grad.view(-1))
            if self.storeepoch:
                self.lgrad[epoch] = [g/(self.numiter*1.0) for g in grad] #grad/(self.numiter*1.0)
            else:
                self.lgrad[(epoch,batch_id)] = [g/(self.numiter*1.0) for g in grad] #grad/#Hard-coded
        if self.trainloss:
            if self.storeepoch:
                self.trainloss[epoch] = loss
            else:
                self.trainloss[(epoch,batch_id)] = loss
            #if self.lgrad is None:
            #    self.lgrad = self.createrowgrad(grad)
            #else:
            #    self.lgrad = self.lgrad.append(self.createrowgrad(grad))

    '''
    -------------------------------------------------------------------------------------
                            A set of helper functions
    --------------------------------------------------------------------------------------
    '''


    def genind(self,epoch,batch_id):
        if epoch is None:
            return "Error" # TODO: Proper error handling   
        if batch_id is None:
            if self.storeepoch:
                return epoch  #Not the best way to write this
            else:
                return "Error"
        return (epoch,batch_id)

    def query(self,df,layer, epoch=None, batch_id=None):
        ''' Returns element at a given epoch number and/or batch_id
        '''
        ind = self.genind(epoch,batch_id)
        if isinstance(ind,tuple):
            x = df.loc[ind,layer].values[0]
        else:
            x = df.loc[ind,layer]
        return x

    def reconstructnet(self,epoch,batch_id):
        '''build network using current weights at iteration i
        '''
        network = copy.deepcopy(self.currnetwork)
        if epoch is None:
            network.zero_grad()
            return network
        d= {}
        for param in list((network.state_dict()).keys()):
            x = self.ithweight(param,epoch,batch_id)
            y = torch.from_numpy(x)
            y.requires_grad = True
            d[param] = y

        network.load_state_dict(d)
        network.zero_grad()
        return network


    '''
    -------------------------------------------------------------------------------------
                            A set of simple data query functions
    --------------------------------------------------------------------------------------
    '''
    def ithweight(self,layer, epoch=None, batch_id=None):
        ''' Returns weight of a specific layer at a specific epoch/batch_id
        '''
        return self.query(self.tweight,layer,epoch,batch_id)
        #return self.tweight.loc[self.genind(epoch,batch_id),layer].values[0]

    def ithnorm(self,layer, epoch=None, batch_id=None):
        ''' Returns norm of a specific layer at a specific epoch/batch_id
        '''
        return self.query(self.tnorm,layer,epoch,batch_id)
        #return self.tnorm.loc[self.genind(epoch,batch_id),layer].values[0]

    def ithdiffnorm(self,layer, epoch=None, batch_id=None):
        ''' Returns difference of a specific layer at a specific epoch/batch_id # Needs some explaination
        '''
        return self.query(self.tdiffnorm,layer,epoch,batch_id)
        #return self.tnorm.loc[self.genind(epoch,batch_id),layer].values[0]

    def ithdictf(self,layer, f_name, epoch=None, batch_id=None):
        ''' Returns difference of a specific layer at a specific epoch/batch_id # Needs some explaination
        '''
        return self.query(self.tdictf,str(layer)+'-'+str(f_name),epoch,batch_id)

    def ithdictg(self,layer, g_name, epoch=None, batch_id=None):
        ''' Returns difference of a specific layer at a specific epoch/batch_id # Needs some explaination
        '''
        return self.query(self.tdictg,str(layer)+'-'+str(g_name),epoch,batch_id)

    def ithgrad(self, epoch=None, batch_id=None):
        ''' Returns the (concatenated) gradient at a specific epoch/batch_id
        '''
        return self.lgrad[self.genind(epoch,batch_id)]


    def ithtrain_accuracy(self,epoch,batch_id):
        return self.trainloss[self.genind(epoch,batch_id)]

    def maxweight(self,layer=None):
        '''Returns row of the table in which the iteration in which norm of the weight was maximum'''
        return

    def maxweightupdate(self,layer):
        return

    #def ithtest_accuracy(self,iteration):
    #    return Should we have it? Too expensive

    '''
    -------------------------------------------------------------------------------------
                            A set of loss landscape related functions
    --------------------------------------------------------------------------------------
    '''

    def ithhess_eigenval(self,epoch=None,batch_id=None,k=1):
        '''Returns top-k eigenvalues of the Hessian of the loss surface at iteration corresponding to the epoch and iteration
            If epoch number not provided, uses the current model
        '''
        #'''
        #eigenvals, eigenvecs = compute_hessian_eigenthings(self.network, self.lgrad[self.genind(epoch,batch_id)],num_eigenthings=k,power_iter_steps=100)
        #for i, (inputs, targets) in enumerate(self.dataloader):
        #    inputs, targets = inputs.to(device=self.device, dtype=self.dtype), targets.to(self.device)
        #    loss = self.criterion(network(inputs), targets)
        #    grad_seq = torch.autograd.grad(loss, network.parameters(),only_inputs=True, create_graph=True, retain_graph=True)

        network = self.reconstructnet(epoch,batch_id)
        for batch_idx, (data, target) in enumerate(self.dataloader):
            data = data.view(data.shape[0], -1)  #Remove this line eventually
            output = network(data)
            loss = self.criterion(output, target)/(len(self.dataloader)*1.0)
            loss.backward(create_graph=True,retain_graph=True)
        grads = []
        for param in network.parameters():
            grads.append(param.grad)

        grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
        compute_hessian_eigenthings(network,grad_vec,num_eigenthings=k)

        #hess = HessianOperator(network, grads)
        #self.lgrad[self.genind(epoch,batch_id)]
        #eigenvalue_analysis(hess,k=k,max_iter=20)
        return





    '''
    -------------------------------------------------------------------------------------
                            A set of visualization functions
                            Plot norm, diffnorm, statistics, loss_landscape ??
    --------------------------------------------------------------------------------------
    '''





    '''
    -------------------------------------------------------------------------------------
                            A set of metadata/statistics functions
                            Like memory footprint, max/min (NOT CLEAR WHAT TO DO HERE)
    --------------------------------------------------------------------------------------
    '''
