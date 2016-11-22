#!/usr/bin/env python
'''
A simulator for a network of rate-neurons written in Python
 by Aditya Gilra in Jan 2016.
'''

# import modules and functions to be used
import numpy as np

# The numpy seed should be set by the script calling this simulator
# Currently, there is no provision to set an independent seed just for this simulator.

# ###########################################
# Umbrella class Simulator
# ###########################################

class Model():
    ''' an umbrella class that holds neuron groups, synapses, probes
        and steps through the simulation
    '''
    def __init__(self,dt):
        self.trange = np.array([0.])
        self.t = 0.                     # current simulation time
        self.stepi = 0                  # current step number (int)
        self.treport = 0.               # last time reported / printed
        self.dt = dt                    # simulation time step
        self.groups = {}
        self.synapses = {}
        self.probes = {}
    
    def addGroup(self,group):
        self.groups[group.name] = group
    
    def addSynapses(self,synapses):
        self.synapses[synapses.name] = synapses
        
    def addProbe(self,probe):
        self.probes[probe.name] = probe
    
    def step(self,dt):
        for synapse in self.synapses.values():
            synapse.step(dt)
        for group in self.groups.values():
            group.step(dt)
        for probe in self.probes.values():
            probe.step(self.stepi)
            
    def simulate(self,tsim):
        tnows = np.arange(self.t+self.dt,self.t+tsim,self.dt)
        for probe in self.probes.values():
            probe.addBlanks(len(tnows))
        for self.t in tnows:
            self.step(self.dt)
            self.stepi += 1
            if self.t >= self.treport+1e4*self.dt:
                print 'Simulated for ',self.t,'s'
                self.treport = self.t
        self.trange = np.append(self.trange,tnows)

# ###########################################
# Neuron Groups
# ###########################################

# decaying rate neuron
# equation: dinp/dt = 1/tau*(-inp + W.r_pre)
#           r = maxR*(tanh(inp/linRange + bias)+1.)/2.

class RateNeuronGroup():
    def __init__(self,N,taum,maxR,bias,linRange,model,name):
        self.N = N
        self.r = np.zeros(N)
        self.inp = np.zeros(N)
        self.taum = taum                # s # Membrane time constant
        self.bias = bias                # constant bias input to all neurons
        self.maxR = maxR                # Hz # max firing rate
        self.linRange = linRange        # Hz # width of the sigmoid transition
        self.model = model
        self.name = name
        model.addGroup(self)
        
    def incrInp(self,deltaInp):
        self.inp += deltaInp/self.taum  # deltaInp should have dt* already
    
    def step(self,dt):
        self.inp -= dt/self.taum*self.inp
        # use only positive part of tanh i.e. for inp>0
        # we need positive rates of course,
        #  but not by shifting the tanh()
        #   so that we have a negative 'buffer' zone for inp<0
        #   i.e. inp<0 (purely inhibitory input) doesn't affect the rates
        self.r = self.maxR*(np.tanh(\
                    np.clip(self.inp/self.linRange+self.bias,0.,None) ))

class ConstantRateNeuronGroup():
    def __init__(self,N,rates=0.):
        ''' fixed rate neuron group
            rates can be a scalar or an ndarray of size N
        '''
        self.N = N
        self.r = rates*np.ones(N)

# ###########################################
# Synaptic connections between Groups
# ###########################################

class Synapses():
    def __init__(self,groupPre,groupPost,model,name):
        self.groupPre = groupPre
        self.groupPost = groupPost
        self.W = np.zeros(shape=(groupPost.N,groupPre.N))
        self.Wconnected = np.zeros(shape=(groupPost.N,groupPre.N))
        self.model = model
        self.name = name
        model.addSynapses(self)
    
    def connect(self,numPerNrn,w):
        # for each post neuron
        for nrni in range(self.groupPost.N):
            # permute the indices of pre neurons and choose numPerNrn
            #  i.e. just sample (without replacement) numPerNrn from pre neurons
            self.Wconnected[ nrni, np.random.permutation(self.groupPre.N)[:numPerNrn] ] = 1
        self.W = self.Wconnected*w
    
    def step(self,dt):
        self.groupPost.incrInp(dt*np.dot(self.W,self.groupPre.r))

class SynapsesPlasticBCM(Synapses):
    def __init__(self,groupPre,groupPost,model,name,\
                    learningRate,kernelPotnIntegral,kernelDepnIntegral,\
                    slowPostKernelIntegral,wmax,\
                    wsyn0,purepostthresh,pureposteta,pureprethresh):
        ''' deltaW = dt * learningRate * ratePre * ratePost * \
                        (ratePost * slowPostKernelIntegral + kernelIntegral)
            *w/wysn0 if wsyn0 is not None (multiplicative plasticity)
            - dt*pureposteta*(ratePost>purepostthresh)*(ratePre<pureprethresh)
            kernelIntegral is usually negative (inhibition dominated STDP)
             to get a positive fixed point for ratePost
        '''
        Synapses.__init__(self,groupPre,groupPost,model,name)
        self.learningRate = learningRate
        self.kernelPotnIntegral = kernelPotnIntegral
        self.kernelDepnIntegral = kernelDepnIntegral
        self.slowPostKernelIntegral = slowPostKernelIntegral
        self.wmax = wmax
        self.wsyn0 = wsyn0
        self.pureposteta = pureposteta
        self.purepostthresh = purepostthresh
        self.pureprethresh = pureprethresh

    def step(self,dt):
        # only the Wconnected == 1 (connected) synapses are modified
        # deltaW_ji = r_j*r_i               # j is post, i is pre
        deltaW = self.Wconnected * dt * self.learningRate * \
                            np.outer(self.groupPost.r,self.groupPre.r)
                                            # pair-based plasticity
        # deltaWTriplet_ji = r_j*r_i*r_j(slow)
        deltaWTriplet = deltaW * np.tile(self.groupPost.r,(self.groupPre.N,1)).T\
                            * self.slowPostKernelIntegral * self.kernelPotnIntegral
                                            # the BCM term
                                            # tile and then element-wise multiplication
        
        # deltaW_ji = r_j*r_i*(kernelIntegral)
        # kernelIntegral should be negative for a positive fixed point (unstable)
        if self.wsyn0 is None:              # additive plasticity
            deltaW *= self.kernelPotnIntegral + self.kernelDepnIntegral
        else:                               # multiplicative plasticity
            deltaW *= self.kernelPotnIntegral + self.kernelDepnIntegral\
                                                * self.W/self.wsyn0
        
        # purepost term, chooses between quadratic versus linear rpost dependence
        if self.purepostthresh == 0:
            if self.pureprethresh is None:
                # cubic purepost (heterosynaptic plasticity)
                # deltaW2_ji = - theta3 * r_j^3
                # set theta3 such that there are 2 positive fixed points for rpre>rpre0
                #  and no fixed points for rpre<rpre0
                deltaW2 = - self.Wconnected * dt * self.pureposteta * \
                                    np.tile(self.groupPost.r**3,(self.groupPre.N,1)).T
            else:
                # quadratic purepost (heterosynaptic plasticity) a la Zenke et al 2015
                # deltaW2_ji = - k2 * r_j^4 * (w_ji - wsyn0)
                #  forces w=wsyn0, post rate just adjusts
                # currently works only for multiplicative plasticity, wsyn0 is not None
                deltaW2 = - self.Wconnected * dt * self.pureposteta * (self.W-self.wsyn0) * \
                                    np.outer(self.groupPost.r**4,\
                                                self.groupPre.r<self.pureprethresh)
        elif self.pureposteta is not None:
            # linear purepost only shifts the unstable fixed point
            # deltaW2_ji = - r_j * (r_j>purepostth) * (r_i<purepreth)
            # reduce weight proportional to post, only if post > purepostth, pre < purepreth
            deltaW2 = - self.Wconnected * dt * self.pureposteta * \
                                np.outer(self.groupPost.r*
                                            (self.groupPost.r>self.purepostthresh),\
                                            self.groupPre.r<self.pureprethresh)
        else: deltaW2 = 0.
        
        # deltaWfinal_ji = dt * eta * r_j * r_i * (r_j(slow) + kernelIntegral) \
        #                           - dt * eta2 * r_j * (r_j>purepostth) * (r_i<purepreth)
        self.W += deltaW + deltaWTriplet + deltaW2
        self.W = np.clip(self.W,0.,self.wmax)
                                            # don't allow negative or too high weights
        Synapses.step(self,dt)

# ###########################################
# Probes
# ###########################################

class RateProbe():
    def __init__(self,rateGroup,model,name):
        self.rateGroup = rateGroup
        self.data = None
        self.model = model
        self.name = name
        model.addProbe(self)
        
    def addBlanks(self,numsteps):
        if self.data is None:
            self.data = np.zeros(shape=(numsteps,self.rateGroup.N))
        else:
            self.data = np.append( self.data,
                    np.zeros(shape=(numsteps,self.rateGroup.N)),
                    axis=0 )
        
    def step(self,stepi):
        self.data[stepi,:] = self.rateGroup.r

class InpProbe(RateProbe):        
    def step(self,stepi):
        self.data[stepi,:] = self.rateGroup.inp

class PopRateProbe(RateProbe):
    def addBlanks(self,numsteps):
        if self.data is None:
            self.data = np.zeros(numsteps)
        else:
            self.data = np.append( self.data,
                    np.zeros(numsteps),
                    axis=0 )
        
    def step(self,stepi):
        self.data[stepi] = np.mean(self.rateGroup.r)

class SynapsesMeanProbe():
    def __init__(self,synObj,model,name):
        self.synObj = synObj
        self.data = None
        self.model = model
        self.name = name
        model.addProbe(self)

    def addBlanks(self,numsteps):
        if self.data is None:
            self.data = np.zeros(numsteps)
        else:
            self.data = np.append( self.data,
                    np.zeros(numsteps),
                    axis=0 )
        
    def step(self,stepi):
        self.data[stepi] = np.mean(self.synObj.W)
