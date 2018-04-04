#!/usr/bin/env python
'''
A simulator for a network of rate-neurons written in Python
 by Aditya Gilra in Jan 2016.
'''

# import modules and functions to be used
import numpy as np
import sys

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
        self.objects = {}
    
    def addGroup(self,group):
        if group.name not in self.groups.keys():
            self.groups[group.name] = group
        else:
            raise ValueError('group name '+group.name+' is duplicate')
    
    def addSynapses(self,synapses):
        if synapses.name not in self.synapses.keys():
            self.synapses[synapses.name] = synapses
        else:
            raise ValueError('synapses name '+synapses.name+' is duplicate')
        
    def addProbe(self,probe):
        if probe.name not in self.probes.keys():
            self.probes[probe.name] = probe
        else:
            raise ValueError('probe name '+probe.name+' is duplicate')
        
    def addObject(self,obj):
        if obj.name not in self.objects.keys():
            self.objects[obj.name] = obj
        else:
            raise ValueError('object name '+obj.name+' is duplicate')
    
    def step(self,dt):
        for obj in self.objects.values():
            obj.step(dt)
        for synapse in self.synapses.values():
            synapse.step(dt)
        for group in self.groups.values():
            group.step(dt)
        for probe in self.probes.values():
            probe.step(self.stepi)

    def reset(self):
        for obj in self.objects.values():
            obj.reset()
        for synapse in self.synapses.values():
            synapse.reset()
        for group in self.groups.values():
            group.reset()
        for probe in self.probes.values():
            probe.reset()
            
    def simulate(self,tsim):
        tnows = np.arange(self.t,self.t+tsim,self.dt)
        for probe in self.probes.values():
            probe.addBlanks(len(tnows))
        for t in tnows:
            self.step(self.dt)
            self.t += self.dt
            self.stepi += 1
            if self.t >= self.treport+1e4*self.dt:
                print 'Simulated for ',self.t,'s'
                self.treport = self.t
        self.trange = np.append(self.trange,tnows)

# ###########################################
# Neuron Groups
# ###########################################

class RateNeuronGroup():
    def __init__(self,N,taum,maxR,bias,linRange,model,name,clip=True):
        '''
        Rate neuron with low-pass filtered input.
         Equation:  dinp/dt = 1/tau*(-inp + W.r_pre)
                    r = maxR * tanh(inp/linRange + bias)
         Can set clipping of negative rates True or False.
        '''
        self.N = N
        self.taum = taum                # s # Membrane time constant
        self.bias = bias                # constant bias input to all neurons
        self.maxR = maxR                # Hz # max firing rate
        self.linRange = linRange        # Hz # width of the sigmoid transition
        self.model = model
        self.name = name
        model.addGroup(self)
        if clip:
            self.clip = lambda x,minx,maxx: np.clip(x,minx,maxx)
        else:
            self.clip = lambda x,minx,maxx: x
        self.reset()
        
    def incrInp(self,deltaInp):
        '''CAUTION: Be sure to include dt* in the deltaInp when passing it in,
         else normalization of the low-pass becomes off!'''
        self.inp += deltaInp/self.taum  # deltaInp should have dt* already
    
    def step(self,dt):
        self.inp -= dt/self.taum*self.inp
        # use only positive part of tanh i.e. for inp>0
        # we need positive rates of course,
        #  but not by shifting the tanh()
        #   so that we have a negative 'buffer' zone for inp<0
        #   i.e. inp<0 (purely inhibitory input) doesn't affect the rates
        self.r = self.maxR*(np.tanh(\
                    self.clip(self.inp/self.linRange+self.bias,0.,None) ))
    
    def reset(self):
        self.inp = np.zeros(self.N)
        self.r = np.zeros(self.N)

class ConstantRateNeuronGroup():
    def __init__(self,N,rates=0.):
        ''' fixed rate neuron group
            rates can be a scalar or an ndarray of size N
        '''
        self.N = N
        self.r = rates*np.ones(N)

    def reset(self): pass

    def step(dt): pass

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
        '''Conect each post-neuron with numPerNrn pre-neurons chosen randomly, with weight w'''
        # for each post neuron
        for nrni in range(self.groupPost.N):
            # permute the indices of pre neurons and choose numPerNrn
            #  i.e. just sample (without replacement) numPerNrn from pre neurons
            self.Wconnected[ nrni, np.random.permutation(self.groupPre.N)[:numPerNrn] ] = 1
        self.W = self.Wconnected*w
        
    def connect(self,W,Wconnected=None):
        '''Connect as per given weight matrix W,
         masked by connection matrix Wconnected (default all-to-all)'''
        if Wconnected is None:
            self.Wconnected = np.ones(shape=(self.groupPost.N,self.groupPre.N))
        else:
            self.Wconnected = Wconnected
        self.W = self.Wconnected*W
    
    def step(self,dt):
        self.groupPost.incrInp(dt*np.dot(self.W,self.groupPre.r))

    def reset(self): pass

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

    def reset(self): pass

class SynapsesNodePerturb(Synapses):
    def __init__(self,groupPre,groupPost,model,name,\
                    learningRate=1e-4,rewardDecayTau=0.1,voltageDecayTau=0.1,\
                    minDeltaW=-1e-5,maxDeltaW=+1e-5,clipW=False,minW=-1.,maxW=1.,numStims=3):
        ''' eligibilityTrace += supra-lin-func( pre_rate * (post_memb_pot - running_mean(post_memb_pot)) )
        deltaW = dt * learningRate * eligibilityTrace * (reward - running_mean(reward))
        SI units: learningRate in Hz, rewardDecayTau and voltageDecayTau in s.
        voltageDecayTau must filter out the fast perturbations in RateNeuronGroup.inp,
            so it must be larger than the latter's filtering tau.
        IMPORTANT: self.reward should be updated before self.step() is called,
        pre and post neuron groups should be updated after self.step() is called,
         both since neural input depends on synaptic input
          and weights / eligibility traces should be updated using previous neural rates/voltages.
        Very important for learning only output connections:
         reset the reward and voltage means across trials!
        But when using stimulus specific reward means,
         do not reset the reward-mean across trials!
        '''
        Synapses.__init__(self,groupPre,groupPost,model,name)
        self.learningRate = learningRate
        self.deltaW = np.zeros(shape=(groupPost.N,groupPre.N))
        self.rewardDecayTau = rewardDecayTau# in Miconi 2017 (1-dt/tau)=0.33
        self.voltageDecayTau = voltageDecayTau
        self.minDeltaW = minDeltaW
        self.maxDeltaW = maxDeltaW
        self.clipW=clipW
        self.minW = minW
        self.maxW = maxW
        self.numStims = numStims
        self.reset()

    def step(self,dt):
        # use the weights to update the post neuronal inp-s.
        Synapses.step(self,dt)

        ## diagnostics
        #print "self.deltaW=",self.deltaW
        #if self.model.t>187.5:
        #    print "self.groupPost.inp",self.groupPost.inp
        #    print "self.postVoltageMean",self.postVoltageMean
        #    print "self.deltaW=",self.deltaW
        #    print "self.eligibilityTraces=",self.eligibilityTraces
        #    print "self.groupPre.r",self.groupPre.r
        #if self.model.t>188.:
        #    sys.exit()

        # eligibilityTraces updated with previous time steps Pre and current Post quantities,
        # important for causality
        # since for Pre rates: RateNeuronGroup.step() is called after this,
        # and for Post inp: synaptic input is already calculated just above
        #                   and perturbations in Simulator.objects are called earlier
        self.eligibilityTraces += np.power( np.outer( 
                                                self.groupPost.inp - self.postVoltageMean,
                                                self.groupPre.r ), 3 )

        # only the Wconnected == 1 (connected) synapses are modified
        self.deltaW = self.Wconnected * dt * self.learningRate \
                    * self.eligibilityTraces * (self.reward - self.rewardMean)
        # clipping the deltaW instead of W yields better stability of learning
        self.deltaW = np.clip(self.deltaW,self.minDeltaW,self.maxDeltaW)
        self.W += self.deltaW
        # clip to not allow too high weights
        if self.clipW:
            self.W = np.clip(self.W,self.minW,self.maxW)

        # update the mean reward and mean voltages only after the weights changes
        self.rewardMean = (1-dt/self.rewardDecayTau)*self.rewardMean \
                                + dt/self.rewardDecayTau*self.reward
        self.postVoltageMean = (1-dt/self.voltageDecayTau)*self.postVoltageMean \
                                + dt/self.voltageDecayTau*self.groupPost.inp
        
    def reset(self):
        self.eligibilityTraces = np.zeros(shape=(self.groupPost.N,self.groupPre.N))
        self.stim = 0           # must be an integer
        self.reward = 0.0
        ## Very important for learning only output connections:
        ##  reset the reward and voltage means across trials!
        ## But when using stimulus specific reward means,
        ##  do not reset the reward-mean across trials!
        self.rewardMean = 0.0
        self.postVoltageMean = np.zeros(self.groupPost.N)

class SynapsesNodePerturbStimulusSpecific(SynapsesNodePerturb):
    def __init__(self,groupPre,groupPost,model,name,\
                    learningRate=1e-2,rewardDecayTau=10.,voltageDecayTau=0.1,\
                    minDeltaW=-1e-4,maxDeltaW=+1e-4,clipW=False,minW=-1.,maxW=1.,numStims=3):
        SynapsesNodePerturb.__init__(self,groupPre,groupPost,model,name,\
                    learningRate,rewardDecayTau,voltageDecayTau,\
                    minDeltaW,maxDeltaW,clipW,minW,maxW,numStims)
        self.rewardMean = np.array([0.]*self.numStims)

    def step(self,dt):
        # use the weights to update the post neuronal inp-s.
        Synapses.step(self,dt)

        # eligibilityTraces updated with previous time steps Pre and current Post quantities,
        # important for causality
        # since for Pre rates: RateNeuronGroup.step() is called after this,
        # and for Post inp: synaptic input is already calculated just above
        #                   and perturbations in Simulator.objects are called earlier
        self.eligibilityTraces += np.power( np.outer( 
                                                self.groupPost.inp - self.postVoltageMean,
                                                self.groupPre.r ), 3 )

        # only the Wconnected == 1 (connected) synapses are modified
        self.deltaW = self.Wconnected * dt * self.learningRate \
                    * self.eligibilityTraces * (self.reward - self.rewardMean[self.stim])
        # clipping the deltaW instead of W yields better stability of learning
        self.deltaW = np.clip(self.deltaW,self.minDeltaW,self.maxDeltaW)
        self.W += self.deltaW
        # clip to not allow too high weights
        if self.clipW:
            self.W = np.clip(self.W,self.minW,self.maxW)

        # update the mean reward and mean voltages only after the weights changes
        # decay all rewardMean traces
        for i in range(self.numStims):
            self.rewardMean[i] = (1-dt/self.rewardDecayTau)*self.rewardMean[i]
        # add reward only to the current stimulus rewardMean
        self.rewardMean[self.stim] += dt/self.rewardDecayTau*self.reward
        # post voltage mean
        self.postVoltageMean = (1-dt/self.voltageDecayTau)*self.postVoltageMean \
                                + dt/self.voltageDecayTau*self.groupPost.inp
        
    def reset(self):
        self.eligibilityTraces = np.zeros(shape=(self.groupPost.N,self.groupPre.N))
        self.stim = 0           # must be an integer, used to index rewardMeans
        self.reward = 0.0
        ## Very important for learning only output connections:
        ##  reset the reward and voltage means across trials!
        ## But when using stimulus specific reward means,
        ##  do not reset the reward-mean across trials!
        self.postVoltageMean = np.zeros(self.groupPost.N)

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

    def reset(self): pass

class InpProbe(RateProbe):        
    def step(self,stepi):
        self.data[stepi,:] = self.rateGroup.inp

class ArbProbe(RateProbe):
    '''Probe an arbitrary attribute of a rateGroup object'''        
    def __init__(self,rateGroup,model,name,attrib='inp'):
        RateProbe.__init__(self,rateGroup,model,name)
        self.attrib = attrib
        
    def step(self,stepi):
        self.data[stepi,:] = getattr(self.rateGroup,self.attrib) 

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

class SynapsesProbe():
    """NOTE: len of data for these probes will be slightly larger than actual number of time steps.
    This is because 1 is added to the number of steps needed,
     in case the array falls short due to truncation and accumulation when multiple model.simulate() calls are made.
    Slice and discard the last values, by calculating using the length of the data of any other 1 dt probe.
    """
    def __init__(self,synObj,model,name,probeSteps=1):
        self.synObj = synObj
        self.data = None
        self.model = model
        self.name = name
        self.probeSteps = probeSteps    # record / probe every probeSteps # of time steps
        model.addProbe(self)
        
    def addBlanks(self,numsteps):
        numsteps = numsteps // self.probeSteps + 1      # one extra in case it falls short
        if self.data is None:
            self.data = np.zeros(shape=(numsteps,
                            self.synObj.groupPost.N,
                            self.synObj.groupPre.N))
        else:
            self.data = np.append( self.data,
                    np.zeros(shape=(numsteps,
                            self.synObj.groupPost.N,
                            self.synObj.groupPre.N)),
                    axis=0 )
        
    def step(self,stepi):
        if stepi % self.probeSteps == 0:
            self.data[stepi//self.probeSteps,:,:] = self.synObj.W

    def reset(self): pass

class SynapsesArbProbe(SynapsesProbe):
    """Probe an arbitrary attribute of a Synapses object.
    NOTE: len of data for these probes will be slightly larger than actual number of time steps.
    This is because 1 is added to the number of steps needed,
     in case the array falls short due to truncation and accumulation when multiple model.simulate() calls are made.
    Slice and discard the last values, by calculating using the length of the data of any other 1 dt probe.
    """
    def __init__(self,synObj,model,name,attrib='W',probeSteps=1):
        SynapsesProbe.__init__(self,synObj,model,name,probeSteps)
        self.attrib = attrib
        self.dataShape = getattr(self.synObj,self.attrib).shape

    def addBlanks(self,numsteps):
        numsteps = numsteps // self.probeSteps + 1      # one extra in case it falls short
        if self.data is None:
            self.data = np.zeros(shape=
                            np.concatenate(([numsteps],self.dataShape)))
        else:
            self.data = np.append( self.data,
                        np.zeros(shape=
                            np.concatenate(([numsteps],self.dataShape))),
                        axis=0 )

    def step(self,stepi):
        if stepi % self.probeSteps == 0:
            self.data[stepi//self.probeSteps,...] = getattr(self.synObj,self.attrib) 

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

    def reset(self): pass

class SynapsesMeanArbProbe(SynapsesMeanProbe):
    '''Probe the mean of an arbitrary attribute of a Synapses object.
    Note: can also use it to probe a scalar attribute of a Synapses object
    (mean is just superfluous then),
    e.g. reward of a SynapsesNodePerturb object.'''        
    def __init__(self,synObj,model,name,attrib):
        SynapsesMeanProbe.__init__(self,synObj,model,name)
        self.attrib = attrib
        
    def step(self,stepi):
        self.data[stepi] = np.mean(getattr(self.synObj,self.attrib))
