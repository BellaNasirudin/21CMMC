"""
Created on Fri Apr 20 16:54:22 2018

@author: bella
"""

import logging
import multiprocessing
from multiprocessing import pool

from functools import partial
import numpy as np
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un
from cached_property import cached_property
import powerbox.dft
from powerbox.tools import angular_average_nd
from py21cmmc.core import CoreLightConeModule
from py21cmmc.likelihood import LikelihoodBaseFile
from scipy.integrate import quad
from scipy.special import erf
from scipy import signal
from .core import CoreInstrumental, ForegroundsBase
from .util import lognormpdf
from fg_instrumental.unitConversion import mK_to_Jy_per_sr, k_perpendicular, k_parallel, obsUnits_to_cosmoUnits, kparallel_wedge

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/cluster/shared/software/libs/cuda/11.2"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import spax

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

logger = logging.getLogger("21cmFAST")

class LikelihoodInstrumental2D(LikelihoodBaseFile):
    required_cores = [CoreInstrumental]

    def __init__(self, n_uv=999, n_ubins=30, uv_max=None, u_min=None, u_max=None, frequency_taper=np.blackman, 
                 nrealisations=100, nthreads=1, model_uncertainty=0.15, eta_min=0,
                 n_obs=1, nparallel = 1, include_fourierGaussianBeam= False, ps_dim=2, ps_type= None, #"delay",
                 **kwargs):
        """
        A likelihood for EoR physical parameters, based on a Gaussian 2D power spectrum.

        In this likelihood, any foregrounds are naturally suppressed by their imposed covariance, in 2D spectral space.
        Nevertheless, it is not required that the :class:`~core.CoreForeground` class be amongst the Core modules for
        this likelihood module to work. Without the foregrounds, the 2D modes are naturally weighted by the sample
        variance of the EoR signal itself.

        The likelihood requires the :class:`~core.CoreInstrumental` Core module.

        Parameters
        ----------
        n_uv : int, optional
            The number of UV cells to grid the visibilities (per side). By default, uses the same number of UV cells
            as the Core (i.e. the same grid used to interpolate the simulation onto the baselines).

        n_ubins : int, optional
            The number of kperp (or u) bins to use when doing a cylindrical average of the power spectrum.

        uv_max : float, optional
            The extent of the UV grid. By default, uses the longest baseline at the highest frequency.

        u_min, u_max : float, optional
            The minimum and maximum of the grid of |u| = sqrt(u^2 + v^2). These define the *bin edges*. By default,
            they will be set as the min/max of the UV grid (along a side of the square).

        frequency_taper : callable, optional
            A function which computes a taper function on an nfreq-array. Callable should
            take single argument, N.

        nrealisations : int, optional
            The number of realisations to use if calculating a *foreground* mean/covariance. Only applicable if
            a ForegroundBase instance is loaded as a core.

        nthreads : int, optional
            Number of processes to use if generating realisations for numerical covariance.

        model_uncertainty : float, optional
            Fractional uncertainty in the signal model power spectrum (this is modelling uncertainty of the code itself)

        eta_min : float, optional
            Minimum eta value to consider in the model. This will be applied at every u value.

        n_obs : int, optional
            Whether to combine observation from different frequency bands.

        nparallel : int, optional
            Specify the number of threads to do parallelization when re-gridding the visibilities.

        include_fourierGaussianBeam : bool, optional
            Whether to regrid the visibilities according to the Fourier transform of the Gaussian beam.

        ps_dim : int, optional
            The dimension of the power spectrum. 1 for 1D, and 2 for 2D.

        Other Parameters
        ----------------
        datafile : str
            A filename referring to a file which contains the observed data (or mock data) to be fit to. The file
            should be a compressed numpy binary (i.e. a npz file), and must contain at least the arrays "kpar", "kperp"
            and "p", which are the parallel/perpendicular modes (in 1/Mpc) and power spectrum (in Mpc^3) respectively.
        """

        super().__init__(**kwargs)

        self._n_uv = n_uv
        self.n_ubins = n_ubins
        self._uv_max = uv_max
        self.frequency_taper = frequency_taper
        self.nrealisations = nrealisations
        self.model_uncertainty = model_uncertainty
        self.eta_min = eta_min
        self._u_min, self._u_max = u_min, u_max
        self._nthreads = nthreads
        self.ps_dim = ps_dim
        self.n_obs = n_obs
        self.nparallel = nparallel

        self.ps_type = ps_type

        self.include_fourierGaussianBeam = include_fourierGaussianBeam

    def setup(self):
        super().setup()

        # we can unpack data now because we know it's always a list of length 1.
        if self.data:
            self.data = self.data[0]
        if self.noise:
            self.noise = self.noise[0]

        #Store only the p_signal
        self.data = {"p_signal":self.data["p_signal"]}

    @cached_property
    def n_uv(self):
        """The number of cells on a side of the (square) UV grid"""
        if self._n_uv is None:
            return self._instr_core.n_cells
        else:
            return self._n_uv

    def reduce_data(self, ctx):
        """
        Simulate datasets to which this very class instance should be compared.
        """
        self.baselines_type = ctx.get("baselines_type")
        self.sample_gain = ctx.get("sample_gain")
        visibilities = ctx.get("visibilities") 
        print(np.shape(visibilities), np.min(visibilities), np.max(visibilities))
        # raise SystemExit
        p_signal = self.compute_power(visibilities)
        print(np.shape(p_signal), np.min(p_signal), np.max(p_signal))
        # if self.simulate == True:
        np.savez("data/skadc_stuff/"+self.datafile[0][:-4], p_signal=p_signal, baselines=self.baselines, frequencies=self.frequencies,
                     u=self.u, eta=self.eta)
        raise SystemExit

        # Remember that the results of "simulate" can be used in two places: (i) the computeLikelihood method, and (ii)
        # as data saved to file. In case of the latter, it is useful to save extra variables to the dictionary to be
        # looked at for diagnosis, even though they are not required in computeLikelihood().
        if self.ps_dim == 2:
            return [dict(p_signal=p_signal, baselines=self.baselines, frequencies=self.frequencies,
                     u=self.u, eta=self.eta)]
        elif self.ps_dim == 1:
            return [dict(p_signal=p_signal, baselines=self.baselines, frequencies=self.frequencies,
                     k = self.k)]

    def define_noise(self, ctx, model):
        """
        Define the properties of the noise... its mean and covariance.

        Note that in general this method should just calculate whatever noise properties are relevant, but in
        this case that is specifically the mean (of the noise) and its covariance.

        Note also that the outputs of this function are by default *saved* to a file within setup, so it can be
        checked later.

        It is *only* run on setup, not every iteration. So noise properties that are parameter-dependent must
        be performed elsewhere.

        Parameters
        ----------
        ctx : dict-like
            The Context object which is assumed to hold the core simulation context.

        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        list of dicts
            In this case, a list with a single dict, which has the mean and covariance in it.

        """
        # Only save the mean/cov if we have foregrounds, and they don't update every iteration (otherwise, get them
        # every iter).
        if self.foreground_cores and not any([fg._updating for fg in self.foreground_cores]):
            if self.nrealisations!=0:
                
                mean, covariance = self.numerical_covariance(ctx,nrealisations=self.nrealisations, nthreads=self._nthreads)

                # thermal_covariance = self.get_thermal_covariance()
                # covariance = [x + y for x, y in zip(covariance, thermal_covariance)]
            elif self.nrealisations==0:
                mean =0
                covariance =0

        else:
            # Only need thermal variance if we don't have foregrounds, otherwise it will be embedded in the
            # above foreground covariance... BUT NOT IF THE FOREGROUND COVARIANCE IS ANALYTIC!!
            #                covariance = self.get_thermal_covariance()
            #                mean = np.repeat(self.noise_power_expectation, len(self.eta)).reshape((len(self.u), len(self.eta)))
            mean = 0
            covariance = 0

        return [{"mean": mean, "covariance": covariance}]

    def computeLikelihood(self, model):
        "Compute the likelihood"
        # remember that model is *exactly* the result of reduce_data(), which is a  *list* of dicts, so unpack
        model = model[0]
        
        lnl = 0
        pos_half = int(len(self.frequencies)/2 + 4)
        for ii in range(self.n_obs):
            total_model = model['p_signal'][ii] # this already have mean noise and foregrounds if we use those modules
            if self.ps_dim == 2:
                sig_cov = self.get_cosmic_variance(model['p_signal'][ii])
                
                # get the covariance
                if self.foreground_cores:
                    # Normal case (foreground parameters are not being updated, or there are no foregrounds)
                    total_cov = np.array([x + y for x, y in zip(self.noise['covariance'], sig_cov)])
                else:
                    logger.info("The covariance is purely based on the cosmic variance")
                    total_cov = sig_cov

                if (ii !=2): ##need to find better way to do this!!
                    lnl +=  -0.5 * np.sum( (self.data['p_signal'][ii][pos_half+1:, :][3:5, 1:4] - total_model[pos_half+1:, :][3:5, 1:4])**2 
                        / total_cov[pos_half+1:, :][3:5, 1:4])
                #lognormpdf(self.data['p_signal'][ii][pos_half:, :11], total_model[pos_half:, :11], total_cov[pos_half:, :11])
            else:
                lnl += -0.5 * np.sum(
                    (self.data['p_signal'][ii] - total_model) ** 2 / (self.model_uncertainty * model['p_signal'][ii]) ** 2)

        print(lnl)
        
        return lnl

    @cached_property
    def _lightcone_core(self):
        for m in self._cores:
            if isinstance(m, CoreLightConeModule):
                return m

        raise AttributeError("No lightcone core loaded")

    @property
    def _instr_core(self):
        for m in self._cores:
            if isinstance(m, CoreInstrumental):
                return m

    @property
    def foreground_cores(self):
        return [m for m in self._cores if isinstance(m, ForegroundsBase)]

    @cached_property
    def frequencies(self):
        return self._instr_core.instrumental_frequencies

    @cached_property
    def baselines(self):
        return self._instr_core.baselines

    def get_thermal_covariance(self):
        """
        Form the thermal variance per u into a full covariance matrix, in the same format as the other covariances.

        Returns
        -------
        cov : list
            A list of arrays defining a block-diagonal covariance matrix, of which the thermal variance is really
            just the diagonal.
        """
        cov = []
        for var in self.noise_power_variance:
            cov.append(np.diag(var * np.ones(len(self.eta))))

        return cov

    def get_cosmic_variance(self, signal_power):
        """
        From a 2D signal (i.e. EoR) power spectrum, make a list of covariances in eta, with length u.

        Parameters
        ----------
        signal_power : (n_eta, n_u)-array
            The 2D power spectrum of the signal.

        Returns
        -------
        cov : list of arrays
            A length-u list of arrays of shape n_eta * n_eta.
        """
        if self.ps_dim == 2:
            cov = []
            grid_weights = self.grid_weights

            for ii, sig_eta in enumerate(signal_power):
                x = (1 / grid_weights[ii] * np.diag(sig_eta)**2)
                x[np.isnan(x)] = 0
                cov.append(x)
            x = 1 / grid_weights * signal_power**2
            x[np.isnan(x)] = 1
            return  x #cov
        else:
            x = 1 / self.grid_weights * signal_power**2
            x[np.isnan(x)] = 0
            return x

    def numerical_covariance(self, ctx, params={}, nrealisations=200, nthreads=1):
        """
        Calculate the covariance of the foregrounds.
    
        Parameters
        ----------
        params: dict
            The parameters of this iteration. If empty, default parameters are used.

        nrealisations: int, optional
            Number of realisations to find the covariance.
        
        Output
        ------
        mean: (nperp, npar)-array
            The mean 2D power spectrum of the foregrounds.
            
        cov: 
            The sparse block diagonal matrix of the covariance if nrealisation is not 1
            Else it is 0
        """

        if nrealisations < 2:
            raise ValueError("nrealisations must be more than one")

        # We use a hack where we define an external function which *passed*
        # this object just so that we can do multiprocessing on it.
        fnc = partial(_produce_mock, self, params)

        pool = MyPool(nthreads)
        
        power, visgrid = zip(*pool.map(fnc, np.arange(nrealisations)))
        
        # Note, this covariance *already* has thermal noise built in.
        cov = []
        mean = []

        for ii in range(self.n_obs):
            
            # if self.ps_dim == 2:
            #     mean.append(np.mean(np.array(power)[:,ii,:,:], axis=0))
            #     cov.append([np.cov(x) for x in np.array(power)[:,ii,:,:].transpose((1, 2, 0))])
            # else:
            mean.append(np.mean(np.array(power)[:,ii,:], axis=0))
            cov = np.var(np.array(power)[:,ii,:] , axis=0)
        
        # Cleanup the memory
        #for i in range(len(power)-1,-1,-1):
        #    del power[i]
                   
        pool.close()
        pool.join()

        return mean, cov

    def compute_power(self, visibilities):
        """
        Compute the 2D power spectrum within the current context.

        Parameters
        ----------
        visibilities : (nbl, nf)-complex-array
            The visibilities of each baseline at each frequency

        Returns
        -------
        power2d : (nperp, npar)-array
            The 2D power spectrum.

        coords : list of 2 arrays
            The first is kperp, and the second is kpar.
        """
        # Grid visibilities only if we're not using "grid_centres"

        if (self.baselines_type != "grid_centres") & (self.ps_type!="delay"):
            if(self.nparallel==1) or (self.include_fourierGaussianBeam==False):
                visgrid, kernel_weights = self.grid_visibilities(visibilities)
            else:
                visgrid, kernel_weights = self.grid_visibilities_parallel(visibilities)
        else:
            visgrid = visibilities
            kernel_weights = np.load("mask_maxbl10000_ncells11520.npz")["mask"]
            self.uvgrid = self.baselines
        print(np.min(visgrid), np.max(visgrid))
        # Transform frequency axis
        visgrid = self.frequency_fft(visgrid, self.frequencies, self.ps_dim, n_obs = self.n_obs, taper=signal.blackmanharris)#,self.frequency_taper)
        # np.savez("/scratch/anasirudin/visgrid_10s", visgrid=visgrid, centres=centres)#, kernel_weights=kernel_weights)
        # Get 2D power from gridded vis.
        if self.ps_type=="delay":
            power2d = self.get_powerDelay(visgrid, ps_dim=self.ps_dim)
        else:
            power2d = self.get_power(visgrid, kernel_weights, ps_dim=self.ps_dim)

            # only re-write the regridding kernel weights if we want to simulate things again 
            if((os.path.exists(self.datafile[0][:-4]+".kernel_weights.npy")==False)):
                np.save(self.datafile[0][:-4]+".kernel_weights.npy",kernel_weights)

        if self.sample_gain == True:
            logger.info("Sampling the effects of gain on power spectrum using PCA")

            pca = spax.PCA_m(
            N = 5,
            devices = jax.devices("gpu"),
            )
            filename = "../../SPax/data/allmodes_fitPSGainPCA_Ncomponents5_Nruns10000.hdf5"

            pca.load(filename)
            sampled_data = pca.sample(1).T.reshape((1, 128, 30))
            power2d+= sampled_data

        return power2d

    def get_powerDelay(self, all_visibilities, ps_dim=2):
        """
        Determine the 2D Delay Spectrum of the observation.

        Parameters
        ----------

        vis : complex (nbaseline, neta)-array
            The ungridded visibilities, fourier-transformed along the frequency axis. Units JyHz.

        coords: list of 3 1D arrays.
            The [u,v,eta] co-ordinates corresponding to the gridded fourier visibilities. u and v in 1/rad, and
            eta in 1/Hz.

        Returns
        -------
        PS : float (n_obs, n_eta, bins)-list
            The cylindrical averaged (or 2D) Power Spectrum, in unit Jy^2 Hz^2.
        """
        logger.info("Calculating the DELAY spectrum")
        PS = []

        len_freqbands = np.shape(all_visibilities)[-1]
        for ii, vis in enumerate(all_visibilities):
            
            # The 3D power spectrum
            power_3d = np.absolute(vis) ** 2

            band_frequencies = self.frequencies[ii*len_freqbands: (ii+1)*len_freqbands]

            u = np.outer(self.baselines[:,0], (band_frequencies / const.c).value)
            v = np.outer(self.baselines[:,1], (band_frequencies / const.c).value)

            r = np.sqrt(u**2 + v**2)
            
            if ps_dim == 2:

                power = np.zeros((len(band_frequencies), len(self.u_edges)-1))
                weights = np.zeros((len(band_frequencies), len(self.u_edges)-1))

                for ff in range(len(band_frequencies)):

                    weights[ff, :] = np.histogram(r[:,ff], self.u_edges)[0]
                    
                    power[ff, :] = np.histogram(r[:,ff], self.u_edges, weights = power_3d[:, ff])[0]


                power[weights>0] /= (weights[weights>0])
    
            elif ps_dim == 1:
                # need to convert uv and eta to same cosmo unit

                # need to change everything to cosmo unit
                sq_cosmo, kperp, kpar = obsUnits_to_cosmoUnits(power_3d, self.sky_size, band_frequencies, r, self.eta)

                k = np.sqrt(kperp**2 + kpar**2)

                # find the wedge
                # remember that sky size is the diameter in lm so need to find radius and multiply by pi to get to radian 
                fg_wedge = kparallel_wedge(kperp, self.sky_size / 2 * np.pi, np.min(frequencies_to_redshifts(band_frequencies)))

                # find foregrounds based on the buffer
                fg_edge = kpar[int(len(band_frequencies)/2) + fg_buffer]

                # and only include this area
                include_kparBuffer = np.where(np.abs(kpar)>=fg_edge)[0] #& (np.abs(kpar)>=fg_wedge)

                sq_cosmo = sq_cosmo[:, include_kparBuffer]
                k = k[:, include_kparBuffer]
                kpar = kpar[include_kparBuffer]
                fg_wedge = fg_wedge[:, include_kparBuffer]

                # make a 2-D shape containing the values of kpar for each kperp
                kpar_copy = np.outer(np.ones(np.shape(fg_wedge)[0]), kpar)

                # and only include all regions larger than the wedge
                include_kparWedge = np.abs(kpar_copy)>fg_wedge #& (np.abs(kpar)>=fg_wedge)

                k = k[include_kparWedge]
                sq_cosmo = sq_cosmo[include_kparWedge]

                # finally bin them
                weights = np.histogram(k, self.u_edges)[0]     
                power = np.histogram(k, self.u_edges, weights = sq_cosmo)[0]

                power[weights>0] /= (weights[weights>0])
            
            power[np.isnan(power)] = 0
            PS.append(power)
     
        return PS

    def get_power(self, gridded_vis, kernel_weights, ps_dim=2):
        """
        Determine the 2D Power Spectrum of the observation.

        Parameters
        ----------

        gridded_vis : complex (ngrid, ngrid, neta)-array
            The gridded visibilities, fourier-transformed along the frequency axis. Units JyHz.

        coords: list of 3 1D arrays.
            The [u,v,eta] co-ordinates corresponding to the gridded fourier visibilities. u and v in 1/rad, and
            eta in 1/Hz.

        Returns
        -------
        PS : float (n_obs, n_eta, bins)-list
            The cylindrical averaged (or 2D) Power Spectrum, in unit Jy^2 Hz^2.
        """
        logger.info("Calculating the power spectrum")
        PS = []

        u, v = np.meshgrid(self.uvgrid, self.uvgrid)
        r = np.sqrt(u**2 + v**2)
        
        for ii, vis in enumerate(gridded_vis):
            # The 3D power spectrum
            power_3d = np.absolute(vis) ** 2
            num_freq = np.shape(power_3d)[2]
            band_frequencies = self.frequencies[int(ii*num_freq): int((ii+1)*num_freq)]
            
            if ps_dim == 2:
                # P = angular_average_nd(
                #     field=power_3d[:,:,int(len(power_3d)/2):],  # return the positive part,
                #     coords=[self.uvgrid, self.uvgrid, self.eta[ii*len(power_3d): (ii+1)*len(power_3d)]],
                #     bins=self.u_edges, n=ps_dim,
                #     weights=np.sum(kernel_weights**2, axis=2),  # weights,
                #     bin_ave=False,
                # )
                P = np.zeros((num_freq, len(self.u_edges)-1))
                weights = np.zeros((num_freq, len(self.u_edges)-1))

                for ff in range(num_freq):

                    weights[ff, :] = np.histogram(r, self.u_edges)[0]
                    
                    P[ff, :] = np.histogram(r, self.u_edges, weights = power_3d[:, :, ff])[0]


                P[weights>0] /= (weights[weights>0])
                print(np.min(P), np.max(P))
    
            elif ps_dim == 1:
                # need to convert uv and eta to same cosmo unit
                zmid = 1420e6/ np.mean(self.frequencies) -1
                kperp = k_perpendicular(self.uvgrid, zmid).value
                kpar = k_parallel(self.eta, zmid).value
                    
                P = angular_average_nd(
                    field=power_3d,
                    coords=[kperp, kperp, kpar],
                    bins=self.u_edges,
                    weights=kernel_weights**2,
                    bin_ave=False,
                )
                
                if self.k is None:
                    self.k = P[1]
                
                P = P[0]
            
            P[np.isnan(P)] = 0
            PS.append(P)
        
        return PS

    @staticmethod
    def fourierBeam(centres, u_bl, v_bl, frequency, a, min_attenuation = 1e-3, N = 20):
        """
        Find the Fourier Transform of the Gaussian beam
        
        Parameter
        ---------
        centres : (ngrid)-array
            The centres of the grid.
        
        u_bl : (n_baselines)-array
            The list of baselines in m.
            
        v_bl : (n_baselines)-array
            The list of baselines in m.
            
        frequency: float
            The frequency in Hz.
        """
        
        indx_u = np.digitize(u_bl, centres)
        indx_v = np.digitize(v_bl, centres)

        C = np.sqrt(np.pi/a)
        P2a = (np.pi**2)/a

        indx_u+= -int(N/2)
        indx_v+= -int(N/2)

        beam = np.zeros([N,N])
        
        x, y = np.meshgrid(centres[indx_u:indx_u+N], centres[indx_v:indx_v+N],copy=False)
        B = (C * np.exp(-  P2a*((x - u_bl)**2 + (y - v_bl)**2 ))).T
        B[B<min_attenuation] = 0
        
        beam[:B.shape[0],:B.shape[1]] =B 
        if indx_u<0:
            indx_u = 0
        if indx_v<0:
            indx_v = 0         
                   
        return beam, indx_u, indx_v

    def grid_visibilities(self, visibilities, N = 30):
        """
        Grid a set of visibilities from baselines onto a UV grid.

        Uses Fourier (Gaussian) beam weighting to perform the gridding.

        Parameters
        ----------
        visibilities : complex (n_baselines, n_freq)-array
            The visibilities at each basline and frequency.

        Returns
        -------
        visgrid : (ngrid, ngrid, n_freq)-array
            The visibility grid, in Jy Hz.
        """
        logger.info("Gridding the visibilities")

        visgrid = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)), dtype=np.complex128)

        if(os.path.exists(self.datafile[0][:-4]+".kernel_weights.npy")):
            kernel_weights = np.load(self.datafile[0][:-4]+".kernel_weights.npy")
            
            if(np.any(visgrid.shape!=kernel_weights.shape)):
                kernel_weights=None
        else:
            kernel_weights=None

        if kernel_weights is None:
            weights = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)))
            
        if self.include_fourierGaussianBeam is True:
            
            for jj, freq in enumerate(self.frequencies):

                u_bl = (self.baselines[:,0] * freq / const.c).value
                v_bl = (self.baselines[:,1] * freq / const.c).value
    
                for kk in range(len(indx_u)):
                    beam, indx_u, indx_v = self.fourierBeam(self.uvgrid, u_bl[kk], v_bl[kk], freq, 1/ (2 * self._instr_core.sigma(freq)**2), min_attenuation = 1e-3, N=N)
        
                    beamsum = np.sum(beam)#,axis=(1,2))                    
                    if beamsum!=0:
                        (beamushape,beamvshape) = np.shape(beam)
    
                        #Check if the beam has gone over the edge of visgrid in the u-plane
                        val =  indx_u+beamushape - self.n_uv
                        if(val>0):
                            ibeamindx_u = beamushape - val
                        else:
                            ibeamindx_u = beamushape
    
                        #Check if the beam has gone over the edge of visgrid in the v-plane
                        val = indx_v+beamvshape - self.n_uv
                        if(val>0):
                            ibeamindx_v = beamvshape - val
                        else:
                            ibeamindx_v = beamvshape
    
                        visgrid[indx_u[kk]:indx_u[kk]+beamushape, indx_v[kk]:indx_v[kk]+beamvshape, jj] += beam[:ibeamindx_u,:ibeamindx_v] / beamsum * visibilities[kk,jj]
    
                        if kernel_weights is None:
                            weights[indx_u[kk]:indx_u[kk]+beamushape, indx_v[kk]:indx_v[kk]+beamvshape, jj] += beam[:ibeamindx_u,:ibeamindx_v] / beamsum
        else:

            u_bl = np.outer(self.baselines[:,0] , self.frequencies) / const.c.value
            v_bl = np.outer(self.baselines[:,1] , self.frequencies) / const.c.value
            
            for jj, freq in enumerate(self.frequencies):

                visgrid[:, :, jj] += np.histogram2d(u_bl[:, jj], v_bl[:, jj], bins = ugrid, weights=np.real(visibilities[:, jj]))[0]
                visgrid[:, :, jj] += np.histogram2d(u_bl[:, jj], v_bl[:, jj], bins = ugrid, weights=np.imag(visibilities[:, jj]))[0] * 1j
                weights[:, :, jj] = np.histogram2d(u_bl[:, jj], v_bl[:, jj], bins = ugrid)[0]

                # for kk in range(u_bl.shape[0]):
                #     indx_u = np.digitize(u_bl[kk,jj], centres)
                #     indx_v = np.digitize(v_bl[kk,jj], centres)

                #     indx_u1 = np.digitize(-1*u_bl[kk,jj], centres)
                #     indx_v1 = np.digitize(-1*v_bl[kk,jj], centres)

                #     visgrid[indx_u[kk,jj]-1, indx_v[kk,jj]-1,jj] += visibilities[kk,jj]
                #     visgrid[indx_u1[kk,jj]-1, indx_v1[kk,jj]-1,jj] += visibilities[kk,jj]
                    
                #     if kernel_weights is None:
                #         weights[indx_u[kk,jj]-1, indx_v[kk,jj]-1,jj] += 1
                #         weights[indx_u[kk,jj]-1, indx_v[kk,jj]-1,jj] += 1
                    
        if kernel_weights is None:
            kernel_weights = weights

        visgrid[kernel_weights!=0] /= kernel_weights[kernel_weights!=0]
        
        return visgrid, kernel_weights

    @staticmethod
    def _grid_visibilities_buff(n_uv,visgrid_buff_real,visgrid_buff_imag,weights_buff, visibilities,frequencies,a,baselines,centres,sigfreq, min_attenuation = 1e-3, N = 30):

        logger.info("Gridding the visibilities")

        nfreq = len(frequencies)

        vis_real = np.frombuffer(visgrid_buff_real).reshape(n_uv,n_uv,nfreq)
        vis_imag = np.frombuffer(visgrid_buff_imag).reshape(n_uv,n_uv,nfreq)
        vis_real[:] = 0
        vis_imag[:] = 0

        if(weights_buff is not None):
            weights = np.frombuffer(weights_buff).reshape(n_uv,n_uv,nfreq)
            weights[:] = 0

        for ii in range(nfreq):

            freq = frequencies[ii]

            u_bl = (baselines[:,0] * freq / const.c).value
            v_bl = (baselines[:,1] * freq / const.c).value

            for kk in range(len(u_bl)):

                beam, indx_u, indx_v = LikelihoodInstrumental2D.fourierBeam(centres, u_bl[kk], v_bl[kk], freq, a[ii], N=N)

                beamsum = np.sum(beam)                
                if beamsum!=0:
                    (beamushape,beamvshape) = np.shape(beam)

                    #Check if the beam has gone over the edge of visgrid in the u-plane
                    val =  indx_u+beamushape - n_uv
                    if(val>0):
                        ibeamindx_u = beamushape - val
                    else:
                        ibeamindx_u = beamushape

                    #Check if the beam has gone over the edge of visgrid in the v-plane
                    val = indx_v+beamvshape - n_uv
                    if(val>0):
                        ibeamindx_v = beamvshape - val
                    else:
                        ibeamindx_v = beamvshape

                    vis_real[indx_u:indx_u+beamushape, indx_v:indx_v+beamvshape, ii] += beam[:ibeamindx_u,:ibeamindx_v] / beamsum * visibilities[kk,ii].real
                    vis_imag[indx_u:indx_u+beamushape, indx_v:indx_v+beamvshape, ii] += beam[:ibeamindx_u,:ibeamindx_v] / beamsum * visibilities[kk,ii].imag

                    if weights_buff is not None:
                        weights[indx_u:indx_u+beamushape, indx_v:indx_v+beamvshape, ii] += beam[:ibeamindx_u,:ibeamindx_v] / beamsum

    def grid_visibilities_parallel(self, visibilities,min_attenuation = 1e-3, N = 30):
        """
        Grid a set of visibilities from baselines onto a UV grid.

        Uses Fourier (Gaussian) beam weighting to perform the gridding.
        
        Parameters
        ----------
        visibilities : complex (n_baselines, n_freq)-array
            The visibilities at each basline and frequency.

        Returns
        -------
        visgrid : (ngrid, ngrid, n_freq)-array
            The visibility grid, in Jy.
        """

        #Find out the number of frequencies to process per thread
        nfreq = len(self.frequencies)
        numperthread = nfreq/self.nparallel
        nfreqstart = np.zeros(self.nparallel,dtype=int)
        nfreqend = np.zeros(self.nparallel,dtype=int)
        for i in range(self.nparallel):
            nfreqstart[i] = round(i * numperthread)
            nfreqend[i] = round((i + 1) * numperthread)

         # Set the last process to the number of frequencies
        nfreqend[-1] = nfreq

        processes = []
        
        visgrid = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)), dtype=np.complex128)


        if(os.path.exists(self.datafile[0][:-4]+".kernel_weights.npy")):
            kernel_weights = np.load(self.datafile[0][:-4]+".kernel_weights.npy")
            
            if(np.any(visgrid.shape!=kernel_weights.shape)):
                kernel_weights=None
        else:
            kernel_weights=None

        if kernel_weights is None:
            weights = np.zeros((self.n_uv, self.n_uv, len(self.frequencies)))

        visgrid_buff_real = []
        visgrid_buff_imag = []
        weights_buff = []

        #Lets split this array up into chunks
        for i in range(self.nparallel):

            visgrid_buff_real.append(multiprocessing.RawArray(np.sctype2char(visgrid.real),visgrid[:,:,nfreqstart[i]:nfreqend[i]].size))
            visgrid_buff_imag.append(multiprocessing.RawArray(np.sctype2char(visgrid.imag),visgrid[:,:,nfreqstart[i]:nfreqend[i]].size))

            if(kernel_weights is None):
                weights_buff.append(multiprocessing.RawArray(np.sctype2char(weights),weights[:,:,nfreqstart[i]:nfreqend[i]].size))
            else:
                weights_buff.append(None)

            processes.append(multiprocessing.Process(target=self._grid_visibilities_buff,args=(self.n_uv,visgrid_buff_real[i],visgrid_buff_imag[i],
                weights_buff[i], visibilities[:,nfreqstart[i]:nfreqend[i]],self.frequencies[nfreqstart[i]:nfreqend[i]],
                1/ (2 * self._instr_core.sigma(self.frequencies[nfreqstart[i]:nfreqend[i]])**2),self.baselines,self.uvgrid,
                self._instr_core.sigma(self.frequencies[nfreqstart[i]:nfreqend[i]]),min_attenuation, N) ))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        for i in range(self.nparallel):

            visgrid[:,:,nfreqstart[i]:nfreqend[i]].real = np.frombuffer(visgrid_buff_real[i]).reshape(self.n_uv,self.n_uv,nfreqend[i]-nfreqstart[i])
            visgrid[:,:,nfreqstart[i]:nfreqend[i]].imag = np.frombuffer(visgrid_buff_imag[i]).reshape(self.n_uv,self.n_uv,nfreqend[i]-nfreqstart[i])

            if(kernel_weights is None):
                weights[:,:,nfreqstart[i]:nfreqend[i]] = np.frombuffer(weights_buff[i]).reshape(self.n_uv,self.n_uv,nfreqend[i]-nfreqstart[i])

        if kernel_weights is None:
            kernel_weights = weights
        
        visgrid[kernel_weights!=0] /= kernel_weights[kernel_weights!=0]

        return visgrid, kernel_weights

    @cached_property
    def uvgrid(self):
        """
        Centres of the uv grid along a side.
        """
        if self.baselines_type != "grid_centres":
            ugrid = np.linspace(-self.uv_max, self.uv_max, self.n_uv + 1)  # +1 because these are bin edges.
            return (ugrid[1:] + ugrid[:-1]) / 2
        else:
            # return the uv
            return self.baselines

    @cached_property
    def uv_max(self):
        if self._uv_max is None:
            if self.baselines_type != "grid_centres":
                return 1e4 # 5003.296588433671 #(np.max(np.abs(self.baselines)) * 150e6 / const.c).value
            else:
                # return the uv
                return self.baselines.max()
        else:
            return self._uv_max

    @cached_property
    def u_min(self):
        """Minimum of |u| grid"""
        if self._u_min is None:
            return np.abs(self.uvgrid).min()
        else:
            return self._u_min

    @cached_property
    def u_max(self):
        """Maximum of |u| grid"""
        if self._u_max is None:
            return self.uv_max
        else:
            return self._u_max

    @cached_property
    def u_edges(self):
        """Edges of |u| bins where |u| = sqrt(u**2+v**2)"""
        if self.ps_dim == 2:
            return np.linspace(0.02, 1500, self.n_ubins + 1) #np.linspace(self.u_min, self.u_max, self.n_ubins + 1)
        elif self.ps_dim == 1:
            return np.linspace(0.01, 1.5, self.n_ubins + 1)

    @cached_property
    def u(self):
        """Centres of |u| bins"""
        return (self.u_edges[1:] + self.u_edges[:-1]) / 2
    
    @cached_property
    def eta(self):
        "Grid of positive frequency fourier-modes"
        dnu = (self.frequencies[1] - self.frequencies[0])
        eta = powerbox.dft.fftfreq(int(len(self.frequencies) / self.n_obs), d=dnu, b=2 * np.pi)
        if self.ps_dim==2:
            return eta[eta > self.eta_min]
        elif self.ps_dim==1:
            return eta

    @cached_property
    def grid_weights(self):
        """The number of uv cells that go into a single u annulus (related to baseline weights)"""
        if self.ps_type=="delay":
            u = np.outer(self.baselines[:,0], (self.frequencies / const.c).value)
            v = np.outer(self.baselines[:,1], (self.frequencies / const.c).value)

            r = np.sqrt(u**2 + v**2)
            weights = np.zeros((len(self.frequencies), len(self.u_edges)-1))

            for ff in range(len(self.frequencies)):
                weights[ff] = np.histogram(r[:,ff], self.u_edges)[0]   
            return weights
        else:
            # if(os.path.exists(self.datafile[0][:-4]+".kernel_weights.npy")):
            #     field = np.load(self.datafile[0][:-4]+".kernel_weights.npy")   
            # else:
            #     field = np.ones((len(self.uvgrid),len(self.uvgrid),len(self.frequencies)))
                    
            if self.ps_dim == 2:

                # return angular_average_nd(
                #     field = field**2,
                #     coords=[self.uvgrid, self.uvgrid, self.eta],
                #     bins=self.u_edges, n=self.ps_dim, bin_ave=False,
                #     average=False)[0][:,int(len(self.frequencies)/2):]

                u, v = np.meshgrid(self.uvgrid, self.uvgrid)
                r = np.sqrt(u**2 + v**2)
                
                all_weights = []
                for ii in range(self.n_obs):
                    
                    num_freq = len(self.frequencies[int(ii*num_freq): int((ii+1)*num_freq)])
                    
                    weights = np.zeros((num_freq, len(self.u_edges)-1))

                    for ff in range(num_freq):

                        weights[ff, :] = np.histogram(r, self.u_edges)[0]

                    all_weights.append(weights)

                return all_weights

            elif self.ps_dim == 1:
                zmid = 1420e6/ np.mean(self.frequencies) -1

                return angular_average_nd(
                    field= field**2,
                    coords=[k_perpendicular(self.uvgrid, zmid).value,k_perpendicular(self.uvgrid, zmid).value, k_parallel(self.eta, zmid).value],
                    bins=self.u_edges, bin_ave=False,
                    average=False)[0]

    @staticmethod
    def frequency_fft(vis, frequencies, dim, taper=np.ones_like, n_obs =1):
        """
        Fourier-transform a gridded visibility along the frequency axis.

        Parameters
        ----------
        vis : complex (ncells, ncells, nfreq)-array
            The gridded visibilities.

        frequencies : (nfreq)-array
            The linearly-spaced frequencies of the observation.

        taper : callable, optional
            A function which computes a taper function on an nfreq-array. Default is to have no taper. Callable should
            take single argument, N.
        
        n_obs : int, optional
            Number of observations used to separate the visibilities into different bandwidths.

        Returns
        -------
        ft : (ncells, ncells, nfreq/2)-array
            The fourier-transformed signal, with negative eta removed.

        eta : (nfreq/2)-array
            The eta-coordinates, without negative values.
        """
        all_fts = []

        W = (frequencies.max() - frequencies.min()) / n_obs
        L = int(len(frequencies) / n_obs)
        
        for ii in range(n_obs):
            if len(np.shape(vis))==2:
                ft = np.fft.fftshift(np.fft.fft(vis[:,ii*L:(ii+1)*L] * taper(L), axis=-1), axes=-1) * np.diff(frequencies)[0]
            else:
                ft = powerbox.dft.fft(vis[:,:,ii*L:(ii+1)*L] * taper(L), W, axes=(2,), a=0, b=2 * np.pi)[0]
        
            all_fts.append(ft)
            
        ft = np.array(all_fts)

        return ft

def _produce_mock(self, params, i):
    """Produces a mock power spectrum for purposes of getting numerical_covariances"""
    # Create an empty context with the given parameters.
    np.random.seed(i)

    ctx = self.chain.createChainContext(params)

    # For each realisation, run every foreground core (not the signal!)
    for core in self.foreground_cores:
        core.simulate_mock(ctx)

    # And turn them into visibilities
    self._instr_core.simulate_mock(ctx)

    # And compute the power
    power = self.compute_power(ctx.get("visibilities"))

    return power, ctx.get("visibilities")
