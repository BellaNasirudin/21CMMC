import logging
import os
import sys

import numpy as np
from powerbox.dft import fft
from powerbox.tools import angular_average_nd
from py21cmmc.core import CoreLightConeModule
from py21cmmc.mcmc import run_mcmc as _run_mcmc

from fg_instrumental.core import CoreInstrumental, CorePointSourceForegrounds, CoreDiffuseForegrounds
from fg_instrumental.likelihood import LikelihoodInstrumental2D


# ============== SET THESE VARIABLES ========================

model_name = "testing"

DEBUG = int(os.environ.get("DEBUG", 0))
LOGLEVEL = 1

# for the lightcone module
z_step_factor = 1.04

# for the instrument
antenna_posfile = 'ska_low_v5' #'mwa_phase2'#
freq_min = 150.0 # in MHz
freq_max = 160.0 # in MHz
n_obs = 1 # this will be the number of observation between the min and max frequency (multi-redshift)
sky_size = 0.35  # in lm
u_min = 10
noise_integration_time = 3600000  # 1000 hours of observation time
tile_diameter = 35 
max_bl_length = 500 #250 if DEBUG else 300
Tsys = 0 #240
effective_collecting_area = 300.0

tot_daily_obs_time = 6 # in hours
beam_synthesis_time = 15 * 60 #3600 #in seconds
# foreground parameters 
S_min = 50e-6
S_max = 50e-3

# MCMC OPTIONS
params = dict(  # Parameter dict as described above.
    HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0],
    # ION_Tvir_MIN=[4.7, 2, 8, 0.1],
)

# ----- Options that differ between DEBUG levels --------
HII_DIM = [128, 75, 30][DEBUG]
DIM = 4 * HII_DIM
BOX_LEN = 4 * HII_DIM

# Instrument Options
nfreq = 100 if DEBUG else 100 * n_obs
n_cells = 256 #  900 if DEBUG else 800

# Likelihood options
n_ubins = 30
nrealisations = [500, 10, 2][DEBUG]
nthreads = [8, 4, 1][DEBUG]

walkersRatio = [8, 4, 2][DEBUG]  # The number of walkers will be walkersRatio*nparams
sampleIterations = [300, 20, 5][DEBUG]

# ============== END OF USER-SETTABLE STUFF =================

# Use -c at the end to continue sampling instead of starting again.
CONTINUE = "-c" in sys.argv

if DEBUG > 2 or DEBUG < 0:
    raise ValueError("DEBUG should be 0,1,2")

logger = logging.getLogger("21cmFAST")
logger.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][LOGLEVEL])
logger.info("Running in DEBUG=%s mode." % DEBUG)

z_min = 1420. / freq_max - 1
z_max = 1420. / freq_min - 1

def _store_lightcone(ctx):
    """A storage function for lightcone slices"""
    return 0 #ctx.get("lightcone").brightness_temp[0]

def _store_2dps(ctx):
    return 0
        
core_eor = CoreLightConeModule(
    redshift=z_min,  # Lower redshift of the lightcone
    max_redshift=z_max,  # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params=dict(
        HII_DIM=HII_DIM,
        BOX_LEN=BOX_LEN,
        DIM=DIM
    ),
    astro_params={
        "HII_EFF_FACTOR":params["HII_EFF_FACTOR"][0], 
        # "ION_Tvir_MIN":params["ION_Tvir_MIN"][0]
    },
    z_step_factor=z_step_factor,  # How large the steps between evaluated redshifts are (log).
    regenerate=False,
    keep_data_in_memory=DEBUG,
    store={
        "lc_slices": _store_lightcone,
        "2DPS": _store_2dps
    },    
    change_seed_every_iter=False,
    initial_conditions_seed=42,
    cache_mcmc=False,
    cache_dir="/data/21cmfast-data/."
)

class CustomCoreInstrument(CoreInstrumental):
    def __init__(self, freq_min=freq_min, freq_max=freq_max, nfreq=nfreq,
                 sky_size=sky_size, n_cells=n_cells, tile_diameter=tile_diameter,
                 noise_integration_time=noise_integration_time,max_bl_length = max_bl_length,
                 **kwargs):
        super().__init__(freq_max = freq_max, freq_min=freq_min, n_obs = n_obs,
                         nfreq=  nfreq, tile_diameter = tile_diameter, noise_integration_time = noise_integration_time,
                         sky_extent = sky_size, n_cells = n_cells, max_bl_length = max_bl_length,
                         tot_daily_obs_time = tot_daily_obs_time, beam_synthesis_time = beam_synthesis_time,
                         **kwargs)

class CustomLikelihood(LikelihoodInstrumental2D):
    def __init__(self, n_ubins=n_ubins, uv_max=None, nrealisations = nrealisations,
                 **kwargs):
        super().__init__(n_ubins=n_ubins, uv_max=uv_max, u_min=u_min, n_obs = n_obs, nparallel = 1,
                         simulate=False, nthreads=nthreads, nrealisations = nrealisations, ps_dim=2, include_fourierGaussianBeam=False,
                         **kwargs)

    def store(self, model, storage):
        """Store stuff"""
        storage['signal'] = model[0]['p_signal']

def run_mcmc(*args, model_name, params=params, **kwargs):
    return _run_mcmc(
        *args,
        datadir='data',  # Directory for all outputs
        model_name=model_name,  # Filename of main chain output
        params=params,
        walkersRatio=walkersRatio,  # The number of walkers will be walkersRatio*nparams
        burninIterations=0,  # Number of iterations to save as burnin. Recommended to leave as zero.
        sampleIterations=sampleIterations,  # Number of iterations to sample, per walker.
        threadCount=nthreads,  # Number of processes to use in MCMC (best as a factor of walkersRatio)
        continue_sampling=CONTINUE,  # Whether to contine sampling from previous run *up to* sampleIterations.
        **kwargs
    )

core_instr = CustomCoreInstrument(antenna_posfile = antenna_posfile, Tsys = Tsys, effective_collecting_area = effective_collecting_area, include_beam=True, beam_type="OSKAR")

# Add foregrounds core
core_fg_ps = CorePointSourceForegrounds(S_min = S_min, S_max= S_max)
core_fg_diff = CoreDiffuseForegrounds()

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz'],
    noisefile=[f'data/{model_name}.noise.npz'],
    use_analytical_noise=False
)

if __name__ == "__main__":
    chain = run_mcmc(
        [core_eor, core_fg_ps, core_fg_diff, core_instr], likelihood,
        model_name=model_name,             # Filename of main chain output
    )

