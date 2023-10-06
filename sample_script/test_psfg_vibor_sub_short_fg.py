import logging
import os
import sys
import pymultinest
import numpy as np
from py21cmmc.core import CoreLightConeModule
from py21cmmc.mcmc import run_mcmc as _run_mcmc
from py21cmmc import analyse
from astropy.cosmology import Planck18
from fg_instrumental.core import CoreInstrumental, CorePointSourceForegrounds, CoreDiffuseForegrounds
from fg_instrumental.likelihood import LikelihoodInstrumental2D

nthreads = int(sys.argv[1]) #51 #[24, 24, 24][DEBUG]
diffuse_realization = int(sys.argv[2])
eor_model = int(sys.argv[3])
fband = int(sys.argv[4])
fband_name = ["106-121", "121-136", "136-151", "151-166", "166-181", "181-196"][fband]
name_extension = "SKADC" #str(sys.argv[1])
# ============== SET THESE VARIABLES ========================

model_name = "testing_stuff" #%(fband_name, eor_model)

DEBUG = int(os.environ.get("DEBUG", 0))
LOGLEVEL = 2

input_data_dir = "/data/dev/tmp/data/"

# for the instrument
# Instrument Options
antenna_posfile = "new_ska"#'ska_low_v5' #'mwa_phase2'#
freq_min = [106.0, 121.0, 136.0, 151.0, 166.0, 181.0][fband] # in MHz
freq_max = [121.0, 136.0, 151.0, 166.0, 181.0, 196.0][fband] # in MHz
n_obs = 1 # this will be the number of observation between the min and max frequency (multi-redshift)
nfreq = 150 #308 #128 #if DEBUG else 100 * n_obs
n_cells = 512 # #* 6#  900 if DEBUG else 800
sky_size = np.deg2rad(8) #0.14  # in radian
u_min = 10
noise_integration_time = 1000 * 60 * 60 #3600000  # 1000 hours of observation time
tile_diameter = 35 
max_bl_length = None #10000 #500 #250 if DEBUG else 300
Tsys = 240
effective_collecting_area = 300 #np.pi * (tile_diameter/2)**2

include_earth_rotation_synthesis = False #True #
tot_daily_obs_time = 4 # in hours
beam_synthesis_time = 4 * 60 * 60 # in seconds
# foreground parameters 
S_min = 1e-6
S_max = 100e-3
uv_max = 1e4
padding_size = None 
n_uv = 4096 

# MCMC OPTIONS
# params = dict(  # Parameter dict as described above.
#     HII_EFF_FACTOR=[30.0, 10.0, 50.0, 3.0],
#     ION_Tvir_MIN=[4.7, 2, 8, 0.1],
# )
params=dict(             # Parameter dict as described above.
        F_STAR10 = [-1.3, -3, 0, 1.0], # -1.3
        ALPHA_STAR = [0.5, -0.5, 1.0, 1.0], # 0.5
        M_TURN = [7.69897, 8, 10, 0.1], # 8.69897
        t_STAR = [0.5, 0.01, 1, 0.01], # 0.5
        F_ESC10 = [-1, -3, 0, 1.0], # -1

        #L_X = [40.5, 38, 42, 0.1 ],
        #NU_X_THRESH = [500, 100, 1500, 10 ],
        #X_RAY_SPEC_INDEX = [1.0, -1, 3, 0.1 ],
        #ALPHA_ESC = [-0.5, -1.0, 0.5, 1.0],
    )

mcmc_options = {
        "n_live_points": 5,
        'importance_nested_sampling': False,
        'sampling_efficiency': 0.8,
        'evidence_tolerance': 0.5,
        # 'multimodal': True,
        'max_iter': 10,
        'write_output': True}

cosmo_params = {'SIGMA_8': 0.8118, 'hlittle': 0.6688, 'OMm': 0.321, 'OMb':0.04952, 'POWER_INDEX':0.9626}
flag_options = {'USE_MASS_DEPENDENT_ZETA': True}
global_params = {'Z_HEAT_MAX': 15.0, 'ZPRIME_STEP_FACTOR': 1.04}
z_step_factor = 1.04

# ----- Options that differ between DEBUG levels --------
HII_DIM = [512, 275, 150][DEBUG]
DIM = 3 * HII_DIM
BOX_LEN = (Planck18.comoving_transverse_distance(1420/freq_min -1).value * sky_size) / [1, 2, 8][DEBUG]

# Likelihood options
n_ubins = 30
nrealisations = [10, 10, 2][DEBUG]
# nthreads = int(sys.argv[1]) #51 #[24, 24, 24][DEBUG]

walkersRatio = [6, 4, 2][DEBUG]  # The number of walkers will be walkersRatio*nparams
sampleIterations = [1000, 20, 5][DEBUG]

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
    redshift = z_min,  # Lower redshift of the lightcone
    max_redshift = z_max,  # Approximate maximum redshift of the lightcone (will be exceeded).
    user_params = dict(
        HII_DIM = HII_DIM,
        BOX_LEN = BOX_LEN,
        DIM = DIM,
        N_THREADS = nthreads,
        USE_INTERPOLATION_TABLES = True
    ),
    astro_params = {
        # "HII_EFF_FACTOR":params["HII_EFF_FACTOR"][0], 
        # "ION_Tvir_MIN":params["ION_Tvir_MIN"][0]

        "F_STAR10": params["F_STAR10"][0],
        "ALPHA_STAR": params["ALPHA_STAR"][0],
        "M_TURN": params["M_TURN"][0],
        "t_STAR": params["t_STAR"][0],
        "F_ESC10": params["F_ESC10"][0]
    },
    z_step_factor = z_step_factor,  # How large the steps between evaluated redshifts are (log).
    regenerate = False,
    keep_data_in_memory = DEBUG,
    store = {
        "lc_slices": _store_lightcone,
        "2DPS": _store_2dps
    },
    # cosmo_params=cosmo_params,
    flag_options=flag_options,
    global_params=global_params,
    # change_seed_every_iter = True,
    initial_conditions_seed = 42,
    cache_mcmc = False,
    cache_dir = "/scratch/anasirudin/." #"/data/21cmfast-data/."
)

class CustomCoreInstrument(CoreInstrumental):
    def __init__(self, freq_min=freq_min, freq_max=freq_max, nfreq=nfreq,
                 sky_size=sky_size, n_cells=n_cells, tile_diameter=tile_diameter,
                 noise_integration_time=noise_integration_time,max_bl_length = max_bl_length,
                 **kwargs):
        super().__init__(freq_max = freq_max, freq_min=freq_min, n_obs = n_obs, nparallel = nthreads,
                         nfreq=  nfreq, tile_diameter = tile_diameter, noise_integration_time = noise_integration_time,
                         sky_extent = sky_size, n_cells = n_cells, max_bl_length = max_bl_length,
                         tot_daily_obs_time = tot_daily_obs_time, beam_synthesis_time = beam_synthesis_time,
                         padding_size = padding_size, include_earth_rotation_synthesis = include_earth_rotation_synthesis,
                         include_beam=True, beam_type=name_extension, diffuse_realization=diffuse_realization,
                         input_data_dir=input_data_dir,
                         **kwargs)

class CustomLikelihood(LikelihoodInstrumental2D):
    def __init__(self, n_ubins=n_ubins, uv_max=uv_max, nrealisations = nrealisations,
                 **kwargs):
        super().__init__(n_ubins=n_ubins, uv_max=uv_max, u_min=u_min, n_obs = n_obs, nparallel = nthreads, n_uv=n_uv,
                         simulate=True, nthreads=nthreads, nrealisations = nrealisations, ps_dim=2, include_fourierGaussianBeam=True,
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
        use_multinest=True,
        **mcmc_options #kwargs
    )

core_instr = CustomCoreInstrument(antenna_posfile = antenna_posfile, Tsys = Tsys, effective_collecting_area = effective_collecting_area,
    same_foreground = True, simulate_foreground=False)

# Add foregrounds core
core_fg_ps = CorePointSourceForegrounds(S_min = S_min, S_max= S_max)
# core_fg_diff = CoreDiffuseForegrounds()

likelihood = CustomLikelihood(
    datafile=[f'data/{model_name}.npz'],
    noisefile=[f'data/{model_name}.noise.npz'],
)

if __name__ == "__main__":
    chain = run_mcmc(
        [core_eor, core_fg_ps, core_instr], likelihood, #  core_eor, core_fg_ps, 
        model_name=model_name,             # Filename of main chain output
    )

    # nest = pymultinest.Analyzer(4, outputfiles_basename="data/MultiNest/%s" % model_name)
    # data = nest.get_data()

    # print(data.shape)

