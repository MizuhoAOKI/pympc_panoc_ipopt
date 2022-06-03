"""
_    ____ ____ ____ ____ ____ 
|    |  | | __ | __ |___ |__/ 
|___ |__| |__] |__] |___ |  \ 

"""
# official libraries
import numpy as np
from casadi import *
# custom libraries
from include.generic_module import *
from include.csvhandler import *

# Class for logging simulation result
class Logger():
    # Constructor
    def __init__(self, mpc_setting, sim_setting):
        # load settings
        self.mpc_setting = loadyaml(mpc_setting)
        self.sim_setting = loadyaml(sim_setting)
        # initialize log storages
        self.timestamp = np.empty(0) # common timestamp between mpc and simulator
        self.calc_time = np.empty(0)
        self.prog_dist = np.empty(0)
        # setup loggers
        self.set_mpc_logger()
        self.set_simulator_logger()

    # set up logger for mpc
    def set_mpc_logger(self): 
        xdim = self.mpc_setting["mpc_x_dim"]
        udim = self.mpc_setting["mpc_u_dim"]
        # initialize strage of mpc states and inputs
        self.mpc_x_log = np.empty((0, xdim))
        self.mpc_u_log = np.empty((0, udim))

    # set up logger for simulator
    def set_simulator_logger(self):
        xdim = self.sim_setting["sim_x_dim"]
        udim = self.sim_setting["sim_u_dim"]
        # initialize storage of simulator states and inputs
        self.sim_x_log = np.empty((0, xdim))
        self.sim_u_log = np.empty((0, udim))

    # output mpc log.
    def export_mpc_log(self, output_filename="../result/mpc_result", file_format=".csv", title_array=None):
       
        if title_array is None:
            print("Error. Title name is not specified. ")
            return False

        if file_format == ".csv":
            __output_path = output_filename + file_format
            __data_array  = np.block([self.timestamp.reshape(-1, 1), self.mpc_x_log, self.mpc_u_log, self.calc_time.reshape(-1, 1), self.prog_dist.reshape(-1, 1)])
            csv_writer(__output_path, __data_array, title_array)
            print(f"Output MPC log at {__output_path}")
        else:
            print(f"Error. Specified format {file_format} is not supported.")

    # output simulator log.
    def export_simulator_log(self, output_filename="../result/simulator_result", file_format=".csv", title_array=None):

        if title_array is None:
            print("Error. Title name is not specified. ")
            return False

        if file_format == ".csv":
            __output_path = output_filename + file_format
            __data_array  = np.block([self.timestamp.reshape(-1, 1), self.sim_x_log, self.sim_u_log, self.calc_time.reshape(-1, 1), self.prog_dist.reshape(-1, 1)])
            csv_writer(__output_path, __data_array, title_array)
            print(f"Output simulator log at {__output_path}")
        else: 
            print(f"Error. Specified format {file_format} is not supported.")