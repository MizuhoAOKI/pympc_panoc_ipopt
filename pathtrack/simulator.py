"""
____ _ _  _ _  _ _    ____ ___ ____ ____ 
[__  | |\/| |  | |    |__|  |  |  | |__/ 
___] | |  | |__| |___ |  |  |  |__| |  \ 

"""
# official libraries
import sys
import numpy as np
from casadi import *
# custom libraries
from include.generic_module import *
from include.csvhandler import *

"""
  Vehicle Simulator with Kinematic Bicycle Model
"""
class KBM_Simulator():
    """ Simulator Model
    - state values : 
        x = [x[0], x[1], x[2], x[3]] = [X, Y, Yaw, V]
    - input values : 
        u = [u[0], u[1]] = [\delta, a] (front steer angle, acceleration)
    """
    # constructor
    def __init__(self, simulation_setting, simulator_model_config):
        # Load config files
        self.simulation_setting = loadyaml(simulation_setting)
        self.s_model = loadyaml(simulator_model_config)
        # Initialize state variables
        self.x = []
        self.u = []
        self.dt = 0
        self.sim_time = 0
        self.__initialize_variables()

    # initialize variables
    def __initialize_variables(self):
        self.x  = self.simulation_setting["initial_state"]["x"]
        self.u  = self.simulation_setting["initial_state"]["u"]
        self.dt = self.simulation_setting["delta_time"]
        self.sim_time = self.simulation_setting["simulation_time"]

    # update x with given dt
    def simulate(self, x, u, dt):
        # update state
        if self.s_model["integral_approach"] == "runge-kutta":
            # (1) update 4-dim Runge-Kutta Method
            k1 = dt * self.func_dxdt(x            , u, dt)
            k2 = dt * self.func_dxdt(x + 0.5 * k1 , u, dt)
            k3 = dt * self.func_dxdt(x + 0.5 * k2 , u, dt)
            k4 = dt * self.func_dxdt(x +       k3 , u, dt)
            x += (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        elif self.s_model["integral_approach"] == "euler":
            # (2) update states with Euler Method
            x += self.dt * self.func_dxdt(x, u, dt)
        else :
            print("[Error] Please set integral_approach in prediction_model.yaml")
            sys.exit()

        x[2] = np.arctan2( np.sin(x[2]) , np.cos(x[2]) ) # convert yaw range to -pi ~ pi
        return x

    # derivative of x by t
    def func_dxdt(self, x, u, dt):
        # load vehicle parameters
        l_f = self.s_model["vehicle_parameter"]["l_f"]
        l_r = self.s_model["vehicle_parameter"]["l_r"]

        # calculate dxdt by Kinematic Bicycle Model in Global frame
        beta = np.arctan((l_r)/(l_f + l_r) * np.tan(u[0]))
        x_dot = x[3]*np.cos(x[2]+beta)
        y_dot = x[3]*np.sin(x[2]+beta)
        yaw_rate = x[3]*np.sin(beta) / l_r
        vel_dot = u[1]

        # adjust format and return dxdt
        dxdt = np.array([x_dot, y_dot, yaw_rate, vel_dot])
        return dxdt
