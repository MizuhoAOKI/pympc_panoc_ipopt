"""
Pathtracking MPC simulation. 
"""

""" TODO
- output summary param file
- output movie
"""

# official libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import opengen as og # OpEn solver (PANOC Algorythm)
from casadi import *
from argparse import ArgumentParser
# custom libraries
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from include.generic_module import *
from include.csvhandler import *
from include import cubicspline as cubic
from include.svg_visualizer import svg_visualizer # Save vehicle trajectory as a SVG animation.

"""
____ _ _  _ _  _ _    ____ ___ ____ ____ 
[__  | |\/| |  | |    |__|  |  |  | |__/ 
___] | |  | |__| |___ |  |  |  |__| |  \ 

"""
# Vehicle Simulator with Kinematic Bicycle Model
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

"""
____ ____ _  _ ___ ____ ____ _    _    ____ ____ 
|    |  | |\ |  |  |__/ |  | |    |    |___ |__/ 
|___ |__| | \|  |  |  \ |__| |___ |___ |___ |  \ 

"""

#  Model Predictive Controller. Prediction model is Kinematic Bicycle Model (KBM)
class KBM_MPC():
    """ Prediction Model
    - state values : 
        x = [x[0], x[1], x[2], x[3]] = [y_e, \theta_e, x_e, V]
    - input values : 
        u = [u[0], u[1]] = [\delta, a] (front steer angle, acceleration)
    - state equation in the tex format: 
        \frac{d}{dt} y_e = V \sin(\theta_e + \beta)
        \frac{d}{dt} \theta_e = (V/l_r)\sin\beta - \frac{\rho V \cos(\theta_e + \beta)}{1 - \rho y_e}
        \frac{d}{dt} x_e = \frac{V \cos(\theta_e + \beta)}{1 - \rho y_e}
        \frac{d}{dt} V = a
        where, \beta = \arctan( \frac{l_r}{l_f + l_r} \tan\delta )
        l_f, l_r are parameters of the vehicle.
    """
    # constructor
    def __init__(self, mpc_parameter, simulation_setting, optimizer_config, prediction_model_config, planner_instance, solver_type, buildflag=True):
        self.planner = planner_instance
        self.current_s = 0.0
        self.solver_calc_time = 0.0
        self.curvature_list = []
        self.buildflag = buildflag
        self.solver_type = solver_type
        self.config_mpc = loadyaml(mpc_parameter)
        self.simulation_setting = loadyaml(simulation_setting)
        self.config_opt = loadyaml(optimizer_config)
        self.p_model = loadyaml(prediction_model_config)


        if self.solver_type == "ipopt": # if target solver is officially supported by casadi
            self.__init_normalsolver()
        elif self.solver_type == "panoc": # if you want to use unsupported panoc solver
            self.__init_panoc()
        else :
            print("Error. Please set integral_approach in prediction_model.yaml")
            sys.exit()

    # destructor
    def __del__(self):
        if self.solver_type == "panoc":
            self.mng.kill() # close tcp server

    ##### IPOPT #####

    # initialization process for supported solvers by casadi
    def __init_normalsolver(self):
        print(f"[INFO] Start initialization process to use {self.solver_type} solver.")

        # Construct nlp optimizer
        optimizer = self.optimizer = casadi.Opti()

        # Set prediction horizon
        self.N = self.n_step = self.config_mpc["prediction_step"]

        # Define variables to be optimized (u : control input)
        self.u = optimizer.variable(self.p_model["mpc_u_dim"],self.N)

        # Define parameters
        self.dt = optimizer.parameter(1)
        self.progressed_distance = optimizer.parameter(1)
        self.x0  = optimizer.parameter(self.p_model["mpc_x_dim"])
        self.q_terminal_weight = optimizer.parameter(len(self.config_mpc["q_terminal_weight"]))
        self.q_stage_weight = optimizer.parameter(len(self.config_mpc["q_stage_weight"]))
        self.r_stage_weight = optimizer.parameter(len(self.config_mpc["r_stage_weight"]))
        self.r_stage_diff_weight = optimizer.parameter(len(self.config_mpc["r_stage_diff_weight"]))
        self.curvature_list = optimizer.parameter(self.N)

        # set up the optimization problem
        x_t = [horzcat(self.x0)]
        self.total_cost = 0

        for t in range(0, self.N):
            if t != 0:
                self.total_cost += self.stage_diff_cost(self.u[:,t-1], self.u[:,t]) * self.dt # add cost to variable diff.
            self.total_cost += self.stage_cost(x_t[0][:], self.u[:,t]) * self.dt # update cost
            x_t = [horzcat(self.prediction_model(x_t[0][:], self.u[:,t], t))] # update state

            # set constraints
            max_steer = self.config_mpc["constraints"]["max_steer"]   # [deg]
            max_accel = self.config_mpc["constraints"]["max_accel"] # [deg] 
            self.optimizer.subject_to( self.u[0, t]**2 < (max_steer * np.pi / 180)**2 )
            self.optimizer.subject_to( self.u[1, t]**2 < (max_accel)**2 )

        # determine total cost
        self.total_cost += self.terminal_cost(x_t[0][:])  # terminal cost

        # finish setting up problem
        optimizer.minimize(self.total_cost)

    # compute the optimier with normalsolver
    def __calc_input_normalsolver(self, x0, initial_solution = None):

        # prepare curvature list
        curvature_list = []
        for t in range(self.N):
            curvature_list.append(self.planner.calc_curv(self.current_s + t * self.config_mpc["prediction_dt"]))

        # set parameters
        self.optimizer.set_value(self.x0, x0)
        self.optimizer.set_value(self.dt, self.config_mpc["prediction_dt"] )
        self.optimizer.set_value(self.q_terminal_weight, self.config_mpc["q_terminal_weight"])
        self.optimizer.set_value(self.q_stage_weight,   self.config_mpc["q_stage_weight"])
        self.optimizer.set_value(self.r_stage_weight,   self.config_mpc["r_stage_weight"])
        self.optimizer.set_value(self.r_stage_diff_weight,   self.config_mpc["r_stage_diff_weight"])
        self.optimizer.set_value(self.curvature_list, curvature_list)

        # set initial solution
        if initial_solution is None:
            initial_solution = [self.simulation_setting["initial_state"]["u"] for _ in range(self.n_step)]

        for k in range(self.n_step):
            self.optimizer.set_initial(
                self.u[:, k],
                initial_solution[k])

        # set optimizer
        # check https://helve-blog.com/posts/math/ipopt-print-level/ for detailed information
        self.optimizer.solver(self.config_opt["solver"], self.config_opt["s_opt"], self.config_opt["p_opt"])

        # optimization
        try:
            solution = self.optimizer.solve()
        except Exception:
            import traceback
            traceback.print_exc()
            solution = self.optimizer.debug
            print("\n ### Error termination due to solver failure. ###\n")
            return False

        # return time series of control input
        return solution.value(self.u).T

    ##### PANOC #####

    # Initialize panoc as a NLP solver.
    def __init_panoc(self):
        print(f"[INFO] Start initialization process to use {self.solver_type} solver.")

        # memory of solver result
        self.exit_status = ""
        self.num_outer_iterations = 0
        self.num_inner_iterations = 0
        self.last_problem_norm_fpr = 0.0
        self.delta_y_norm_over_c = 0.0
        self.f2_norm = 0.0
        # self.solve_time_ms = 0.0
        self.penalty = 0.0
        self.solution = []
        self.lagrange_multipliers = []

        # load parameters
        self.udim = self.p_model["mpc_u_dim"]
        self.xdim = self.p_model["mpc_x_dim"]
        self.N = self.n_step = self.config_mpc["prediction_step"]
        self.dt = self.config_mpc["prediction_dt"]

        if self.buildflag:

            # declare optimization parameters
            self.optimization_parameters = MX.sym('params', \
                                            self.xdim +
                                            len(self.config_mpc["q_terminal_weight"]) +
                                            len(self.config_mpc["q_stage_weight"]) +
                                            len(self.config_mpc["r_stage_weight"]) +
                                            len(self.config_mpc["r_stage_diff_weight"]) +
                                            self.N # to give curvature list
                                        )

            # TODO need refactoring
            self.x0 = self.optimization_parameters[0:4]
            self.q_terminal_weight = self.optimization_parameters[4:8]
            self.q_stage_weight = self.optimization_parameters[8:12]
            self.r_stage_weight = self.optimization_parameters[12:14]
            self.r_stage_diff_weight = self.optimization_parameters[14:16]
            self.curvature_list = self.optimization_parameters[16:16+self.N]

            # cotrol input series : u[u1, u2] for N step
            self.u_seq = [MX.sym('u_' + str(i), self.udim) for i in range(self.N)]

            # declare optimization parameters
            self.optimization_variables = self.u_seq

            # set up optimization problem
            x_t = self.x0
            self.total_cost = 0
            for t in range(0, self.N):
                if t != 0:
                    self.total_cost += self.stage_diff_cost(self.u_seq[t-1], self.u_seq[t]) * self.dt
                self.total_cost += self.stage_cost(x_t, self.u_seq[t]) * self.dt # update cost
                x_t = self.prediction_model(x_t, self.u_seq[t], t)        # update state

            # determine total cost
            self.total_cost += self.terminal_cost(x_t)  # terminal cost

            # build problem
            self.__build_problem()

        # launch tcp server
        self.__launch_tcp_server()

    def __build_problem(self):
        self.optimization_variables  = vertcat(*self.optimization_variables)

        # absolute max value of front steer angle and each tire torque
        max_steer = self.config_mpc["constraints"]["max_steer"]
        max_accel = self.config_mpc["constraints"]["max_accel"]
        umin = [-max_steer*np.pi/180.0, -max_accel] * self.N
        umax = [ max_steer*np.pi/180.0,  max_accel] * self.N
        bounds = og.constraints.Rectangle(umin, umax)

        problem = og.builder.Problem(self.optimization_variables, self.optimization_parameters, self.total_cost)\
                 .with_constraints(bounds) # add inequality constraints

        # set meta data
        meta = og.config.OptimizerMeta()                \
            .with_version("0.0.0")                      \
            .with_authors(["Mizuho Aoki"])              \
            .with_licence("CC4.0-By")                   \
            .with_optimizer_name("pathtrack") # longer name causes error on windows. 

        build_config = og.config.BuildConfiguration()  \
            .with_build_directory("python_build")  \
            .with_build_mode("release")              \
            .with_tcp_interface_config() # to call optimizer from python

        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-5)\
            .with_max_inner_iterations(1000)

        builder = og.builder.OpEnOptimizerBuilder(problem,
            metadata=meta,
            build_configuration=build_config,
            solver_configuration=solver_config)

        # start build (this process takes long time.)
        builder.build()

    def __launch_tcp_server(self):
        print("[INFO] Launch TCP server")
        self.mng = og.tcp.OptimizerTcpManager('python_build/pathtrack')
        self.mng.start()
        pong = self.mng.ping() # check if the server is alive
        print(pong)
        if pong["Pong"]==1:
            print("[INFO] Connection succeeded.")
        else:
            print("[ERROR] Connection failed.")
            import sys
            sys.exit()

    def __calc_input_panoc(self, x, u):

        # prepare curvature list
        curvature_list = []
        for t in range(self.N):
            curvature_list.append(self.planner.calc_curv(self.current_s + t * self.dt))

        print("[INFO] Call solver")
        optimization_parameters = []
        optimization_parameters = [*x] \
                                 + self.config_mpc["q_terminal_weight"]\
                                 + self.config_mpc["q_stage_weight"]\
                                 + self.config_mpc["r_stage_weight"]\
                                 + self.config_mpc["r_stage_diff_weight"]\
                                 + curvature_list

        response = self.mng.call(p=optimization_parameters, initial_guess=np.ravel(u).tolist()) # call the solver over TCP
        print("Got response from solver")

        if response.is_ok():
            # Solver returned a solution
            solution_data = response.get()
            u_star = np.array(solution_data.solution).reshape(self.N, self.udim)
            self.exit_status = solution_data.exit_status
            self.solver_calc_time = solution_data.solve_time_ms
            self.f2_norm = solution_data.f2_norm
            # print(f"optimal solution : u = \n{u_star}")
            print(f"exit status : {self.exit_status}")
            print(f"computation time = {self.solver_calc_time} [ms]")
            print(f"f2_norm = {self.f2_norm}")
            return u_star
        else:
            # Invocation failed - an error report is returned
            solver_error = response.get()
            error_code = solver_error.code
            error_msg = solver_error.message
            print(f"[ERROR] {error_code}")
            print(error_msg)
            return False

    ##### COMMON #####

    # set stage cost
    def stage_cost(self, x, u):
        cost = 0.5 * (self.q_stage_weight[0]*x[0]**2 + self.q_stage_weight[1]*x[1]**2 + self.q_stage_weight[2]*x[2]**2 + self.q_stage_weight[3]*(x[3]-self.simulation_setting["reference"]["v_ref"]/3.6)**2) \
               + self.r_stage_weight[0]*u[0]**2 + self.r_stage_weight[1]*u[1]**2
        return cost

    # set stage cost of input change
    def stage_diff_cost(self, old_u, new_u):
        cost = self.r_stage_diff_weight[0] * (new_u[0]-old_u[0])**2 + \
               self.r_stage_diff_weight[1] * (new_u[1]-old_u[1])**2
        return cost

    # set terminal cost
    def terminal_cost(self, x):
        cost = 0.5 * (self.q_terminal_weight[0]*x[0]**2 + self.q_terminal_weight[1]*x[1]**2 + self.q_terminal_weight[2]*x[2]**2 + self.q_terminal_weight[3]*(x[3]-self.simulation_setting["reference"]["v_ref"]/3.6)**2)
        return cost

    # discrete-time prediction model
    def prediction_model(self, x, u, k):

        # update states
        if self.p_model["integral_approach"] == "runge-kutta":
            # (1) update states with 4-dim Runge-Kutta Method
            k1 = self.dt * hcat( self.func_dxdt(x           , u, self.curvature_list[k]) ).T
            k2 = self.dt * hcat( self.func_dxdt(x + 0.5 * k1, u, self.curvature_list[k]) ).T
            k3 = self.dt * hcat( self.func_dxdt(x + 0.5 * k2, u, self.curvature_list[k]) ).T
            k4 = self.dt * hcat( self.func_dxdt(x +       k3, u, self.curvature_list[k]) ).T
            x = x + (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        elif self.p_model["integral_approach"] == "euler":
            # (2) update states with Euler Method
            x = x + self.dt * hcat( self.func_dxdt(x, u, self.curvature_list[k]) ).T
        else :
            print("Error. Please set integral_approach in prediction_model.yaml")
            sys.exit()
 
        x[1] = atan2( sin(x[1]) , cos(x[1]) ) # convert \theta_e range to -pi ~ pi

        # pass x series for N step
        return x

    # derivative of x by t
    def func_dxdt(self, x, u, curvature):

        # load vehicle parameters
        l_f = self.p_model["vehicle_parameter"]["l_f"]
        l_r = self.p_model["vehicle_parameter"]["l_r"]

        # Prediction model (KBM)
        beta = atan2( (l_r)*tan(u[0]), (l_f + l_r) )
        y_e_dot = x[3]*sin(x[1] + beta)
        theta_e_dot = x[3]/l_r * sin(beta) - (curvature*x[3]*cos(x[1] + beta))/(1.0 - curvature * x[0])
        x_e_dot = x[3]*cos(x[1] + beta)/(1.0 - curvature * x[0])
        velocity_dot = u[1]

        # core of time state control
        s_dot = (x[3]*cos(beta)*cos(x[1]) - x[3]*sin(beta)*sin(x[1])) / (1.0 - x[0] * fabs(curvature))

        # adjust format and return dxdt
        dxdt = [y_e_dot/s_dot, theta_e_dot/s_dot, x_e_dot/s_dot, velocity_dot/s_dot]
        return dxdt # type = casadi.casadi.MX

    def calc_input(self, x, u=None):
        if self.solver_type == "ipopt":
            return self.__calc_input_normalsolver(x, u)
        elif self.solver_type == "panoc":
            return self.__calc_input_panoc(x, u)
        else:
            print("Error. Please set solver_type in simulation_setting.yaml")
            sys.exit()

"""
___  _    ____ _  _ _  _ ____ ____ 
|__] |    |__| |\ | |\ | |___ |__/ 
|    |___ |  | | \| | \| |___ |  \ 

"""
# Class to handle reference path
class Pathtrack_Planner():
    def __init__(self, simulation_setting):
        self.simulation_setting = loadyaml(simulation_setting)
        self.ref_data_path = self.simulation_setting["reference"]["path"]
        self.ref = CSVHandler(self.ref_data_path)
        self.ref.csv_reader_for_float_array(ignore_row_num=1, get_info=False)
        # calculate cubic-spline interpolation of reference points
        self.x, self.y, self.yaw, self.curv_list, self.travel, self.spcoef, self.s, self.spline \
             = cubic.calc_2d_spline_interpolation(np.ravel(self.ref.points[:,0]), np.ravel(self.ref.points[:,1]), num=max(100, self.ref.rowsize))
        self.gx, self.gy, self.gyaw = self.x[0], self.y[0], self.yaw[0]
        self.nx, self.ny = self.x[0], self.y[0]
        self.prev_s, self.nearest_s = self.s[0], self.s[0]
        self.minor_s_index, self.major_s_index = (0, 1)
        self.MAX_FORWARD_SEARCH_POINTS = 10000 # [points]
        self.current_curvature = 0.0

    def calc_curv(self, traveled_distance):
        for i in range(0, self.ref.rowsize): # TODO : how about the case in going backward?
            if traveled_distance < self.s[i] : 
                __target_index = i
                break
        try:
            # TODO : use spline formula to get more accurate value
            __current_curvature = self.curv_list[__target_index]
        except UnboundLocalError:
            print("Maybe your car reached the end of the reference path.")
            return False

        return __current_curvature

    # coordinate transformation from global to frenet
    # from : state x(t) = [X[m], Y[m], Yaw[rad], V[m/s], Yaw_rate[rad/s], \beta[rad], ax[m/s^2], ay[m/s^2]]
    # to   : state x(s) = [y_e, \theta_e, V, Yaw_rate, \beta, ax, ay]
    def global_to_frenet(self, gx, gy, gyaw, gvelocity):
        # run nns() and determine nearest s 
        self.gx, self.gy, self.gyaw = gx, gy, gyaw

        if not self.nns():
            print("Error occured during NNS")
            sys.exit()

        # Frenet-Serret Frame yaw_e (heading error from the path direction)
        path_yaw = self.spline.calc_yaw(self.nearest_s)
        yaw_diff = gyaw - path_yaw
        heading_error = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff)) # -pi ~ pi

        # Frenet-Serret Frame y_e (lateral error from the path)
        self.nx, self.ny = self.spline.calc_position(self.nearest_s)
        # judge if (gx, gy) is left/right side of the path using tangent.
        # coefficients of tangent equation : Ax + By + C = 0
        A = -np.tan(path_yaw)
        B = 1.0
        C = -A*self.nx - self.ny
        sign = np.sign(A*gx + B*gy + C) # left : + , right : - .
        if not ( abs(path_yaw) < np.pi/2.0 ) : sign = -sign
        lateral_error = sign * np.sqrt( (self.nx - gx)**2 + (self.ny - gy)**2 )

        # x_e in Frenet-Serret frame  (traveled_distance on the path)
        traveled_distance = self.nearest_s
        self.current_curvature = self.calc_curv(traveled_distance)

        # return position in Frenet-Serret frame.
        return traveled_distance, [lateral_error, heading_error, traveled_distance, gvelocity]

    # nearest Neighbor Search
    def nns(self):
        __s_candidates = []

        for target_i in range(self.minor_s_index , max(self.minor_s_index + self.MAX_FORWARD_SEARCH_POINTS , self.ref.rowsize)-1):
            # find the closest line segment
            print(f"target_i = {target_i}")
            __s_coef = self.__calc_AtoF(self.s[target_i], self.gx, self.gy, *self.spcoef[target_i,:,0], *self.spcoef[target_i,:,1])
            __s_candidates = [i.real  for i in np.real_if_close(np.roots(__s_coef), tol=1000) if i.imag == 0.0 and (i >= self.s[target_i] and i<= self.s[target_i + 1])] # return only answers whose imaginary parts are very small.
            # print(__s_candidates)
            if __s_candidates:
                self.nearest_s = __s_candidates[0] # Are there any cases to get double solution?
                self.minor_s_index = target_i
                self.major_s_index = target_i + 1
                return True
            else:
                print(f"No solution when target_i = {target_i}, continue...")
                pass

        print("Appropriate nearest point was not found in the nns. Now switching the process to the simple_nns...")
        return False

    # calculate coefficients of equation to solve smallest distance between reference path and the target vehicle.
    def __calc_AtoF(self, s_j, x_G, y_G, a_x, b_x, c_x, d_x, a_y, b_y, c_y, d_y): # For cubic spline
        A = 6*d_x**2 + 6*d_y**2
        B = 10*c_x*d_x + 10*c_y*d_y - 30*d_x**2*s_j - 30*d_y**2*s_j
        C = 8*b_x*d_x + 8*b_y*d_y + 4*c_x**2 - 40*c_x*d_x*s_j + 4*c_y**2 - 40*c_y*d_y*s_j + 60*d_x**2*s_j**2 + 60*d_y**2*s_j**2
        D = 6*a_x*d_x + 6*a_y*d_y + 6*b_x*c_x - 24*b_x*d_x*s_j + 6*b_y*c_y - 24*b_y*d_y*s_j - 12*c_x**2*s_j + 60*c_x*d_x*s_j**2 - 12*c_y**2*s_j + 60*c_y*d_y*s_j**2 - 60*d_x**2*s_j**3 - 6*d_x*x_G - 60*d_y**2*s_j**3 - 6*d_y*y_G
        E = 4*a_x*c_x - 12*a_x*d_x*s_j + 4*a_y*c_y - 12*a_y*d_y*s_j + 2*b_x**2 - 12*b_x*c_x*s_j + 24*b_x*d_x*s_j**2 + 2*b_y**2 - 12*b_y*c_y*s_j + 24*b_y*d_y*s_j**2 + 12*c_x**2*s_j**2 - 40*c_x*d_x*s_j**3 - 4*c_x*x_G + 12*c_y**2*s_j**2 - 40*c_y*d_y*s_j**3 - 4*c_y*y_G + 30*d_x**2*s_j**4 + 12*d_x*s_j*x_G + 30*d_y**2*s_j**4 + 12*d_y*s_j*y_G
        F = 2*a_x*b_x - 4*a_x*c_x*s_j + 6*a_x*d_x*s_j**2 + 2*a_y*b_y - 4*a_y*c_y*s_j + 6*a_y*d_y*s_j**2 - 2*b_x**2*s_j + 6*b_x*c_x*s_j**2 - 8*b_x*d_x*s_j**3 - 2*b_x*x_G - 2*b_y**2*s_j + 6*b_y*c_y*s_j**2 - 8*b_y*d_y*s_j**3 - 2*b_y*y_G - 4*c_x**2*s_j**3 + 10*c_x*d_x*s_j**4 + 4*c_x*s_j*x_G - 4*c_y**2*s_j**3 + 10*c_y*d_y*s_j**4 + 4*c_y*s_j*y_G - 6*d_x**2*s_j**5 - 6*d_x*s_j**2*x_G - 6*d_y**2*s_j**5 - 6*d_y*s_j**2*y_G
        # print(f"S(s) = {a_x} (s-{s_j})^3 + {b_x} (s-{s_j})^2 + {c_x} (s-{s_j})^1 + {d_x} ")
        # print(f"S(s) = {a_y} (s-{s_j})^3 + {b_y} (s-{s_j})^2 + {c_y} (s-{s_j})^1 + {d_y} ")
        return (A, B, C, D, E, F)

"""
_    ____ ____ ____ ____ ____ 
|    |  | | __ | __ |___ |__/ 
|___ |__| |__] |__] |___ |  \ 

"""
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

"""
_  _ _ ____ _  _ ____ _    _ ___  ____ ____ 
|  | | [__  |  | |__| |    |   /  |___ |__/ 
 \/  | ___] |__| |  | |___ |  /__ |___ |  \ 

"""
# Visualizer of simulation with KBM.
class KBM_Visualizer():
    # constructor
    def __init__(self, sim_config, mpc_config, logger_instance):
        self.log = logger_instance
        self.setting    = loadyaml(sim_config)
        self.mpc_config = loadyaml(mpc_config)
        self.showfig    = self.setting["output"]["show_fig"]
        self.savefig    = self.setting["output"]["save_fig"]
        self.show_realtime_visualize = self.setting["output"]["show_realtime_visualize"]
        self.press_enter_to_start_visualization = self.setting["press_enter_to_start_visualization"]

        if self.show_realtime_visualize:
            self.max_steer = self.mpc_config["constraints"]["max_steer"]
            self.max_accel = self.mpc_config["constraints"]["max_accel"]
            self.__init_realtime_visualize()

    # set up figure for realtime visualization
    def __init_realtime_visualize(self):

        # prepare figure and axis
        plt.rcParams["font.size"] = 16
        self.realtime_fig, _ = plt.subplots() #figsize = (16,9)
        self.realtime_traj_ax           = plt.subplot2grid((3,6), (0,0), rowspan=2, colspan=6)
        self.realtime_delta_ax          = plt.subplot2grid((3,6), (2,1))
        self.realtime_accel_ax       = plt.subplot2grid((3,6), (2,4))

        # set labels
        ##  ax of main figure
        self.realtime_traj_ax.set_xlabel("Global X [m]")
        self.realtime_traj_ax.set_ylabel("Global Y [m]")
        self.realtime_traj_ax.set_aspect('equal')
        ## ax of delta
        self.realtime_delta_ax.set_title("Steering Angle", fontsize="12")
        ## add line of prediction horizon
        self.prediction_horizon, = self.realtime_traj_ax.plot([0.0], [0.0], linewidth=3, marker='.', markersize=10)
        self.cog_point, = self.realtime_traj_ax.plot(0., 0., marker='x', markersize=15, color="black")
        self.ref_path,  = self.realtime_traj_ax.plot([0.0], [0.0], color='purple',  linestyle='dashed')
        ## add lines shaped the car_body and the wheels
        self.car_body, = self.realtime_traj_ax.plot([0.],[0.],color="black",linewidth=1, zorder=3)
        self.wheel_a,  = self.realtime_traj_ax.plot([0.],[0.],color="black",linewidth=1, zorder=3)
        self.wheel_b,  = self.realtime_traj_ax.plot([0.],[0.],color="black",linewidth=1, zorder=3)
        self.wheel_c,  = self.realtime_traj_ax.fill([0.],[0.],edgecolor="black",fc="black",alpha=0.5,linewidth=1, zorder=3)
        self.wheel_d,  = self.realtime_traj_ax.fill([0.],[0.],edgecolor="black",fc="black",alpha=0.5,linewidth=1, zorder=3)

        # maximize matplotlib window
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        # figure layout setting
        self.realtime_fig.tight_layout(rect=[0,0,1,0.9])
        self.realtime_fig.subplots_adjust(wspace=0.4, hspace=0.6)

    # TODO: need more refactoring
    def realtime_visualize(self, current_time, mpc_x0, sim_x0, u, prediction_step_num, prediction_delta_time, update_state_func, planner):
        self.ref_path.set_data(planner.x, planner.y)
        self.realtime_fig.suptitle(f"Time = {current_time:.3f} [s]", fontsize=16)

        info_to_show =  r"$\rm{u = [delta[rad], accel[m/s^2]] = }$" + \
                        f"[{u[0,0]:.2f}"+r"$\rm{[rad]}$"+f", {u[0,1]:.2f}"+r"$\rm{[m/s^2]}$"+"] \n" + \
                        r"$\rm{mpc\_x = [y_e[m], \theta_e[rad/s], x_e[m], V[m/s]] = }$" +\
                        f"[{mpc_x0[0]:.2f}"+r"$\rm{[m]}$"+f" {mpc_x0[1]:.2f}"+r"$\rm{[rad]}$"+f", {mpc_x0[2]:.2f}"+r"$\rm{[m]}$"+f", {mpc_x0[3]:.2f}"+r"$\rm{[m/s]}$"+"] \n"+ \
                        r"$\rm{sim\_x = [X[m], Y[m], Yaw[rad], V[m/s]] = }$" +\
                        f"[{sim_x0[0]:.2f}"+r"$\rm{[m]}$"+f", {sim_x0[1]:.2f}"+r"$\rm{[m]}$"+f" {sim_x0[2]:.2f}"+r"$\rm{[rad]}$"+f", {sim_x0[3]:.2f}"+r"$\rm{[m/s]}$" +"]"

        self.realtime_traj_ax.set_title(info_to_show, fontsize=14)

        # calculate future x series in the prediction horizon
        future_x = np.array([sim_x0]) # x state at 0 step (now)
        for k in range(prediction_step_num): # k is 0 ~ N-1, step 1 ~ N

            if k == 0 :
                k_step_x = future_x # use initial value (0 step)
            else:
                k_step_x = future_x[-1,:] # keep k step x value

            # NOTE : call the update_state_func by value using .copy() func.
            k_plus_1_step_x = update_state_func(np.ravel(k_step_x.copy()), u[k,:], prediction_delta_time/np.ravel(k_step_x.copy())[3]) # get x value at k+1 step
            future_x = np.append(future_x, np.array([k_plus_1_step_x]), axis=0) # add x state at k+1 step (x is 1 ~ N in this loop)

        # draw future vehicle location in the prediction horizon
        self.prediction_horizon.set_data(future_x[:, 0], future_x[:, 1])

        # make alias of given variables
        front_tire_angle = u[0,0]
        global_x = sim_x0[0]
        global_y = sim_x0[1]
        global_yaw = sim_x0[2]

        # draw center of gravity of the vehicle
        self.cog_point.set_data([global_x, global_y])

        # draw vehicle body
        # vehicle visualization params
        L = 3.5/2. # length of vehicle body 3.5m 
        W = 2.1/2. # width of vehicle body 2.1m

        A = [-L,+W]
        B = [-L,-W]
        C = [+L,-W]
        D = [+L,+W]

        # wheel width : ww, wheel length : wl
        ww = 0.215/2. # 0.215m
        wl = 0.650/2. # 0.650m

        # points of wheel A
        AW = [-L+L/3.,+W-W/3.]
        AW_a = [AW[0]-wl,AW[1]+ww]
        AW_b = [AW[0]-wl,AW[1]-ww]
        AW_c = [AW[0]+wl,AW[1]-ww]
        AW_d = [AW[0]+wl,AW[1]+ww]

        # points of wheel B
        BW = [-L+L/3.,-W+W/3.]
        BW_a = [BW[0]-wl,BW[1]+ww]
        BW_b = [BW[0]-wl,BW[1]-ww]
        BW_c = [BW[0]+wl,BW[1]-ww]
        BW_d = [BW[0]+wl,BW[1]+ww]

        # points of wheel C
        CW = [+L-L/3.,-W+W/3.]
        CW_a = [CW[0]-wl,CW[1]+ww]
        CW_b = [CW[0]-wl,CW[1]-ww]
        CW_c = [CW[0]+wl,CW[1]-ww]
        CW_d = [CW[0]+wl,CW[1]+ww]

        # points of wheel D
        DW = [+L-L/3.,+W-W/3.]
        DW_a = [DW[0]-wl,DW[1]+ww]
        DW_b = [DW[0]-wl,DW[1]-ww]
        DW_c = [DW[0]+wl,DW[1]-ww]
        DW_d = [DW[0]+wl,DW[1]+ww]

        # draw vehicle body lines
        rotated_x, rotated_y =self.__rotate([A[0],B[0],C[0],D[0]],[A[1],B[1],C[1],D[1]],global_yaw,[global_x,global_y])
        self.car_body.set_data( rotated_x, rotated_y )

        # draw tires
        ## A wheel
        rotated_x_a, rotated_y_a =self.__rotate([AW_a[0],AW_b[0],AW_c[0],AW_d[0]],[AW_a[1],AW_b[1],AW_c[1],AW_d[1]],global_yaw,[global_x,global_y])
        self.wheel_a.set_data(rotated_x_a, rotated_y_a)
        ## B wheel
        rotated_x_b, rotated_y_b =self.__rotate([BW_a[0],BW_b[0],BW_c[0],BW_d[0]],[BW_a[1],BW_b[1],BW_c[1],BW_d[1]],global_yaw,[global_x,global_y])
        self.wheel_b.set_data(rotated_x_b, rotated_y_b)

        ## center of C wheel shaft
        rCWx, rCWy =self.__rotate([CW[0]],[CW[1]],global_yaw,[global_x,global_y])
        # C wheel
        rcx , rcy =self.__rotate([CW_a[0],CW_b[0],CW_c[0],CW_d[0]],[CW_a[1],CW_b[1],CW_c[1],CW_d[1]],global_yaw,[global_x,global_y])
        # tilt steer angle
        rotated_x_c,rotated_y_c =self.__rotate([rcx[0]-rCWx[0],rcx[1]-rCWx[0],rcx[2]-rCWx[0],rcx[3]-rCWx[0]],[rcy[0]-rCWy[0],rcy[1]-rCWy[0],rcy[2]-rCWy[0],rcy[3]-rCWy[0]],front_tire_angle,[rCWx[0],rCWy[0]])
        self.wheel_c.set_xy(np.array([rotated_x_c,rotated_y_c]).T)

        ## center of D wheel shaft
        rDWx, rDWy =self.__rotate([DW[0]],[DW[1]],global_yaw,[global_x,global_y])
        # D wheel
        rdx , rdy =self.__rotate([DW_a[0],DW_b[0],DW_c[0],DW_d[0]],[DW_a[1],DW_b[1],DW_c[1],DW_d[1]],global_yaw,[global_x,global_y])
        # tilt steer angle
        rotated_x_d,rotated_y_d =self.__rotate([rdx[0]-rDWx[0],rdx[1]-rDWx[0],rdx[2]-rDWx[0],rdx[3]-rDWx[0]],[rdy[0]-rDWy[0],rdy[1]-rDWy[0],rdy[2]-rDWy[0],rdy[3]-rDWy[0]],front_tire_angle,[rDWx[0],rDWy[0]])
        self.wheel_d.set_xy(np.array([rotated_x_d,rotated_y_d]).T)

        # set limit of plot area adding margin
        MARGIN_RATE = 15 # % margin
        MIN_MARGIN_X = 6 # [m]
        MIN_MARGIN_Y = 3 # [m]
        min_x, max_x = min(future_x[:,0]), max(future_x[:,0])
        min_y, max_y = min(future_x[:,1]), max(future_x[:,1])
        margin_x = max( (max_x - min_x) * MARGIN_RATE / 100.0, MIN_MARGIN_X)
        margin_y = max( (max_y - min_y) * MARGIN_RATE / 100.0, MIN_MARGIN_Y)
        self.realtime_traj_ax.set_xlim((min_x - margin_x), (max_x + margin_x))
        self.realtime_traj_ax.set_ylim((min_y - margin_y), (max_y + margin_y))

        # set params using constraints of control input (steering angle, acceleration)
        MAX_STEER_ANGLE = self.max_steer * np.pi /180 
        MAX_ACCEL= self.max_accel
        PIE_RATE = 3/4 # parameter to set layout of pie chart
        PIE_STARTANGLE = 225 # parameter to set layout of pie chart

        # update delta value
        self.realtime_delta_ax.clear()
        self.realtime_delta_ax.set_title("Tire Steer Angle", fontsize="12")
        steer = np.abs(u[0,0])

        # when turning right
        if u[0,0] < 0:
            color_delta = 'C3'
            print([MAX_STEER_ANGLE*PIE_RATE, steer*PIE_RATE, (MAX_STEER_ANGLE-steer)*PIE_RATE, 2*MAX_STEER_ANGLE*(1-PIE_RATE)])
            self.realtime_delta_ax.pie([MAX_STEER_ANGLE*PIE_RATE, steer*PIE_RATE, (MAX_STEER_ANGLE-steer)*PIE_RATE, 2*MAX_STEER_ANGLE*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", color_delta, "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        # when turning left
        else:
            color_delta = 'C0'
            print([MAX_STEER_ANGLE*PIE_RATE, steer*PIE_RATE, (MAX_STEER_ANGLE-steer)*PIE_RATE, 2*MAX_STEER_ANGLE*(1-PIE_RATE)])
            self.realtime_delta_ax.pie([(MAX_STEER_ANGLE-steer)*PIE_RATE, steer*PIE_RATE, MAX_STEER_ANGLE*PIE_RATE, 2*MAX_STEER_ANGLE*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", color_delta, "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        self.realtime_delta_ax.text(0, -1, f"{np.sign(u[0,1]) * steer:.2f} " + r"$ \rm{[rad]}$", size = 14, horizontalalignment='center', verticalalignment='center')

        # draw a steering wheel
        OUTER_RAD = 0.4 # just a param for layout
        INNER_RAD = 0.1 # just a param for layout
        outer_steering_wheel = patches.Circle(xy=(0, 0), radius=OUTER_RAD, ec='black', fc = 'white', linewidth = 4, fill=True)
        inner_steering_wheel = patches.Circle(xy=(0, 0), radius=INNER_RAD, ec='black', fc = 'white', linewidth = 4, fill=True)
        hbar = patches.Rectangle(xy=(-OUTER_RAD*np.cos(u[0,0]), -OUTER_RAD*np.sin(u[0,0])), width=2*OUTER_RAD, angle = (u[0,0] * 180.0/np.pi),  height=0.0, linewidth=3, ec='black', fill=False)
        vbar = patches.Rectangle(xy=(0, 0), width=OUTER_RAD, height=0.0, angle = 270 + (u[0,0] * 180.0/np.pi), linewidth=3, ec='black', fill=False)
        self.realtime_delta_ax.add_patch(outer_steering_wheel)
        self.realtime_delta_ax.add_patch(hbar)
        self.realtime_delta_ax.add_patch(vbar)
        self.realtime_delta_ax.add_patch(inner_steering_wheel)

        # update accel value
        color_fl = 'C0' if u[0,1] >= 0 else 'C3' # color is blue if torque >= 0, red if torque <= 0
        accel = min(np.abs(u[0,1]), MAX_ACCEL)
        self.realtime_accel_ax.clear()
        self.realtime_accel_ax.set_title("Acceleration", fontsize="12")
        self.realtime_accel_ax.pie([accel*PIE_RATE, (MAX_ACCEL-accel)*PIE_RATE, MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=[color_fl, "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        self.realtime_accel_ax.text(0, -1, f"{np.sign(u[0,1]) * accel:.2f} " + r"$ \rm{[m/s^2]}$", size = 14, horizontalalignment='center', verticalalignment='center')

        # update plot
        plt.pause(0.001)

    # rotate shape and return location on the x-y plane.
    def __rotate(self, xlist, ylist, angle, translation=[0.0, 0.0]): # angle [rad]
        rotated_x = []
        rotated_y = []
        if len(xlist) != len(ylist):
            print("In rotate() error occured! xlist and ylist must have same size.")
            return False
        for i, xval in enumerate(xlist):
            rotated_x.append((xlist[i])*np.cos(angle)-(ylist[i])*np.sin(angle)+translation[0])
            rotated_y.append((xlist[i])*np.sin(angle)+(ylist[i])*np.cos(angle)+translation[1])
        rotated_x.append(rotated_x[0])
        rotated_y.append(rotated_y[0])

        return rotated_x, rotated_y

    def mpc_visualize(self, output_filename="../result/mpc_result", file_format=".png", cmd_arg_comment=[]):
        __output_path = output_filename + file_format

        try:
            # MPC state values
            fig, ax = plt.subplots(2,3, figsize = (16,9))

            # Insert your comment given by second command line argument into the figure title.
            if len(cmd_arg_comment) >= 2 :
                fig.suptitle(str(cmd_arg_comment[1]))
            else: 
                fig.suptitle("Simulation Result (Pathtrack MPC with KBM)")

            ax[0][0].plot(self.log.timestamp, self.log.mpc_x_log[:,0])
            ax[0][0].set(title="", xlabel = "Time [s]", ylabel="y_e [m]")
            ax[0][1].plot(self.log.timestamp, self.log.mpc_x_log[:,1])
            ax[0][1].set(title="", xlabel = "Time [s]", ylabel="theta_e [rad]")
            ax[0][2].plot(self.log.timestamp, self.log.mpc_x_log[:,2])
            ax[0][2].set(title="", xlabel = "Time [s]", ylabel="x_e [m]")
            ax[1][0].plot(self.log.timestamp, self.log.mpc_x_log[:,3])
            ax[1][0].set(title="", xlabel = "Time [s]", ylabel="V [m/s]")
            ax[1][1].plot(self.log.timestamp, self.log.mpc_u_log[:,0], color="purple")
            ax[1][1].set(title="", xlabel = "Time [s]", ylabel="delta [rad]")
            ax[1][2].plot(self.log.timestamp, self.log.mpc_u_log[:,1], color="purple")
            ax[1][2].set(title="", xlabel = "Time [s]", ylabel="a [m/s^2]")
            fig.tight_layout()
            if self.savefig : fig.savefig(__output_path)
            if self.showfig : plt.show()
            print(f"Output MPC graph at {__output_path}")

        except:
            import traceback
            traceback.print_exc()
            sys.exit(False)

    def simulator_visualize(self, output_filename="../result/sim_result", file_format=".png", cmd_arg_comment=[]):
        __output_path = output_filename + file_format

        try:
            # MPC state values
            fig, ax = plt.subplots(2,3, figsize = (16,9))

            # Insert your comment given by second command line argument into the figure title.
            if len(cmd_arg_comment) >= 2 :
                fig.suptitle(str(cmd_arg_comment[1]))
            else: 
                fig.suptitle("Simulation Result (KBM Simulator)")

            ax[0][0].plot(self.log.timestamp, self.log.sim_x_log[:,0])
            ax[0][0].set(title="", xlabel = "Time [s]", ylabel="X [m]")
            ax[0][1].plot(self.log.timestamp, self.log.sim_x_log[:,1])
            ax[0][1].set(title="", xlabel = "Time [s]", ylabel="Y [m]")
            ax[0][2].plot(self.log.timestamp, self.log.sim_x_log[:,2])
            ax[0][2].set(title="", xlabel = "Time [s]", ylabel="Yaw [rad]")
            ax[1][0].plot(self.log.timestamp, self.log.sim_x_log[:,3])
            ax[1][0].set(title="", xlabel = "Time [s]", ylabel="V [m/s]")
            ax[1][1].plot(self.log.timestamp, self.log.sim_u_log[:,0], color="purple")
            ax[1][1].set(title="", xlabel = "Time [s]", ylabel="delta [rad]")
            ax[1][2].plot(self.log.timestamp, self.log.sim_u_log[:,1], color="purple")
            ax[1][2].set(title="", xlabel = "Time [s]", ylabel="a [m/s^2]")
            fig.tight_layout()
            if self.savefig : fig.savefig(__output_path)
            if self.showfig : plt.show()
            print(f"Output simulator graph at {__output_path}")

        except:
            import traceback
            traceback.print_exc()
            sys.exit(False)

    def trajectory_visualize(self, planner, output_filename="./result/trajectory", file_format=".png"):
        __output_fig_path = output_filename + file_format
        __output_svg_path = output_filename + ".svg"

        # Vehicle trajectory
        fig_traj, ax_traj = plt.subplots(1, 1, figsize = (16,9))
        fig_traj.suptitle("Vehicle Trajectory")
        ax_traj.plot(self.log.sim_x_log[:,0], self.log.sim_x_log[:,1], color='blue')
        ax_traj.plot(planner.x, planner.y, color='purple',  linestyle='dashed')
        ax_traj.set(title="", xlabel = "X [m]", ylabel="Y [m]")

        # Save figures
        if self.savefig : fig_traj.savefig(__output_fig_path)
        if self.showfig : plt.show()
        print(f"Output vehicle trajectory at {__output_fig_path}")

        # Save as svg animation
        svg_visualizer(self.log.timestamp, self.log.sim_x_log[:,0], self.log.sim_x_log[:,1], np.ravel(planner.x), np.ravel(planner.y),  __output_svg_path)
        print(f"Output svg animation at {__output_svg_path}")

"""
_  _ ____ _ _  _ ___  ____ ____ ____ ____ ____ ____ 
|\/| |__| | |\ | |__] |__/ |  | |    |___ [__  [__  
|  | |  | | | \| |    |  \ |__| |___ |___ ___] ___] 

"""
# main process to run simulation
def run_simulation(buildflag=True):

    # set path of config files
    COMMON_DIRPATH = "./config/" 
    MPC_CONFIG = COMMON_DIRPATH + "mpc_parameter.yaml"
    OPT_CONFIG = COMMON_DIRPATH + "optimizer_setting.yaml"
    SIM_CONFIG = COMMON_DIRPATH + "simulation_setting.yaml"
    PREDICTION_MODEL_CONFIG =  COMMON_DIRPATH + "prediction_model.yaml"
    SIMULATOR_MODEL_CONFIG  =  COMMON_DIRPATH + "simulator_model.yaml"

    # load simulation setting file
    setting = loadyaml(SIM_CONFIG)

    # launch planner, mpc, simulator, logger, and visualizer
    planner    = Pathtrack_Planner(SIM_CONFIG)
    mpc        = KBM_MPC(MPC_CONFIG, SIM_CONFIG, OPT_CONFIG, PREDICTION_MODEL_CONFIG, planner, setting["solver_type"], buildflag)
    plant      = KBM_Simulator(SIM_CONFIG, SIMULATOR_MODEL_CONFIG) # use full-vehicle-model simulator
    logger     = Logger(PREDICTION_MODEL_CONFIG, SIMULATOR_MODEL_CONFIG)
    visualizer = KBM_Visualizer(SIM_CONFIG, MPC_CONFIG, logger)

    # simulation setups
    sim_time = setting["simulation_time"] if setting["simulation_time"] != 0.0 else 100000000000 # simulation_time [s] 0 means running until the path ends.
    sim_dist = setting["simulation_dist"] if setting["simulation_dist"] != 0.0 else float('inf') # simulation_distance [m] 0 means running until the path ends.
    step_len = plant.dt                      # delta_time of simulation [s]
    sim_step = int (sim_time / step_len)     # simulation step [step] 
    control_input = plant.u                  # initial value of control input "u"
    u = [plant.u for _ in range(mpc.n_step)] # initial solution series

    # start simulation loop
    try: 
        sim_time = 0.0
        for k in range(sim_step):

            # coordinate transformation
            mpc.current_s, mpc_x = planner.global_to_frenet(*plant.x)

            # show current status
            print("##########")
            print(f"Time = {sim_time:.3f} [s]")
            print(f"Traveled distance = {mpc.current_s:.2f} [m]")
            print(f"mpc_x = [y_e, theta_e, x_e, V] = {mpc_x}")
            print(f"sim_x = [X, Y, Yaw, V] = {plant.x}")
            print(f"u = [steer[rad], accel[m/s^2]] = {control_input}")

            # termination condition
            if mpc.current_s > sim_dist : 
                print(f"Preset distance has been reached : {sim_dist} [m]")
                print("End of simulation.")
                break

            # save log
            logger.timestamp = np.append(logger.timestamp, [sim_time], axis=0)
            logger.calc_time = np.append(logger.calc_time, [mpc.solver_calc_time], axis=0)
            logger.prog_dist = np.append(logger.prog_dist, [mpc_x[2]], axis=0)
            logger.mpc_x_log = np.append(logger.mpc_x_log, np.array([mpc_x]), axis=0)
            logger.mpc_u_log = np.append(logger.mpc_u_log, [control_input], axis=0)
            logger.sim_x_log = np.append(logger.sim_x_log, np.array([plant.x]), axis=0)
            logger.sim_u_log = np.append(logger.sim_u_log, [control_input], axis=0)

            # optimization
            u = mpc.calc_input(mpc_x, u)
            if type(u) == bool : break   # if optimization failed : break for loop.
            control_input = u[0,:]       # if optimization succeeded : continue.

            # realtime visualization of prediction horizon
            if visualizer.show_realtime_visualize:
                visualizer.realtime_visualize(sim_time, mpc_x, plant.x, u, mpc.n_step, mpc.config_mpc["prediction_dt"], plant.simulate, planner)

                if visualizer.press_enter_to_start_visualization : 
                    input("Press Enter to start visualization")
                    visualizer.press_enter_to_start_visualization = False

            # simulation
            plant.x = plant.simulate(plant.x, control_input, step_len)
            if type(plant.x) == bool : break   # if optimization failed : break for loop.

            # update time
            sim_time += step_len

    except KeyboardInterrupt :
        # Catch "Ctrl + C" to close the program normally.
        print("\n ### Keyboard Interruption ###\n")
        pass

    print("Saving Results ...")

    # Export results in csv format
    logger.export_mpc_log(
        output_filename = setting["output"]["mpc"]["log"]["filename"], 
        file_format     = setting["output"]["mpc"]["log"]["format"],
        title_array     = setting["output"]["mpc"]["title_array"]
    )
    logger.export_simulator_log(
        output_filename = setting["output"]["simulator"]["log"]["filename"], 
        file_format     = setting["output"]["simulator"]["log"]["format"],
        title_array     = setting["output"]["simulator"]["title_array"]
    )

    # Visualize results and save figure
    visualizer.mpc_visualize(
        output_filename = setting["output"]["mpc"]["figure"]["filename"], 
        file_format     = setting["output"]["mpc"]["figure"]["format"],
        cmd_arg_comment = "" # give comment if necessary
    )
    visualizer.simulator_visualize(
        output_filename = setting["output"]["simulator"]["figure"]["filename"], 
        file_format     = setting["output"]["simulator"]["figure"]["format"],
        cmd_arg_comment = "" # give comment if necessary
    )
    visualizer.trajectory_visualize(planner)

    # delete instances
    del planner, mpc, plant, logger, visualizer

# set up argument parser
def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-nb', '--nobuild', action='store_true',
                           default=False,
                           help='Add --nobuild option in order not to build optimization problem.')
    return argparser.parse_args()

if __name__ == '__main__' :
    # get argument
    args = get_option()
    print(f"[INFO] build-flag is {str(not args.nobuild)}")

    # run main process
    run_simulation(buildflag=not args.nobuild)