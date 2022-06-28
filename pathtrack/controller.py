"""
____ ____ _  _ ___ ____ ____ _    _    ____ ____ 
|    |  | |\ |  |  |__/ |  | |    |    |___ |__/ 
|___ |__| | \|  |  |  \ |__| |___ |___ |___ |  \ 

"""
# official libraries
import sys
import numpy as np
import opengen as og # OpEn solver (PANOC Algorythm)
from casadi import *
# custom libraries
from include.generic_module import *
from include.csvhandler import *
"""
  Model Predictive Controller. Prediction model is Kinematic Bicycle Model (KBM)
"""
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
        self.prog_dist = 0.0
        self.calc_time = 0.0
        self.curvature_list = []
        self.buildflag = buildflag
        self.solver_type = solver_type
        self.config_mpc = loadyaml(mpc_parameter)
        self.simulation_setting = loadyaml(simulation_setting)
        self.config_opt = loadyaml(optimizer_config)
        self.p_model = loadyaml(prediction_model_config)
        self.x = [0] * self.p_model["mpc_x_dim"]
        self.u = [0] * self.p_model["mpc_u_dim"]

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

    # update states info
    def update_states(self, lateral_error, heading_error, prog_dist, velocity):
        self.x = [lateral_error, heading_error, prog_dist, velocity]
        self.prog_dist = prog_dist

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
            curvature_list.append(self.planner.calc_curv(self.prog_dist + t * self.config_mpc["prediction_dt"]))

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
            curvature_list.append(self.planner.calc_curv(self.prog_dist + t * self.dt))

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
            self.calc_time = solution_data.solve_time_ms
            self.f2_norm = solution_data.f2_norm
            # print(f"optimal solution : u = \n{u_star}")
            print(f"exit status : {self.exit_status}")
            print(f"computation time = {self.calc_time} [ms]")
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
