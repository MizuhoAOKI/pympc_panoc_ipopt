"""
_  _ ____ _ _  _ ___  ____ ____ ____ ____ ____ ____ 
|\/| |__| | |\ | |__] |__/ |  | |    |___ [__  [__  
|  | |  | | | \| |    |  \ |__| |___ |___ ___] ___] 

"""

# official libraries
import os
import numpy as np
from casadi import *
from argparse import ArgumentParser
# custom libraries
from include.generic_module import *
from include.csvhandler import *
# import components for the simulation
from planner import Pathtrack_Planner
from controller import KBM_MPC
from simulator import KBM_Simulator
from logger import Logger
from visualizer import KBM_Visualizer

# main process to run simulation
def run_simulation(buildflag=True): # if buildflag is False, skip building problem of panoc.

    # set path of config files
    COMMON_DIRPATH = "./config/"
    MPC_CONFIG = os.path.join(COMMON_DIRPATH, "mpc_parameter.yaml")
    OPT_CONFIG = os.path.join(COMMON_DIRPATH, "optimizer_setting.yaml")
    SIM_CONFIG = os.path.join(COMMON_DIRPATH, "simulation_setting.yaml")
    PREDICTION_MODEL_CONFIG = os.path.join(COMMON_DIRPATH + "prediction_model.yaml")
    SIMULATOR_MODEL_CONFIG  = os.path.join(COMMON_DIRPATH + "simulator_model.yaml")

    # load simulation setting file
    setting = loadyaml(SIM_CONFIG)

    # launch planner, mpc, simulator, logger, and visualizer
    planner    = Pathtrack_Planner(SIM_CONFIG)
    mpc        = KBM_MPC(MPC_CONFIG, SIM_CONFIG, OPT_CONFIG, PREDICTION_MODEL_CONFIG, planner, setting["solver_type"], buildflag)
    plant      = KBM_Simulator(SIM_CONFIG, SIMULATOR_MODEL_CONFIG)
    logger     = Logger(PREDICTION_MODEL_CONFIG, SIMULATOR_MODEL_CONFIG)
    visualizer = KBM_Visualizer(SIM_CONFIG, MPC_CONFIG, logger)

    # simulation setups
    sim_time = setting["simulation_time"] if setting["simulation_time"] != 0.0 else 100000000000 # simulation_time [s], 0 means running until the path ends.
    sim_dist = setting["simulation_dist"] if setting["simulation_dist"] != 0.0 else float('inf') # simulation_distance [m], 0 means running until the path ends.
    step_len = plant.dt                      # delta_time of simulation [s]
    sim_step = int (sim_time / step_len)     # simulation step [step]
    u = [plant.u for _ in range(mpc.n_step)] # initial solution series

    # start simulation loop
    try: 
        sim_time = 0.0
        for k in range(sim_step): # loop for simulation steps

            # coordinate transformation from global frame to frenet-serret frame.
            lat_err, head_err, prog_dist = planner.global_to_frenet(plant.gx, plant.gy, plant.gyaw)

            # update mpc states
            mpc.update_states( \
                lateral_error = lat_err,   \
                heading_error = head_err,  \
                prog_dist     = prog_dist, \
                velocity      = plant.vel  \
            )

            # show current status
            print("#"*30)
            print(f"Time = {sim_time:.3f} [s]")
            print(f"Traveled distance = {mpc.prog_dist:.2f} [m]")
            print(f"mpc.x = [y_e[m], theta_e[rad], x_e[m], V[m/s]] = {mpc.x}")
            print(f"sim_x = [X[m], Y[m], Yaw[rad], V[m/s]] = {plant.x}")
            print(f"u = [steer[rad], accel[m/s^2]] = {plant.u}")

            # termination condition according to progressed distance
            if mpc.prog_dist > sim_dist : 
                print(f"Preset distance has been reached : {sim_dist} [m]")
                print("End of simulation.")
                break

            # save log
            logger.timestamp = np.append(logger.timestamp, [sim_time]             , axis=0)
            logger.calc_time = np.append(logger.calc_time, [mpc.calc_time] , axis=0)
            logger.prog_dist = np.append(logger.prog_dist, [mpc.prog_dist]        , axis=0)
            logger.mpc_x_log = np.append(logger.mpc_x_log, [mpc.x]                , axis=0)
            logger.mpc_u_log = np.append(logger.mpc_u_log, [mpc.u]                , axis=0)
            logger.sim_x_log = np.append(logger.sim_x_log, [plant.x]              , axis=0)
            logger.sim_u_log = np.append(logger.sim_u_log, [plant.u]              , axis=0)

            # calculate optimal control inputs
            u_star = mpc.calc_input(mpc.x, u)
            if type(u_star) == bool : break   # if optimization failed : break for loop.
            mpc.u = plant.u = u_star[0,:]     # if optimization succeeded : continue. use only first step as an actual control input.

            # realtime visualization of prediction horizon
            if visualizer.show_realtime_visualize:
                visualizer.realtime_visualize(sim_time, mpc.x, plant.x, u_star, mpc.n_step, mpc.config_mpc["prediction_dt"], plant.simulate, planner)

                # wait for pressing Enter to start first visualization
                if visualizer.press_enter_to_start_visualization : 
                    input("Press Enter to start visualization")
                    visualizer.press_enter_to_start_visualization = False

            # update plant states
            plant.x = plant.simulate(plant.x, plant.u, step_len)
            if type(plant.x) == bool : break   # if optimization failed : break for loop.
            else : plant.update_states()

            # update time
            sim_time += step_len

    except KeyboardInterrupt :
        # catch "Ctrl + C" to close the program normally.
        print("\n ### Keyboard Interruption ###\n")
        pass

    print("Saving Results ...")

    # export results in csv format
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
        cmd_arg_comment = "" # pass comment if necessary
    )
    visualizer.simulator_visualize(
        output_filename = setting["output"]["simulator"]["figure"]["filename"], 
        file_format     = setting["output"]["simulator"]["figure"]["format"],
        cmd_arg_comment = "" # pass comment if necessary
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

# load runtime arguments and launch main process.
if __name__ == '__main__' :
    # get argument
    args = get_option()
    print(f"[INFO] build-flag is {str(not args.nobuild)}")
    # run simulation
    run_simulation(buildflag=not args.nobuild)