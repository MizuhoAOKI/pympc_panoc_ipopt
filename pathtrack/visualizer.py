
"""
_  _ _ ____ _  _ ____ _    _ ___  ____ ____ 
|  | | [__  |  | |__| |    |   /  |___ |__/ 
 \/  | ___] |__| |  | |___ |  /__ |___ |  \ 

"""
# official libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from casadi import *
# custom libraries
from include.generic_module import *
from include.csvhandler import *
from include.svg_visualizer import svg_visualizer # Save vehicle trajectory as a SVG animation.

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
