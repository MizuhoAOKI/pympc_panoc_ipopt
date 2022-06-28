"""
___  _    ____ _  _ _  _ ____ ____ 
|__] |    |__| |\ | |\ | |___ |__/ 
|    |___ |  | | \| | \| |___ |  \ 

"""
# official libraries
import sys
import numpy as np
import opengen as og # OpEn solver (PANOC Algorythm)
from casadi import *
# custom libraries
from include.generic_module import *
from include.csvhandler import *
from include import cubicspline as cubic

"""
  Class to handle reference path
"""
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

    def calc_curv(self, progressed_distance):
        for i in range(0, self.ref.rowsize): # TODO : deal with the case in going backward?
            if progressed_distance < self.s[i] : 
                __target_index = i
                break
        try:
            # TODO : use spline formula to get more accurate value
            __current_curvature = self.curv_list[__target_index]
        except UnboundLocalError:
            print("Your car is assumed to be reached the end of the reference path.")
            return False

        return __current_curvature

    # coordinate transformation from global to frenet
    # from : state x(t) = [X[m], Y[m], Yaw[rad], V[m/s], Yaw_rate[rad/s], \beta[rad], ax[m/s^2], ay[m/s^2]]
    # to   : state x(s) = [y_e, \theta_e, V, Yaw_rate, \beta, ax, ay]
    def global_to_frenet(self, gx, gy, gyaw):
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
        sign = np.sign(A*gx + B*gy + C) # vehicle is [left : + , right : - ] side of the ref. path.
        if not ( abs(path_yaw) < np.pi/2.0 ) : sign = -sign
        lateral_error = sign * np.sqrt( (self.nx - gx)**2 + (self.ny - gy)**2 )

        # x_e in Frenet-Serret frame  (traveled_distance on the path)
        progressed_distance = self.nearest_s
        self.current_curvature = self.calc_curv(progressed_distance)

        # return position in Frenet-Serret frame.
        return lateral_error, heading_error, progressed_distance

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
