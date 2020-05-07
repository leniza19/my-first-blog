import numpy as np
import pandas as pd
from scipy.integrate import ode, solve_ivp
import matplotlib.pyplot as plt
import os
from mysite.settings import STATIC_ROOT
import time
import math


def ode_steam_ref(times, init, parms):
    keq_, GHSV_, L_, P0_, degree_, curretn_T_, k_ = parms
    R = 8.314
    P = P0_ * 10**5  # Давление
    m = np.array([44, 44, 2, 16, 18]) / 10**3
    x = np.array(init)
    kk = np.array(k_)

    if x[0] < 0.001:
        x[0] = 0

    if x[1] < 0.001:
        x[1] = 0

    if x[2] < 0.001:
        x[2] = 0

    if x[3] < 0.001:
        x[3] = 0

    if x[4] < 0.001:
        x[4] = 0

    yi = (x / m) * P0_ / np.sum(x / m)  # C3H8, CO2, H2, CH4, H2O

    Ci = yi * P / (R * curretn_T_)

    p_H2O = yi[m.size-1]
    p_CO2 = yi[1]
    p_H2 = yi[2]
    p_CH4 = yi[3]

    Diam_reac = 0.008  # 8 mm
    S_react = 3.14 * Diam_reac**2 / 4
    GHSV = GHSV_ / 3600  # s ^ (-1)
    V_react = S_react * L_
    u0 = V_react * GHSV
    F = u0 * P / (R * 298)
    g1 = np.sum(yi * m) * F
    G = g1 / S_react

    v_ref1 = np.array([-1, 3, 10, 0, - 6])  # for C3
    v_met = np.array([0, -1, -4, 1, 2])  # C3H8, CO2, H2, CH4, H2O

    if Ci[0] < 0.001:
        w_ref1 = 0
        Ci[0] = 0
    else:
        if degree_ < 0:
            w_ref1 = k_[0] / math.pow(Ci[0], (-1) * degree_)
        else:
            w_ref1 = k_[0] * math.pow(Ci[0], degree_)

    C_H2 = Ci[2]
    if (p_CH4 * p_H2O ** 2) < (keq_ * p_CO2 * p_H2 ** 4):
        w_met = kk[1] * C_H2 * (1 - (p_CH4 * p_H2O ** 2) / (keq_ * p_CO2 * p_H2 ** 4))
    else:
        w_met = 0

    #if curretn_T_ > 485:
    #    print(Ci, w_ref1, w_met)

    xdot = np.zeros(x.size)

    xdot = (1 / G) * ((v_met * w_met + v_ref1 * w_ref1) * m)

    return xdot


def k_eq(T):
    # [T] = K
    R = 8.314
    v = [-1, -4, 1, 2]  # CO2, H2, CH4, H2O

    # ------------- Табличные данные - --------------------------------------
    delH0F = np.array([-393.51, 0, - 74.85, - 241.81]) * 10**3
    S0_298 = np.array([213.66, 130.52, 188.72, 186.27])
    a = np.array([44.14, 27.28, 14.32, 30.00])
    b = np.array([9.04, 3.26, 74.66, 10.71]) / (10 ** 3)
    c = np.array([-8.54 * 10**5, 0.5 * 10**5, - 17.43 / 10**6, 0.33 * 10**5])

    delFH = np.zeros(4)
    delRH = np.zeros(4)
    delFS = np.zeros(4)

    # ----------------------------------------------------------------------
    delFH[0] = delH0F[0] + a[0] * (T - 298) + b[0] * (T**2 - 298**2) / 2 - c[0] * (1/T - 1/298)  # CO2
    delFH[1] = delH0F[1] + a[1] * (T - 298) + b[1] * (T**2 - 298**2) / 2 - c[1] * (1/T - 1/298)  # H2
    delFH[2] = delH0F[2] + a[2] * (T - 298) + b[2] * (T**2 - 298**2) / 2 + c[2] * (T**3 - 298**3) / 3  # CH4
    delFH[3] = delH0F[3] + a[3] * (T - 298) + b[3] * (T**2 - 298**2) / 2 - c[3] * (1/T - 1/298)  # H2O

    delRH = np.sum(v * delFH)

    delFS[0] = S0_298[0] + a[0] * (np.log(T) - np.log(298)) + b[0] * (T - 298) - (c[0] / 2) * (1/T**2 - 1/298**2)  # CO2
    delFS[1] = S0_298[1] + a[1] * (np.log(T) - np.log(298)) + b[1] * (T - 298) - (c[1] / 2) * (1/T**2 - 1/298**2)  # H2
    delFS[2] = S0_298[2] + a[2] * (np.log(T) - np.log(298)) + b[2] * (T - 298) + (c[2] / 2) * (T**2 - 298**2)  # CH4
    delFS[3] = S0_298[3] + a[3] * (np.log(T) - np.log(298)) + b[3] * (T - 298) - (c[3] / 2) * (1/T**2 - 1/298**2)  # H2O

    delRS = np.sum(v * delFS)
    delRG = delRH - T * delRS
    y = np.exp(-delRG / (R * T))
    return y

def arren(k0, Ea, T):
    R = 8.314
    T = T + 273.15
    y = k0 * np.exp(-Ea / (R * T))
    return y

def Ea_maker(x):
    """ From J to kJ """
    y = x * 1000
    return y


def k0_maker(x):
    """ exponentiation """
    y = np.power(10, x)
    return y

def dir_problem_power(Eref, k_ref, Emet, k_met, degree):
    print('starting direct - power model')
    #kin_param = np.array([1.1743e+02, 1.0997e+01, 5.9343e+01, 6.6367e+00, 6.2124e-01, 5.7235e-02])  # working
    kin_param = np.array([float(Eref), float(k_ref), float(Emet), float(k_met), float(degree)])  # working
    print(kin_param)
    degree = kin_param[4]
    print(degree)

    int_step = 0.4  # шаг интегрирования
    R = 8.314

    temper = []
    xexp = []

    const_count = 2  # без константы   равновесия (ref3, met)

    Ea = np.zeros(const_count)
    k0 = np.zeros(const_count)
    k = np.zeros(const_count)

    for j in range(0, const_count):
        Ea[j] = Ea_maker(kin_param[2 * j])  # 1, 3, 5, 7
        k0[j] = k0_maker((kin_param[2 * j + 1]))  # 2, 4, 6, 8

    # C3H8, CO2, H2, CH4, H2O
    c0_m = np.array([25, 0, 0, 0, 75])
    m = np.array([44, 44, 2, 16, 18]) / np.power(10, 3)  # g/Mole

    xi = c0_m / 100
    yi = xi * m / np.sum(xi * m)

    for exp_number in range(1, 5):
        if exp_number == 1:
            # ###################### exp 1 #######################################

            temper = np.array([223, 233, 245, 253, 260, 267, 277, 290, 293, 306, 319, 334, 353])  # C

            xexp = np.array([[86.425, 80.285, 72.215, 64.544, 59.181, 52.588, 41.744, 30.487, 28.250, 20.715, 11.281,  5.098,  1.101],  # C3H8
                [3.298, 4.569, 6.082, 7.289, 8.418, 9.450, 11.100, 11.863, 13.210, 14.586, 15.867, 17.147, 17.740],  # CO2
                [6.139, 8.018, 10.602, 9.211, 11.801, 12.237, 11.573, 3.341, 12.143, 10.737, 10.037, 10.919, 10.186],   # H2
                [4.008, 6.933, 10.831, 18.492, 20.167, 25.260, 35.053, 53.883, 45.900, 53.517, 62.496, 66.652, 70.908]  # CH4
                ]) / 100

            GHSV = 4000  # wet
            L = 3.3  # высота слоя, см
            P0 = 1.05

        if exp_number == 2:
            # ###################### exp 2 #######################################

            temper = np.array([235, 249, 264, 274, 283, 290, 298, 306, 313, 317, 331, 338, 347, 360, 374, 384])  # C
            xexp = np.array(
                [[89.685, 82.697, 74.778, 67.309, 62.231, 55.728, 47.813, 40.215, 32.368, 26.326, 13.824, 10.190, 7.788, 5.184, 2.996, 1.526],  # C3H8
                 [2.923, 4.285, 5.800, 7.200, 8.330, 9.293, 10.560, 11.709, 13.031, 13.528, 15.973, 16.648, 17.168, 17.961, 18.051, 18.389],  # CO2
                 [5.363, 7.690, 10.426, 12.262, 14.010, 13.859, 13.859, 12.910, 13.120, 12.047, 12.847, 13.504, 14.403, 15.520, 16.802, 16.783],  # H2
                 [1.982, 5.220, 8.835, 13.027, 15.230, 20.860, 27.479, 34.905, 41.230, 47.837, 57.214, 59.547, 60.554, 61.279, 62.117, 63.302]  # CH4
                 ]) / 100
            GHSV = 12000  # wet
            L = 3.3  # высота слоя, см
            P0 = 1.05

        if exp_number == 3:
            ###################### exp 3 #######################################
            temper = np.array([291, 308, 325, 341, 356, 372])  # C

            xexp = np.array(
                [[55.494, 37.138, 17.886, 6.972, 5.363, 0.624],  # C3H8
                [7.373, 10.436, 13.518, 15.166, 15.522, 16.125],  # CO2
                 [5.246, 6.372, 6.560, 6.217, 6.318, 6.263],  # H2
                 [31.887, 46.054, 62.036, 71.644, 72.796, 76.988]  # CH4
                  ]) / 100

            GHSV = 4000 / 5 # wet
            L = 2.3  # высота слоя, см
            P0 = 5

        if exp_number == 4:
            ###################### exp 4 #######################################
            temper = np.array([312, 328, 343, 375, 384])  # C

            xexp = np.array(
                [[77.272, 49.980, 25.788, 5.035, 0.235],  # C3H8
                [4.652, 8.442, 12.386, 15.821, 16.251],  # CO2
                [10.396, 8.982, 7.126, 9.807, 7.647],  # H2
                [7.679, 32.596, 54.699, 69.337, 75.866]  # CH4
                ]) / 100

            GHSV = 12000 / 5 # wet
            L = 2.3  # высота слоя, см
            P0 = 5

        #################################################################

        temper_calc = np.arange(temper[0] - 1, temper[temper.size - 1] + 2, 5)

        tspan = np.arange(0, L, int_step)

        #Outlet_all = np.zeros(xexp.shape)

        my_init = yi
        my_times = tspan

        Outlet_all = np.zeros((temper_calc.size, 4))
        F = 0

        for j in range(temper_calc.size):
            curretn_T = temper_calc[j] + 273.15  # K
            print(curretn_T)

            for i in range(const_count):
                k[i] = arren(k0[i], Ea[i], temper_calc[j])

            keq = k_eq(curretn_T)

            my_parms = [keq, GHSV, L, P0, degree, curretn_T, k]
            sir_sol = solve_ivp(fun=lambda t, y: ode_steam_ref(t, y, my_parms), t_span=[min(my_times), max(my_times)], method='Radau',
                                y0=my_init, t_eval=my_times)
            last = sir_sol.y.shape[1] - 1

            temp_sum = 0
            for i in range(m.size - 1):  # без H2O - последняя вода
                temp_sum = temp_sum + sir_sol.y[i][last] / m[i]

            outlet = np.array([sir_sol.y[0][last], sir_sol.y[1][last], sir_sol.y[2][last], sir_sol.y[3][last], sir_sol.y[4][last]])

            outlet = (outlet / m) / temp_sum  # xi(dry) - без H2O

            Outlet_all[j] = np.array([outlet[0], outlet[1], outlet[2], outlet[3]])  # - без H2O

            for i in range(temper.size):
                if temper[i] == temper_calc[j]:
                    F_temp = 0

                    for ii in range(m.size - 1):  # без H2O - последняя вода
                        F_temp = F_temp + abs(outlet[ii] - xexp[ii][i]) / xexp[ii][i]

                    F_temp = F_temp / (m.size - 1)  # без H2O - последняя вода;
                    F = F + F_temp
            del sir_sol, outlet


        F = F/temper.size
        #print(F)
        print('----------------------------------------')

        fig, ax = plt.subplots()
        ax.plot(temper_calc, Outlet_all[:, 0], label='C3H8', c = 'r')
        ax.plot(temper_calc, Outlet_all[:, 1], label='CO2', c = 'b')
        ax.plot(temper_calc, Outlet_all[:, 2], label='H2', c = 'g')
        ax.plot(temper_calc, Outlet_all[:, 3], label='CH4', c = 'c')

        ax.plot(temper, xexp[:][0], 'ro', marker='o', label='C3H8_exp')
        ax.plot(temper, xexp[:][1], 'bo', marker='o', label='CO2_exp')
        ax.plot(temper, xexp[:][2], 'go', marker='o', label='H2_exp')
        ax.plot(temper, xexp[:][3], 'co', marker='o', label='CH4_exp')


        plt.legend(loc='upper center', borderaxespad=0., ncol=2)

        ax.set(xlabel='T, C', ylabel='vol. %')
        ax.grid()

        filename = 'power_model' + str(exp_number) + '.png'
        file_path = os.path.join(STATIC_ROOT, filename)
        fig.savefig(file_path)
        #plt.show()
        # clear all variables
        del Outlet_all
