import numpy as np
import matplotlib.pyplot as plt 
import argparse
import json

def cost_function(w, W, H, theta_h, theta_v):
    return np.abs(
                np.tan(w * W / 2) / (w * np.tan(theta_h / 2)) - \
                np.tan(w * H / 2) / (w * np.tan(theta_v / 2))
        )

def pinhole_focal(W, H, theta_h, theta_v):
    fx = (W / 2) / (np.tan(theta_h / 2))
    fy = (H / 2) / (np.tan(theta_v / 2))
    err = fx - fy
    return fx, fy, err

def fx_from_w(w, W, theta_h):
    top = np.tan(w * W / 2)
    bot = w * np.tan(theta_h / 2)
    return top / bot

def fy_from_w(w, H, theta_v):
    top = np.tan(w * H / 2)
    bot = w * np.tan(theta_v / 2)
    return top / bot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find proper fisheye camera focal length with spec')
    parser.add_argument("-W",   "--Width",      type=int,   default=1920,   help='Width information of image frame')
    parser.add_argument("-H",   "--Height",     type=int,   default=1440,   help='Height information of image frame')
    parser.add_argument("-hf",  "--Hfov",       type=float, default=122.0,  help='Horizontal field of view angle. It should be float')
    parser.add_argument("-vf",  "--Vfov",       type=float, default=94.0,   help='Vertical field of view angle. It should be float')
    parser.add_argument("-d", "--draw", type=bool, default=False, help='draw plot or not')
    arg = parser.parse_args()

    # initialize params
    W, H = arg.Width, arg.Height
    cx, cy = (W - 1) / 2, (H - 1) / 2
    theta_h, theta_v = np.deg2rad(arg.Hfov), np.deg2rad(arg.Vfov)
    focal = 1000.
    
    # set optimize params 
    learning_rate = 0.1
    itr = 100
    w = 1. / focal
    w_list = []
    best_w = -1
    best_cost = np.inf 
    err_threshold = 1

    # find optimal w with stable newton-rapshon
    for i in range(itr):
        # define j stable value
        j_stable_1 = np.tan(theta_v / 2) / np.tan(theta_h / 2)
        j_stable_2 = np.tan(w * H / 2) / np.tan(w * W / 2)
        j_stable = j_stable_1 - j_stable_2
            
        # define diff j value
        diff_j_stable_1 = (-H / 2) * np.arccos(w * H / 2) ** 2 / np.tan(w * W / 2)        
        diff_j_stable_2 = (-W / 2) * np.tan(w * H / 2) / (1 + (w * W / 2) ** 2)
        diff_j_stable = diff_j_stable_1 + diff_j_stable_2

        # calculate w
        w = w - learning_rate * j_stable / (np.fabs(diff_j_stable) + 1e-6)
        w_list.append(w)

        # check error
        cost = cost_function(w, W, H, theta_h, theta_v)
        if cost < err_threshold and best_cost > cost:
            best_w = w
            best_cost = cost

    # calculate all costs
    cost_values = []
    for w in w_list:
        cost = cost_function(w, W, H, theta_h, theta_v)
        cost_values.append(cost)

    plt.plot(w_list, cost_values, '*', 'b')

    # draw costfunction with linspace
    w_list = np.linspace(0, 1.5 * 1e-3, 500)
    Jw_list = np.array([cost_function(w, W, H, theta_h, theta_v) for w in w_list])
    
    plt.plot(w_list, Jw_list, 'r')
    plt.plot(w_list, np.zeros_like(w_list), 'c')

    plt.xlabel('w')
    plt.ylabel('j')
    if arg.draw:
        plt.show()
    
    fx = fx_from_w(best_w, W, theta_h)
    fy = fy_from_w(best_w, H, theta_v)

    print('best cost:', best_cost)    
    print(f'fx: {fx:.2f}')
    print(f'fy: {fy:.2f}')
