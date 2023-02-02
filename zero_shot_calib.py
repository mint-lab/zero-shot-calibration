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

def zero_shot_calib_focal(width: float, 
                    height: float,
                    hfov: float,
                    vfov: float,
                    focal: float = 1000,
                    lr: float = 0.1,
                    itr: int = 100,
                    viz_cost_plot: bool = False) -> float:
    w = 1. / focal
    w_list = []
    best_w = -1
    best_cost = np.inf 
    err_threshold = 1

    for i in range(itr):
        # define j stable value
        j_stable_1 = np.tan(vfov / 2) / np.tan(hfov / 2)
        j_stable_2 = np.tan(w * height / 2) / np.tan(w * width / 2)
        j_stable = j_stable_1 - j_stable_2
            
        # define diff j value
        diff_j_stable_1 = (-height / 2) * np.arccos(w * height / 2) ** 2 / np.tan(w * width / 2)
        diff_j_stable_2 = (-width / 2) * np.tan(w * height / 2) / (1 + (w * width / 2) ** 2)
        diff_j_stable = diff_j_stable_1 + diff_j_stable_2

        # calculate w
        w = w - lr * j_stable / (np.fabs(diff_j_stable) + 1e-6)
        
        if viz_cost_plot:
            w_list.append(w)

        # check error
        cost = cost_function(w, width, height, hfov, vfov)
        if cost < err_threshold and best_cost > cost:
            best_w = w
            best_cost = cost

    # Check optimization results
    if best_cost == np.inf :
        raise Exception("Cost is too high. Focal length is not optimized")

    elif best_cost > 1. :
        raise Exception("Cost is high. Focal length is not fully optimized. You can change itr bigger")

    # Plotting cost values
    if viz_cost_plot:
        cost_values = []
        for w in w_list:
            cost = cost_function(w, W, H, theta_h, theta_v)
            cost_values.append(cost)

        plt.plot(w_list, cost_values, '*', 'b')

        w_base = np.linspace(0, 1.5 * 1e-3, 500)
        J_list = np.array([cost_function(w, W, H, theta_h, theta_v) for w in w_base])
        
        plt.plot(w_base, J_list, 'r', label='Optimization plotting')
        plt.plot(w_base, np.zeros_like(w_base), 'c', label='cost in all ranges')

        plt.legend()

        plt.xlabel('w')
        plt.ylabel('j')
        plt.show()


    fx = fx_from_w(best_w, width, hfov)
    fy = fy_from_w(best_w, height, vfov)

    return (fx + fy) / 2.

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
    focal = W / 2.
    
    fx, fy, err = pinhole_focal(W, H, theta_h, theta_v)
    print("before calibration")
    print(f"fx: {fx:.2f}, fy: {fy:.2f}, err: {err:.2f}")
    print(zero_shot_calib_focal(W, H, theta_h, theta_v, focal, 0.1, viz_cost_plot=True))