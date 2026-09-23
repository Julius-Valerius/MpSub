#Codes of a model-based method for solving large-scale DFO
#Copyright: Pengcheng Xie & Ya-xiang Yuan
#Connect: xpc@lsec.cc.ac.cn



from typing import Callable
import numpy as np
from trust_sub import trust_sub
from scipy.linalg import null_space
from math import sqrt


class Option:
    def __init__(self) -> None:
        self.delta = 1
        self.delta_low = 1e-4
        self.delta_up = 1e4
        self.gamma1 = 10
        self.gamma2 = 1 / 10
        self.eta = 1 / 5
        self.eta0 = 1 / 10
        self.verbose = False


class FuncWrapper:
    def __init__(self, f: Callable[[np.ndarray], float]) -> None:
        self.call_counter = 0
        self.f = f

    def __call__(self, x: np.ndarray) -> float:
        self.call_counter += 1
        return self.f(x)


def solvetry(f: Callable[[np.ndarray], float], x0: np.ndarray, option: Option = None)\
        -> np.ndarray:
    temp = []
    # preparison
    option = option or Option()
    n = x0.size
    k = 1  # iteration
    total_k = 5    #min(100 * n, 100000)#max iteration steps
    fwp = FuncWrapper(f)#calc function result

    # start
    if option.verbose:#log
        print(f"Iter: {k}")
    d0 = np.r_[1, np.zeros(n - 1)]
    ya, yb = x0, x0 + option.delta * d0#move one step
    fya, fyb = fwp(ya), fwp(yb)

    if fya >= fyb:
        yc = ya + 2 * option.delta * d0#good-->move an extra step
    else:
        yc = ya - option.delta * d0#bad-->move one step opposite
    fyc = fwp(yc)

    ys = [ya, yb, yc]
    y_min_idx = np.argmin([fya, fyb, fyc])
    ymin1 = ys[y_min_idx]#get the best point in ya,yb and yc
    fymin1 = [fya, fyb, fyc][y_min_idx]#get the best function result in ya,yb,yc

    xk = ymin1#set as the starter point for the algorithm
    if option.verbose:
        print(f"xk: {xk:.6f}")

    y_max_idx = np.argmax([fya, fyb, fyc])
    ymax1 = ys[y_max_idx]
    if np.linalg.norm(ymin1 - ymax1) == 0:
        ymax1 = ys[2 - y_max_idx]
    dd1: np.ndarray = (ymin1 - ymax1) / np.linalg.norm(ymin1 - ymax1)
    # Fast Gram-Schmidt projection to avoid massive matrix SVD
    rand_vec = np.random.randn(n)
    dd2 = rand_vec - np.dot(rand_vec, dd1) * dd1
    dd2 = dd2 / np.linalg.norm(dd2)

    alpha_a = (ya - ymin1) @ dd1
    alpha_b = (yb - ymin1) @ dd1
    alpha_c = (yc - ymin1) @ dd1

    A = np.array([
        [alpha_a, alpha_a * alpha_a],
        [alpha_b, alpha_b * alpha_b],
        [alpha_c, alpha_c * alpha_c]
    ])
    b = np.array([
        [fya - fymin1],
        [fyb - fymin1],
        [fyc - fymin1]
    ])
    a_value, b_value = np.linalg.lstsq(A, b, rcond=None)[0].flatten()

    while True:#if xk have not changed, reuse previous data
        if k > 1 and np.linalg.norm(xk - y1) == 0:
            fxk = fy1
        elif k > 1 and np.linalg.norm(xk - y2) == 0:
            fxk = fy2
        elif k > 1 and np.linalg.norm(xk - y3) == 0:
            fxk = fy3
        elif k > 1 and np.linalg.norm(xk - xkm1) == 0:
            fxk = fxkm1
        elif 'y6' in locals().keys() and np.linalg.norm(xk - y6) == 0:
            fxk = fy6
        elif 'y8' in locals().keys() and np.linalg.norm(xk - y8) == 0:
            fxk = fy8
        elif k > 1 and np.linalg.norm(xk - xk_plus) == 0:
            fxk = fxk_plus
        elif ('xk_plus_again' in locals().keys() and
              np.linalg.norm(xk - xk_plus_again) == 0):
            fxk = fxk_plus_again
        else:
            fxk = fwp(xk)

        y1 = xk + option.delta * dd2#explore second dimension

        if k > 1 and np.linalg.norm(y1 - xkm1) == 0:
            fy1 = fxkm1
        else:
            fy1 = fwp(y1)

        if fy1 <= fxk:
            y2 = xk + 2 * option.delta * dd2
        else:
            y2 = xk - option.delta * dd2

        if k > 1 and np.linalg.norm(y2 - xkm1) == 0:
            fy2 = fxkm1
        else:
            fy2 = fwp(y2)

        Y2s = [y1, y2]
        Y2val = [fy1, fy2]
        Y_min_idx = np.argmin(Y2val)
        ymin2 = Y2s[Y_min_idx]
        y3 = ymin2 + option.delta * dd1

        if k > 1 and np.linalg.norm(y3 - xkm1) == 0:
            fy3 = fxkm1
        else:
            fy3 = fwp(y3)

        flag = True

        while True:

            # Perform the interpolation for Q_k

            temp_1 = y1 - xk
            temp_2 = y2 - xk
            temp_3 = y3 - xk

            alpha1 = temp_1 @ dd1
            beta1 = temp_1 @ dd2

            alpha2 = temp_2 @ dd1
            beta2 = temp_2 @ dd2

            alpha3 = temp_3 @ dd1
            beta3 = temp_3 @ dd2

            A = np.array([
                [beta1, beta1 * beta1, alpha1 * beta1],
                [beta2, beta2 * beta2, alpha2 * beta2],
                [beta3, beta3 * beta3, alpha3 * beta3]
            ])
            b = np.array([
                [fy1 - fxk - a_value * alpha1 - b_value * alpha1 * alpha1],
                [fy2 - fxk - a_value * alpha2 - b_value * alpha2 * alpha2],
                [fy3 - fxk - a_value * alpha3 - b_value * alpha3 * alpha3]
            ])
            [c_value, d_value, e_value] = np.linalg.lstsq(A, b, rcond=None)[0].flatten()

            # Trust-region trial step
            g = np.array([[a_value], [c_value]])
            H = np.array([[2 * b_value, e_value], [e_value, 2 * d_value]])

            s_raw = trust_sub(g, H, option.delta)
            s = s_raw[0] if isinstance(s_raw, tuple) else s_raw

            s = np.clip(s, -5e-3, 5e-3)
            xk_plus = xk + (np.array([dd1, dd2]).T @ s).flatten()

            fxk_plus = fwp(xk_plus)

            if np.linalg.norm(s) == 0:

                xkp1 = xk
                fxkp1 = fxk
                flag = False
                break
            else:

                rho_k = ((fxk_plus - fxk) / (1 / 2 * s.T @ H @ s + g.T @ s + 1e-8)).item()

                if rho_k >= option.eta0:
                    xkp1 = xk_plus
                    fxkp1 = fxk_plus
                    temp.append(k)
                    break
                else:

                    if (k > 1):
                        temp_4_again = xkm1 - xk
                        temp_5_again = xk_plus - xk

                        alpha4_again = temp_4_again @ dd1
                        beta4_again = temp_4_again @ dd2

                        alpha5_again = temp_5_again @ dd1
                        beta5_again = temp_5_again @ dd2

                        if np.linalg.norm(xk - xkm1) != 0:
                            A = np.array([
                                [alpha1, alpha1 ** 2, beta1, beta1 ** 2, alpha1 * beta1],
                                [alpha2, alpha2 ** 2, beta2, beta2 ** 2, alpha2 * beta2],
                                [alpha3, alpha3 ** 2, beta3, beta3 ** 2, alpha3 * beta3],
                                [alpha4_again, alpha4_again * 2, beta4_again, beta4_again * 2, alpha4_again * beta4_again],
                                [alpha5_again, alpha5_again * 2, beta5_again, beta5_again * 2, alpha5_again * beta5_again]
                            ])
                            b = np.array([
                                [fy1 - fxk], [fy2 - fxk], [fy3 - fxk],
                                [fxkm1 - fxk], [fxk_plus - fxk]
                            ])
                        else:
                            y8_again = xk + \
                                sqrt(2) / 2 * option.delta * dd2 + \
                                sqrt(2) / 2 * option.delta * dd1
                            # y8_again refers to y4 in the paper

                            if np.linalg.norm(xk_plus - y8) != 0:

                                fy8_again = fwp(y8_again)

                                temp_plus_8_again = y8_again - xk
                                alpha8_again = temp_plus_8 @ dd1
                                beta8_again = temp_plus_8 @ dd2

                                A = np.array([
                                    [alpha1, alpha1 ** 2, beta1, beta1 ** 2, alpha1 * beta1],
                                    [alpha2, alpha2 ** 2, beta2, beta2 ** 2, alpha2 * beta2],
                                    [alpha3, alpha3 ** 2, beta3, beta3 ** 2, alpha3 * beta3],
                                    [alpha8_again, alpha8_again * 2, beta8_again, beta8_again * 2, alpha8_again * beta8_again],
                                    [alpha5_again, alpha5_again * 2, beta5_again, beta5_again * 2, alpha5_again * beta5_again]
                                ])

                                b = np.array([fy1 - fxk, fy2 - fxk, fy3 - fxk,
                                              fy8_again - fxk, fxk_plus - fxk])[:, np.newaxis]

                            else:

                                y6_again = xk + option.delta * dd1
                                # y6_again refers to y5 in the paper

                                fy6_again = fwp(y6_again)

                                temp_plus_6_again = y6_again - xk
                                alpha6_again = temp_plus_6_again  @ dd1
                                beta6_again = temp_plus_6_again  @ dd2

                                A = np.array([
                                    [alpha1, alpha1 ** 2, beta1, beta1 ** 2, alpha1 * beta1],
                                    [alpha2, alpha2 ** 2, beta2, beta2 ** 2, alpha2 * beta2],
                                    [alpha3, alpha3 ** 2, beta3, beta3 ** 2, alpha3 * beta3],
                                    [alpha6_again, alpha6_again ** 2, beta6_again, beta6_again ** 2, alpha6_again * beta6_again],
                                    [alpha5_again, alpha5_again ** 2, beta5_again, beta5_again ** 2, alpha5_again * beta5_again]
                                ])

                                b = np.array([[fy1 - fxk], [fy2 - fxk], [fy3 - fxk],
                                              [fy6_again - fxk], [fxk_plus - fxk]])

                        [a_value_again, b_value_again, c_value_again, d_value_again, e_value_again] = np.linalg.lstsq(A, b, rcond=None)[0].flatten()

                        # Trust-region trial step (again) for Qmod
                        g_again = np.array([[a_value_again], [c_value_again]])
                        H_again = np.array([[2 * b_value_again, e_value_again], [e_value_again, 2 * d_value_again]])

                        s_again_raw = trust_sub(g_again, H_again, option.delta)
                        s_again = s_again_raw[0] if isinstance(s_again_raw, tuple) else s_again_raw

                        s_again = np.clip(s_again, -5e-3, 5e-3)
                        xk_plus_again = xk + (np.array([dd1, dd2]).T @ s_again).flatten()

                        fxk_plus_again = fwp(xk_plus_again)
                        if fxk_plus_again < fxk_plus:
                            xk_plus = xk_plus_again
                            fxk_plus = fxk_plus_again

                        rho_k = (fxk_plus - fxk) / ((1 / 2 * s_again.T @ H_again @ s_again + g_again.T @ s_again + 1e-8))

                        if rho_k >= 0:
                            xkp1 = xk_plus
                            fxkp1 = fxk_plus
                            temp.append(k)
                            break
                        else:
                            xkp1 = xk
                            fxkp1 = fxk
                            flag = False
                            break

                    else:
                        xkp1 = xk
                        fxkp1 = fxk
                        flag = False
                        break

        oldDelta = option.delta

        # Update the trust-region radius and the subspace
        if (option.delta < option.delta_low) or (k > total_k):
            xreturn = xk
            fval = fxk
            return xreturn, fval, temp, k

        if rho_k >= option.eta:
            if option.delta <= option.delta_up:
                option.delta = option.gamma1 * option.delta
            else:
                option.delta = 1 * option.delta
        else:
            if np.linalg.norm(s) > 1e-6:
                option.delta = option.gamma2 * option.delta

        if flag:
            dd1new = (xkp1 - xk) / np.linalg.norm(xkp1 - xk)
        else:
            dd1new = dd1

        tp1 = dd1new @ dd1
        tp2 = dd1new @ dd2

        dd2star = np.array([[-tp2], [tp1]])
        dd2star = (np.array([dd1, dd2]).T @ dd2star).flatten()

        # Fast Gram-Schmidt projection to avoid massive matrix SVD
        rand_vec = np.random.randn(n)
        dd2new = rand_vec - np.dot(rand_vec, dd1) * dd1
        dd2new = dd2new / np.linalg.norm(dd2new)

        dd1old = dd1
        dd2old = dd2

        dd1 = dd1new
        dd2 = dd2new

        # Perform the interpolation for Q_k^+
        temp_plus_1 = y1 - xkp1
        temp_plus_2 = y2 - xkp1
        temp_plus_3 = y3 - xkp1
        temp_plus_4 = xk - xkp1

        alpha1 = temp_plus_1 @ dd1
        beta1 = temp_plus_1 @ dd2star

        alpha2 = temp_plus_2 @ dd1
        beta2 = temp_plus_2 @ dd2star

        alpha3 = temp_plus_3 @ dd1
        beta3 = temp_plus_3 @ dd2star

        alpha4 = temp_plus_4 @ dd1
        beta4 = temp_plus_4 @ dd2star

        if k < 2:
            xtemp = x0
            fxtemp = f(xtemp)
        else:
            xtemp = xkm1
            fxtemp = fxkm1

        temp_plus_5 = xtemp - xkp1

        alpha5 = temp_plus_5 @ dd1
        beta5 = temp_plus_5 @ dd2star

        y8 = xk + sqrt(2) / 2 * oldDelta * dd2old + sqrt(2) / 2 * oldDelta * dd1old
        temp_plus_8 = y8 - xkp1
        alpha8 = temp_plus_8 @ dd1
        beta8 = temp_plus_8 @ dd2star
        # y8 refers to y4 in the paper

        y6 = xk + oldDelta * dd1old
        temp_plus_6 = y6 - xkp1
        alpha6 = temp_plus_6 @ dd1
        beta6 = temp_plus_6 @ dd2star
        # y6 refers to y5 in the paper

        A1 = np.array([alpha1, alpha1 ** 2, beta1, beta1 ** 2, alpha1 * beta1])
        A2 = np.array([alpha2, alpha2 ** 2, beta2, beta2 ** 2, alpha2 * beta2])
        A3 = np.array([alpha3, alpha3 ** 2, beta3, beta3 ** 2, alpha3 * beta3])
        A4 = np.array([alpha4, alpha4 ** 2, beta4, beta4 ** 2, alpha4 * beta4])
        A5 = np.array([alpha5, alpha5 ** 2, beta5, beta5 ** 2, alpha5 * beta5])
        A6 = np.array([alpha6, alpha6 ** 2, beta6, beta6 ** 2, alpha6 * beta6])
        A8 = np.array([alpha8, alpha8 ** 2, beta8, beta8 ** 2, alpha8 * beta8])

        b1 = fy1 - fxkp1
        b2 = fy2 - fxkp1
        b3 = fy3 - fxkp1
        b4 = fxk - fxkp1
        b5 = fxtemp - fxkp1

        min_indexfordrop = np.argmin([fy1, fy2])

        if np.linalg.det(np.array([A1, A2, A3, A4, A5])) != 0:
            A = np.array([A1, A2, A3, A4, A5])
            b = np.array([[b1], [b2], [b3], [b4], [b5]])
            tempRes = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
        elif np.linalg.det(np.array([A1, A6, A3, A4, A5])) != 0 and min_indexfordrop == 0:

            if 'y6_again' in locals().keys() and (np.linalg.norm(y6 - y6_again) == 0):
                fy6 = fy6_again
            else:
                fy6 = fwp(y6)

            b6 = fy6 - fxkp1
            A = np.array([A1, A6, A3, A4, A5])
            b = np.array([[b1], [b6], [b3], [b4], [b5]])
            tempRes = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
        elif np.linalg.det(np.array([A6, A2, A3, A4, A5])) != 0:

            if 'y6_again' in locals().keys() and np.linalg.norm(y6 - y6_again) == 0:
                fy6 = fy6_again
            else:
                fy6 = fwp(y6)

            b6 = fy6 - fxkp1
            A = np.array([A6, A2, A3, A4, A5])
            b = np.array([[b6], [b2], [b3], [b4], [b5]])
            tempRes = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
        elif np.linalg.det(np.array([A1, A2, A3, A8, A5])) != 0:

            if 'y8_again' in locals().keys() and np.linalg.norm(y8 - y8_again) == 0:
                fy8 = fy8_again
            else:
                fy8 = fwp(y8)

            b8 = fy8 - fxkp1
            A = np.array([A1, A2, A3, A8, A5])
            b = np.array([[b1], [b2], [b3], [b8], [b5]])
            tempRes = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
        else:

            if 'y6_again' in locals().keys() and np.linalg.norm(y6 - y6_again) == 0:
                fy6 = fy6_again
            else:
                fy6 = fwp(y6)

            b6 = fy6 - fxkp1

            if 'y8_again' in locals().keys() and np.linalg.norm(y8 - y8_again) == 0:
                fy8 = fy8_again
            else:
                fy8 = fwp(y8)

            b8 = fy8 - fxkp1
            A = np.array([A1, A2, A3, A8, A6])
            b = np.array([[b1], [b2], [b3], [b8], [b6]])
            tempRes = np.linalg.lstsq(A, b, rcond=None)[0].flatten()

        a_value, b_value, c_value, d_value, e_value = tempRes

        k = k + 1
        xkm1 = xk
        fxkm1 = fxk
        xk = xkp1
