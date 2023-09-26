import numpy as np
from MySystem import *
import os
import TwoPendulumsSystemFun as tpsf
from systems_fun import constructDistEvent


# Gamma = 0.3
paramK = 0.06

# lambda_data = np.linspace(0, 1.22, 1000)
# gamma_data = []
# x = [0, 1.5]
# y = [1, 1]
# for i in range(len(lambda_data)):
#     gamma_data.append(4./np.pi*lambda_data[i] - 0.305*lambda_data[i]**3)
# plt.plot(lambda_data, gamma_data, c='k')
# plt.plot(x, y, c='k')
# plt.xlim(0, 1.5)
# plt.ylim(0, 1.5)
# plt.show()


def create_phase_portrait():
    Dist, Lambda, Gamma = min_dist_dih()
    # Dist = 1
    # Lambda = 0.23062899646181
    # Lambda = 0.3
    Sys = TwoPendulums(Gamma, Lambda, paramK)
    rhs = Sys.FullSystem

    boundsType = [(-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1)]
    bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]
    jacType = Sys.JacType
    rhsType = Sys.ReducedSystem
    rhsJac = Sys.Jac
    Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
                           sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]


    rhs_vec = lambda t, X: Sys.FullSystem(X)

    startPt_1 = sf.getInitPointsOnUnstable1DSeparatrix(newEq[0], sf.pickBothSeparatrices, sf.STD_PRECISION)

    int_time_1 = 200
    int_time_2 = 200

    coords = newEq[0].coordinates
    event = constructDistEvent(coords, sf.STD_PROXIMITY.toTargetSddlPrxtyEv, tpsf.periodDistance4D)
    event.terminal = False
    event.direction = -1
    sep1 = solve_ivp(rhs_vec, [0, int_time_1], startPt_1[0], rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True)
    fi1_1 = tpsf.normalize_fi_vec(sep1.y[0])
    # print(len(fi1_1))
    V1_1 = sep1.y[1]
    fi2_1 = tpsf.normalize_fi_vec(sep1.y[2])
    V2_1 = sep1.y[3]
    sep2 = solve_ivp(rhs_vec, [0, int_time_2], startPt_1[1], rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True, events=event)
    fi1_2 = tpsf.normalize_fi_vec(sep2.y[0])
    V1_2 = sep2.y[1]
    fi2_2 = tpsf.normalize_fi_vec(sep2.y[2])
    V2_2 = sep2.y[3]

    fig = plt.figure(figsize=(8, 4))

    ax1 = fig.add_subplot(121, projection='3d')
    # plt.title('3D-projection $\phi_1, \phi_2$')
    ax1.scatter(fi1_1, fi2_1, V1_1, color='k', s=0.25)
    ax1.scatter(fi1_2, fi2_2, V1_2, color='k', s=0.25)
    ax1.scatter(Eq[0].coordinates[0], Eq[0].coordinates[2], Eq[0].coordinates[1], c='green', s=20, label='StF')
    ax1.scatter(Eq[1].coordinates[0], Eq[1].coordinates[2], Eq[1].coordinates[1], c='red', s=20, label='S-F')
    ax1.scatter(Eq[3].coordinates[0], Eq[3].coordinates[2], Eq[3].coordinates[1], c='red', s=20)
    ax1.scatter(Eq[2].coordinates[0], Eq[2].coordinates[2], Eq[2].coordinates[1], c='blue', s=20, label='S')
    ax1.set_xlim([0, 2*np.pi])
    ax1.set_ylim([0, 2*np.pi])
    ax1.set_xlabel('$\phi_1$')
    ax1.set_ylabel('$\phi_2$')
    ax1.set_zlabel('V1')
    plt.grid(True)
    plt.legend()


    ax2 = fig.add_subplot(122, projection='3d')
    # plt.title('3D-projection $V_1, V_2$')
    ax2.scatter(fi1_1, fi2_1, V2_1, color='k', s=0.25)
    ax2.scatter(fi1_2, fi2_2, V2_2, color='k', s=0.25)
    ax2.scatter(Eq[0].coordinates[0], Eq[0].coordinates[2], Eq[0].coordinates[3], c='green', s=20, label='StF')
    ax2.scatter(Eq[1].coordinates[0], Eq[1].coordinates[2], Eq[1].coordinates[3], c='red', s=20, label='S-F')
    ax2.scatter(Eq[3].coordinates[0], Eq[3].coordinates[2], Eq[3].coordinates[3], c='red', s=20)
    ax2.scatter(Eq[2].coordinates[0], Eq[2].coordinates[2], Eq[2].coordinates[3], c='blue', s=20, label='S')
    ax2.set_xlim([0, 2*np.pi])
    ax2.set_ylim([0, 2*np.pi])
    ax2.set_xlabel('$\phi_1$')
    ax2.set_ylabel('$\phi_2$')
    ax2.set_zlabel('V2')
    plt.grid(True)

    # pathToDir = 'C:/Users/User/eq-finder/output_files/TwoPendulums/Фазовые портреты сепаратрисы/Гомокиника(k=0.06)'
    # imageName = f"{Gamma = }, {Lambda= }" + ' projection ' + f"{int_time_1 = }, {int_time_2= }"
    # fullOutputName = os.path.join(pathToDir, imageName + '.png')
    # plt.savefig(fullOutputName, dpi=300)
    plt.show()


    plt.suptitle('$\phi_1(t), \phi_2(t)$')

    plt.subplot(221)
    plt.scatter(sep1.t, fi1_1, color='k', s=0.25)
    plt.ylabel('$\phi_1$')
    plt.xlabel('t')
    plt.grid(True, which='both')

    plt.subplot(222)
    plt.scatter(sep1.t, fi2_1, color='k', s=0.25)
    plt.ylabel('$\phi_2$')
    plt.xlabel('t')
    plt.grid(True, which='both')

    t_ = int(len(sep2.t)*0.45)
    print(len(fi1_2))
    print(t_)
    # t_=50
    plt.subplot(223)
    # plt.scatter(sep2.t[:t_], fi1_2[:t_], color='k', s=0.25)
    plt.scatter(sep2.t, fi1_2, color='k', s=0.25)
    plt.ylabel('$\phi_1$')
    plt.xlabel('t')
    plt.grid(True, which='both')

    plt.subplot(224)
    # plt.scatter(sep2.t[:t_], fi2_2[:t_], color='k', s=0.25)
    plt.scatter(sep2.t, fi2_2, color='k', s=0.25)
    plt.ylabel('$\phi_2$')
    plt.xlabel('t')
    plt.grid(True, which='both')
    plt.show()


    # pathToDir = 'C:/Users/User/eq-finder/output_files/TwoPendulums/Фазовые портреты сепаратрисы/Гомокиника(k=0.06)'
    # imageName = f"{Gamma = }, {Lambda= }" + ' fi(t) ' + f"{int_time_1 = }, {int_time_2= }"
    # fullOutputName = os.path.join(pathToDir, imageName + '.png')
    # plt.savefig(fullOutputName, dpi=300)

    plt.suptitle('$\dot{\phi_1}, \dot{\phi_2}$')

    plt.subplot(221)
    plt.scatter(fi1_1, V1_1, color='k', s=0.25)
    plt.ylabel('$\dot{\phi_1}$')
    plt.xlabel('$\phi_1$')
    plt.grid(True, which='both')

    plt.subplot(222)
    plt.scatter(fi2_1, V2_1, color='k', s=0.25)
    plt.ylabel('$\dot{\phi_2}$')
    plt.xlabel('$\phi_2$')
    plt.grid(True, which='both')

    plt.subplot(223)
    # plt.scatter(fi1_2[:t_], V1_2[:t_], color='k', s=0.25)
    plt.scatter(fi1_2, V1_2, color='k', s=0.25)
    plt.ylabel('$\dot{\phi_1}$')
    plt.xlabel('$\phi_1$')
    plt.grid(True, which='both')

    plt.subplot(224)
    # plt.scatter(fi2_2[:t_], V2_2[:t_], color='k', s=0.25)
    plt.scatter(fi2_2, V2_2, color='k', s=0.25)
    plt.ylabel('$\dot{\phi_2}$')
    plt.xlabel('$\phi_2$')
    plt.grid(True, which='both')

    plt.show()
    # return Lambda
    return Dist, Lambda, Gamma


def lambda_Min(lambda_min):
    if lambda_min <= 0.1:
        return 0.1
    else:
        return lambda_min


def lambda_Max(lambda_max):
    if lambda_max >= 1.:
        return 0.99
    else:
        return lambda_max


def min_dist():
    dist_min = 10.
    lambda_min = 0.1
    degree = 1
    # dist_min = 0.010200431756804303, lambda = 0.08896978
    while dist_min > 1e-2:
        for j in np.arange(lambda_Min(lambda_min - pow(0.1, degree-1)), lambda_Max(lambda_min + pow(0.1, degree-1)), pow(0.1, degree)):
            j = round(j, degree)
            print(j)
            Lambda = j

            Sys = TwoPendulums(Gamma, Lambda, paramK)
            rhs = Sys.FullSystem

            boundsType = [(-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1)]
            bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]
            jacType = Sys.JacType
            rhsType = Sys.ReducedSystem
            rhsJac = Sys.Jac
            Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
                                   sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

            newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]
            # for eq in newEq:
            #     print(eq.coordinates)
            pairs_to_check = [[newEq[0], newEq[0]]]

            cnctInfo = FH.checkSeparatrixConnection(pairs_to_check, sf.STD_PRECISION, sf.STD_PROXIMITY, rhs,
                                                    jacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
                                                    sf.anyNumber, 1, 500., tpsf.periodDistance4D, listEqCoords=None)

            for i in cnctInfo:
                # print(i['dist'])
                if i['dist'] < dist_min:
                    dist_min = i['dist']
                    print(f'{dist_min = }, lambda = {j}')
                    lambda_min = j

        degree += 1
    return dist_min, lambda_min

def get_dist(gamma_, lambda_, paramK):
    Sys = TwoPendulums(gamma_, lambda_, paramK)
    rhs = Sys.FullSystem

    boundsType = [(-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1)]
    bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]
    jacType = Sys.JacType
    rhsType = Sys.ReducedSystem
    rhsJac = Sys.Jac
    Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
                           sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]
    pairs_to_check = [[newEq[0], newEq[0]]]

    cnctInfo = FH.checkSeparatrixConnection(pairs_to_check, sf.STD_PRECISION, sf.STD_PROXIMITY, rhs,
                                            jacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
                                            sf.anyNumber, 1, 500., tpsf.periodDistance4D, listEqCoords=None)

    dist_min = min(i['dist'] for i in cnctInfo)
    print(f'{dist_min = }')
    return dist_min


def min_dist_dih():
    Lambda = 0.45
    # paramK = 0.06
    Gamma = 4./np.pi*Lambda - 0.305*Lambda**3
    # Gamma = 0.560848
    print(f'{Gamma = }')
    a = Gamma - 0.02
    b = Gamma + 0.02
    print(f'{a = }, {b = }')
    dist_min = 10.
    count = 0
    c = (b + a) / 2
    dist_c = get_dist(c, Lambda, paramK)
    while dist_min > 1e-3:
        if b-a < 1e-14:
            break
        print(f'{count = }')
        x = (a+c)/2
        y = (c+b)/2
        dist_x = get_dist(x, Lambda, paramK)
        dist_y = get_dist(y, Lambda, paramK)
        if (dist_x <= dist_c) and (dist_c < dist_y):
            b = c
            c = x
            dist_c = dist_x
        elif (dist_x > dist_c) and (dist_c <= dist_y):
            a = x
            b = y
        else:
            a = c
            c = y
            dist_c = dist_y
        count+=1
        dist_min = min(dist_x, dist_y, dist_c)
    gamma_min = (a+b)/2
    return dist_min, Lambda, gamma_min

if __name__ == "__main__":
    dist, Lambda, Gamma = create_phase_portrait()
    # lam = create_phase_portrait()
    print(f'dist between saddle and saparatrix = {dist} at lambda = {Lambda} and gamma = {Gamma}')


# dist between saddle and saparatrix = 0.00935076622817968 at lambda = 0.19113265120448