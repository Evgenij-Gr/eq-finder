import numpy as np
from MySystem import *
import os
import TwoPendulumsSystemFun as tpsf


Gamma = 0.3
paramK = 0.06


def create_phase_portrait():
    Dist, Lambda = min_dist()
    # Dist = 1
    # Lambda = 0.23062899646181
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

    int_time_1 = 1
    int_time_2 = 60

    sep1 = solve_ivp(rhs_vec, [0, int_time_1], startPt_1[0], rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True)
    fi1_1 = tpsf.normalize_fi_vec(sep1.y[0])
    # print(len(fi1_1))
    V1_1 = sep1.y[1]
    fi2_1 = tpsf.normalize_fi_vec(sep1.y[2])
    V2_1 = sep1.y[3]
    sep2 = solve_ivp(rhs_vec, [0, int_time_2], startPt_1[1], rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True)
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

    plt.subplot(223)
    plt.scatter(sep2.t, fi1_2, color='k', s=0.25)
    plt.ylabel('$\phi_1$')
    plt.xlabel('t')
    plt.grid(True, which='both')

    plt.subplot(224)
    plt.scatter(sep2.t, fi2_2, color='k', s=0.25)
    plt.ylabel('$\phi_2$')
    plt.xlabel('t')
    plt.grid(True, which='both')

    # pathToDir = 'C:/Users/User/eq-finder/output_files/TwoPendulums/Фазовые портреты сепаратрисы/Гомокиника(k=0.06)'
    # imageName = f"{Gamma = }, {Lambda= }" + ' fi(t) ' + f"{int_time_1 = }, {int_time_2= }"
    # fullOutputName = os.path.join(pathToDir, imageName + '.png')
    # plt.savefig(fullOutputName, dpi=300)
    plt.show()
    # return Lambda
    return Dist, Lambda


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
                                                    sf.anyNumber, 1, 500., tpsf.periodDistance4D, listEqCoords=[newEq[0]])

            for i in cnctInfo:
                # print(i['dist'])
                if i['dist'] < dist_min:
                    dist_min = i['dist']
                    print(f'{dist_min = }, lambda = {j}')
                    lambda_min = j

        degree += 1
    return dist_min, lambda_min


if __name__ == "__main__":
    dist, lam = create_phase_portrait()
    # lam = create_phase_portrait()
    print(f'dist between saddle and saparatrix = {dist} at lambda = {lam}')


# dist between saddle and saparatrix = 0.00935076622817968 at lambda = 0.19113265120448