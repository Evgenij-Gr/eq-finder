from MySystem import *
import TwoPendulumsSystemFun as tpsf
from systems_fun import constructDistEvent
from matplotlib.backends.backend_pdf import PdfPages

paramK = 0.06
int_time_1 = 200
int_time_2 = 30000
tSkip = 20

def poincareSection_fi_4(t, X):
    fi_1, V_1, fi_2, V_2 = X
    new_fi_2 = tpsf.toStandartAngle(fi_2)
    return new_fi_2 - 4.0

def poincareSection_fi_0_75(t, X):
    fi_1, V_1, fi_2, V_2 = X
    new_fi_1 = tpsf.toStandartAngle(fi_1)
    return new_fi_1 - 0.45


def create_phase_portrait():
    Dist, Lambda, Gamma = min_dist_dih()
    newLambda = Lambda
    newGamma = Gamma
    Sys = TwoPendulums(newGamma, newLambda, paramK)
    rhs = Sys.FullSystem

    boundsType = [(-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1)]
    bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]
    jacType = Sys.JacType
    rhsType = Sys.ReducedSystem
    rhsJac = Sys.Jac
    Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
                           sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]

    isLeadingStable2d = (newEq[0].getEqType(sf.STD_PRECISION)[3] == 1)
    print(isLeadingStable2d)
    rhs_vec = lambda t, X: Sys.FullSystem(X)

    startPt_1 = sf.getInitPointsOnUnstable1DSeparatrix(newEq[0], sf.pickBothSeparatrices, sf.STD_PRECISION)
    # print(startPt_1[1])
    startPt_1[1] = [0.571702507, 0.0, 2.74005986, 0.0]
    # new_start_pt = startPt_1[0] + [0.1, 0., 0.04, 0.]
    # print(new_start_pt)
    coords = newEq[0].coordinates
    # event = constructDistEvent(coords, sf.STD_PROXIMITY.toTargetSddlPrxtyEv, tpsf.periodDistance4D)
    # event = poincareSection_fi_4
    event = poincareSection_fi_0_75
    event.terminal = False
    event.direction = 1
    # sep1 = solve_ivp(rhs_vec, [0, int_time_1], startPt_1[0], rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True, method='DOP853')
    # fi1_1 = tpsf.normalize_fi_vec(sep1.y[0])
    # V1_1 = sep1.y[1]
    # fi2_1 = tpsf.normalize_fi_vec(sep1.y[2])
    # V2_1 = sep1.y[3]
    sep2 = solve_ivp(rhs_vec, [0, int_time_2], startPt_1[1], rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True, events=event, method='DOP853')
    fi1_2 = tpsf.normalize_fi_vec(sep2.y[0])
    V1_2 = sep2.y[1]
    fi2_2 = tpsf.normalize_fi_vec(sep2.y[2])
    V2_2 = sep2.y[3]

    # tr_2 = solve_ivp(rhs_vec, [0, int_time_2], new_start_pt, rtol=sf.STD_PRECISION.rTol, atol=sf.STD_PRECISION.aTol, dense_output=True, events=event, method='DOP853', max_step=0.1)
    # fi1_2_n = tpsf.normalize_fi_vec(tr_2.y[0])
    # V1_2_n = tr_2.y[1]
    # fi2_2_n = tpsf.normalize_fi_vec(tr_2.y[2])
    # V2_2_n = tr_2.y[3]
    # print(len(fi1_2_n))
    evt_fi_1 = tpsf.normalize_fi_vec(sep2.y_events[0][:, 0])
    evt_V_1 = sep2.y_events[0][:, 1]
    evt_fi_2 = tpsf.normalize_fi_vec(sep2.y_events[0][:, 2])
    evt_V_2 = sep2.y_events[0][:, 3]
    print(len(evt_fi_1))
    # fig = plt.figure(figsize=(8, 4))
    #
    # ax1 = fig.add_subplot(121, projection='3d')
    # # plt.title('3D-projection $\phi_1, \phi_2$')
    # ax1.scatter(fi1_1, fi2_1, V1_1, color='b', s=0.25)
    # ax1.scatter(fi1_2, fi2_2, V1_2, color='r', s=0.25)
    # ax1.scatter(Eq[0].coordinates[0], Eq[0].coordinates[2], Eq[0].coordinates[1], c='green', s=20, label='StF')
    # ax1.scatter(Eq[1].coordinates[0], Eq[1].coordinates[2], Eq[1].coordinates[1], c='red', s=20, label='S-F')
    # ax1.scatter(Eq[3].coordinates[0], Eq[3].coordinates[2], Eq[3].coordinates[1], c='red', s=20)
    # ax1.scatter(Eq[2].coordinates[0], Eq[2].coordinates[2], Eq[2].coordinates[1], c='blue', s=20, label='S')
    # ax1.set_xlim([0, 2*np.pi])
    # ax1.set_ylim([0, 2*np.pi])
    # ax1.set_xlabel('$\phi_1$')
    # ax1.set_ylabel('$\phi_2$')
    # ax1.set_zlabel('V1')
    # plt.grid(True)
    #
    #
    # ax2 = fig.add_subplot(122, projection='3d')
    # # plt.title('3D-projection $V_1, V_2$')
    # ax2.scatter(fi1_1, fi2_1, V2_1, color='b', s=0.25)
    # ax2.scatter(fi1_2, fi2_2, V2_2, color='r', s=0.25)
    # ax2.scatter(Eq[0].coordinates[0], Eq[0].coordinates[2], Eq[0].coordinates[3], c='green', s=20, label='StF')
    # ax2.scatter(Eq[1].coordinates[0], Eq[1].coordinates[2], Eq[1].coordinates[3], c='red', s=20, label='S-F')
    # ax2.scatter(Eq[3].coordinates[0], Eq[3].coordinates[2], Eq[3].coordinates[3], c='red', s=20)
    # ax2.scatter(Eq[2].coordinates[0], Eq[2].coordinates[2], Eq[2].coordinates[3], c='blue', s=20, label='S')
    # ax2.set_xlim([0, 2*np.pi])
    # ax2.set_ylim([0, 2*np.pi])
    # ax2.set_xlabel('$\phi_1$')
    # ax2.set_ylabel('$\phi_2$')
    # ax2.set_zlabel('V2')
    # plt.grid(True)
    #
    # # pathToDir = 'C:/Users/User/eq-finder/output_files/TwoPendulums/Фазовые портреты сепаратрисы/Гомокиника(k=0.06)'
    # # imageName = f"{Gamma = }, {Lambda= }" + ' projection ' + f"{int_time_1 = }, {int_time_2= }"
    # # fullOutputName = os.path.join(pathToDir, imageName + '.png')
    # # plt.savefig(fullOutputName, dpi=300)
    # plt.show()


    plt.suptitle('$\phi_1(t), \phi_2(t)$')

    plt.subplot(211)
    # plt.scatter(sep1.t, fi1_1, color='b', s=0.25)
    plt.scatter(sep2.t, fi1_2, color='r', s=0.25)
    # plt.scatter(tr_2.t, fi1_2_n, color='k', s=0.25)
    plt.ylabel('$\phi_1$')
    plt.xlabel('t')
    plt.grid(True, which='both')
    # plt.ylim(0.79, 0.825)

    plt.subplot(212)
    # plt.scatter(sep1.t, fi2_1, color='b', s=0.25)
    plt.scatter(sep2.t, fi2_2, color='r', s=0.25)
    # plt.scatter(tr_2.t, fi2_2_n, color='k', s=0.25)
    plt.ylabel('$\phi_2$')
    plt.xlabel('t')
    plt.grid(True, which='both')
    plt.show()
    # pdf = PdfPages("Figures.pdf")
    # pdf.savefig()
    # pdf.close()


    plt.suptitle('$\dot{\phi}_1, \dot{\phi}_2$')

    plt.subplot(211)
    # # Неустойчивая сепаратриса
    # # plt.plot(fi1_1, V1_1, color='b')
    # # plt.plot(fi1_2, V1_2, color='r')
    # # plt.plot(fi1_2_n[150000:], V1_2_n[150000:], color='k')
    # plt.ylabel('$\dot{\phi}_1$')
    # plt.xlabel('$\phi_1$')

    # Для секущей fi_2 = 4
    # plt.scatter(evt_V_1, evt_V_2, c='k', s=0.75)
    # plt.ylabel('$\dot{\phi}_2$')
    # plt.xlabel('$\dot{\phi}_1$')

    # Для секущей fi_1 = 0.75
    plt.scatter(evt_V_1[2600:], evt_V_2[2600:], c='k', s=1)
    # plt.scatter(evt_fi_1, evt_V_1, c='k', s=1)
    plt.ylabel('$\dot{\phi}_2$')
    plt.xlabel('$\dot{\phi}_1$')
    plt.grid(True, which='both')

    plt.subplot(212)
    # # Неустойчивая сепаратриса
    # # plt.scatter(fi2_1, V2_1, color='b', s=0.25)
    # # plt.scatter(fi2_2, V2_2, color='r', s=0.25)
    # plt.scatter(fi2_2_n[150000:], V2_2_n[150000:], color='k', s=0.25)
    # plt.ylabel('$\dot{\phi}_2$')
    # plt.xlabel('$\phi_2$')

    # Для секущей fi_2 = 4
    # plt.scatter(evt_fi_1, evt_V_1, c='k', s=0.75)
    # plt.ylabel('$\dot{\phi}_1$')
    # plt.xlabel('$\phi_1$')

    # Для секущей fi_1 = 0.75
    plt.scatter(evt_fi_2[2600:], evt_V_2[2600:], c='k', s=1)
    plt.ylabel('$\dot{\phi}_2$')
    plt.xlabel('$\phi_2$')
    plt.grid(True, which='both')

    plt.show()
    # return Lambda
    return Dist, Lambda, Gamma


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
                                            sf.anyNumber, 1, 1000., tpsf.periodDistance4D, tSkip, listEqCoords=None)

    dist_min = min(i['dist'] for i in cnctInfo)
    print(f'{dist_min = }')
    return dist_min


def min_dist_dih():
    # Lambda = 0.55
    # Gamma = 0.6664278241282653

    # Lambda = 0.545
    # Gamma = 0.6614030623209723

    Lambda = 0.36
    Gamma = 0.4580837959705413

    # Lambda = 0.64
    # Gamma = 0.75183725

    # Lambda = 0.3
    # Gamma = 4./np.pi*Lambda - 0.305*Lambda**3
    # Gamma = 0.3859494247853643

    print(f'{Gamma = }')
    a = Gamma - 2e-16
    b = Gamma + 2e-16
    print(f'{a = }, {b = }')
    dist_min = 10.
    count = 0
    c = (b + a) / 2
    dist_c = get_dist(c, Lambda, paramK)
    while dist_min > 5e-3:
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