import systems_fun as sf
import numpy as np
import pytest
import findTHeteroclinic as fth

@pytest.fixture
def stdPS():
    return sf.STD_PRECISION

@pytest.fixture
def stdPROX():
    return sf.STD_PROXIMITY

class TestDescribeEqType:
    def test_saddle(self, stdPS):
        assert sf.describeEqType(np.array([-1, 1]), stdPS) == [1, 0, 1, 0, 0]

    def test_stable_node(self, stdPS):
        assert sf.describeEqType(np.array([-1, -1]), stdPS) == [2, 0, 0, 0, 0]

    def test_stable_focus(self, stdPS):
        assert sf.describeEqType(np.array([-1 + 1j, -1 - 1j]), stdPS) == [2, 0, 0, 1, 0]

    def test_unstable_node(self, stdPS):
        assert sf.describeEqType(np.array([+1, +1]), stdPS) == [0, 0, 2, 0, 0]

    def test_unstable_focus(self, stdPS):
        assert sf.describeEqType(np.array([1 + 1j, 1 - 1j]), stdPS) == [0, 0, 2, 0, 1]

    def test_passingTuple(self, stdPS):
        # rewrite as expecting some exception
        with pytest.raises(TypeError):
            sf.describeEqType([1 + 1j, 1 - 1j], stdPS) == [0, 0, 2, 0, 1]
        # assert sf.describeEqType([1+1j, 1-1j])==(0, 0, 2, 0, 1)

    def test_almost_focus(self, stdPS):
        assert sf.describeEqType(np.array([-1e-15 + 1j, -1e-15 - 1j]), stdPS) == [0, 2, 0, 0, 0]

    def test_center(self, stdPS):
        assert sf.describeEqType(np.array([1j, - 1j]), stdPS) == [0, 2, 0, 0, 0]

class TestISComplex:
    def test_complex(self, stdPS):
        assert stdPS.isComplex(1+2j) == 1

    def test_real(self, stdPS):
        assert stdPS.isComplex(1) == 0

    def test_complex_without_real(self, stdPS):
        assert stdPS.isComplex(8j) == 1

@pytest.fixture
def sorted2DSaddle():
    return sf.Equilibrium([0.]*2, [-1., 1.], [[0]*2]*2)

@pytest.fixture
def unsorted2DSaddle():
    return sf.Equilibrium([0.]*2, [+2., -11.], [[0]*2]*2)

@pytest.fixture
def unsrt3DSadFoc():
    return sf.Equilibrium([0.]*3, [-1+1j, +3, -1-1j], [[0]*3]*3)

@pytest.fixture
def unsrt3DSaddle():
    return sf.Equilibrium([0.]*3, [-1, 1, 2], [[0]*3]*3)

class TestLeadingEigenvalues:
    def test_srt2DSadLeadS(self, sorted2DSaddle, stdPS):
        assert sorted2DSaddle.getLeadSEigRe(stdPS) == -1.
    def test_srt2DSadLeadU(self, sorted2DSaddle, stdPS):
        assert sorted2DSaddle.getLeadUEigRe(stdPS) == +1.
    def test_unsrt2DSadLeadS(self, unsorted2DSaddle, stdPS):
        assert unsorted2DSaddle.getLeadSEigRe(stdPS) == -11.
    def test_unsrt2DSadLeadU(self, unsorted2DSaddle, stdPS):
        assert unsorted2DSaddle.getLeadUEigRe(stdPS) == +2.
    def test_unsrt3DSadFocLeadS(self, unsrt3DSadFoc, stdPS):
        assert unsrt3DSadFoc.getLeadSEigRe(stdPS) == -1.
    def test_unsrt3DSadFocLeadU(self, unsrt3DSadFoc, stdPS):
        assert unsrt3DSadFoc.getLeadUEigRe(stdPS) == +3.
    def test_unsrt3DSaddleLeadS(self, unsrt3DSaddle, stdPS):
        assert unsrt3DSaddle.getLeadSEigRe(stdPS) == -1
    def test_unsrt3DSaddleLeadU(self, unsrt3DSaddle, stdPS):
        assert unsrt3DSaddle.getLeadUEigRe(stdPS) == +1


class TestInBounds:
    def test_inBounds(self):
        assert sf.inBounds((1,1),[(0,2),(0,2)]) == 1

    def test_onBounds(self):
        assert sf.inBounds((1,1),[(1,2),(0,2)]) == 0

    def test_outBounds(self):
        assert sf.inBounds((5,5),[(1,1),(0,6)]) == 0

    def test_outBoundForAction(self):
        assert sf.inBounds((5, 6), [(1, 1), (0, 6)]) == 0

class TestFindEquilibria:
    def rhs(self,X,params):
        x,y=X
        a,b,c1,c2=params
        return [x*(y-a*x), y*(x+ b+ y)]

    def rhsJac(self,X, params):
        x, y = X
        a, b, c1, c2 = params
        return np.array([[y - 2 * a * x, x], [y, b + x + 2 * y]])



    def analyticFind(self,params):
        a,b,c1,c2=params
        #nSinks, nSaddles, nSources, nNonRough
        if ((a>0 and b>0 )  or (a<-1 and b>0 )):
            result = (1,1,0,1)
        elif((a>0 and b<0 )or(a<-1 and b<0) ):
            result = (0,1,1,1)
        elif(-1<a<0 and b>0):
            result = (1, 0, 0, 2)
        elif (-1 < a < 0 and b > 0):
            result = (0, 0, 1, 2)
        elif(a == -1 and b>0):
            result= (0)
        elif (a == -1 and b < 0):
            result = (0)
        elif (a == -1 and b == 0):
            result = (0,0,0,1)
        elif (a < -1 and b == 0 or-1< a < 0 and b == 0 or a > 0 and b == 0):
            result = (0,0,0,1)
        elif (-1< a < 0 and b == 0):
            result = (0,0,0,1)
        return result

    bounds = [(-3.5, 3.5), (-3.5, 3.5)]
    borders = [(-3.5 , 3.5 ), (-3.5 , 3.5)]

    def test_FindEqInSinglePoint(self, stdPS):
        ud = [-0.5,0,0,0]

        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.ShgoEqFinder(300, 30, 1e-10), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)


    def test_FindEqInSinglePoint2(self, stdPS):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.ShgoEqFinder(1000, 100,1e-15), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePoint3(self, stdPS):
        ud = [-1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.ShgoEqFinder(300, 10,4e-14), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

    def test_FindEq2InSinglePointNewton(self, stdPS):
        ud = [-0.5,0,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.NewtonEqFinder(21, 21,1e-15), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewton2(self, stdPS):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.NewtonEqFinder(61, 61,1e-18), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewton3(self, stdPS):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.NewtonEqFinder(61, 61,1e-18), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)


    def test_FindEqInSinglePointNewtonUp(self, stdPS):
        ud = [-0.5,0,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.NewtonEqFinderUp(81, 81,1e-20), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewtonUp2(self, stdPS):
        ud = [1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.NewtonEqFinderUp(71, 71,1e-16), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

    def test_FindEqInSinglePointNewtonUp3(self, stdPS):
        ud = [-1.5,0.5,0,0]
        rhsCurrent = lambda X: self.rhs(X, ud)
        rhsJacCurrent = lambda X: self.rhsJac(X, ud)
        res = sf.findEquilibria(rhsCurrent, rhsJacCurrent, self.bounds, self.borders,
                                 sf.NewtonEqFinderUp(101, 101,8e-17), stdPS)
        data = []
        for eq in res:
            data.append(eq.getEqType(stdPS)[0:3])
        describe = sf.describePortrType(data)
        assert describe == self.analyticFind(ud)

@pytest.fixture
def duffingSetup():
    class Duffing:
        def __init__(self):
            pass

        def rhs(self, X):
            x, y = X
            return [y, -x * (1 - x * x)]

        def rhsJac(self, X):
            x, y = X
            return [[0, 1], [-1 + 3 * x * x, 0]]

    fstCds = (-1, 0)
    sndCds = (+1, 0)
    ob = Duffing()
    fstEq = sf.getEquilibriumInfo(fstCds, ob.rhsJac)
    sndEq = sf.getEquilibriumInfo(sndCds, ob.rhsJac)
    pairsToCheck = [(fstEq, sndEq)]

    def rightSep(ptCoord, eqCoord):
        return ptCoord[0] > eqCoord[0]

    return (ob, rightSep, pairsToCheck)

def test_Duffing(duffingSetup):
    ob, rightSep, pairsToCheck = duffingSetup
    out = fth.checkSeparatrixConnection(pairsToCheck, fth.sf.STD_PRECISION, fth.sf.STD_PROXIMITY, ob.rhs, ob.rhsJac,
                              sf.idTransform, rightSep, sf.idListTransform, sf.hasExactly(1), 1e-5, 1000.)
    assert out[0]['dist'] < 1e-5