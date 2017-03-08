import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# All constants in cgs
SB = 5.670373e-5
kB = 1.3806488e-16
c = 2.99792458e10
mp = 1.672621777e-24
me = 9.10938291e-28
ka_es = 0.4  #Fully ionized
RsSolar = 2.953250077e5
MSolar = 1.9885e33
G = 6.6738e-8
Year = 3.15569252e7


GMSolar = 0.5*RsSolar*c*c

def pressureGas(rho, T, pars):
    return rho*kB*T/(mu*mp)

def pressureRad(rho, T, pars):
    return 4*SB*T*T*T*T/(3*c)

def pressure(rho, T, pars, deriv=False):
    mu = pars[0]
    Pgas = rho*kB*T/(mu*mp)
    Prad = 4*SB*T*T*T*T/(3*c)
    P = Pgas + Prad
    if deriv:
        dPdrho = Pgas/rho
        dPdT = (Pgas + 4*Prad)/T
        return P, dPdrho, dPdT
    return P

def opacityES(rho, T, pars):
    return 0.4 * np.ones(rho.shape) * np.ones(T.shape)

def opacityFF(rho, T, pars):
    return 5.0e24 * rho * np.power(T,-3.5)

def opacity(rho, T, pars, deriv=False):
    ka_es = 0.4
    ka_ff = 5.0e24 * rho * np.power(T, -3.5)
    ka = ka_es + ka_ff
    if deriv:
        dkadrho = ka_ff/rho
        dkadT = -3.5*ka_ff/T
        return ka, dkadrho, dkadT
    return ka

def calc_thin_disk(r, GM, Mdot, Jdot, al, mu):

    N = len(r)
    TOL = 1.0e-10

    omK = np.sqrt(GM/(r*r*r))
    Mdotf = Mdot - Jdot/(r*r*omK)
    Q = 3.0/(8*np.pi) * Mdotf * omK*omK
    Om = omK

    pars = (mu,)

    Cp = np.power(8*Q/(9*al), 2.0/3.0)
    Cka = 4*SB/3.0 * math.pow(9.0*al/8.0,1.0/3.0) * np.power(Q,-4.0/3) * omK

    lCp = np.log(Cp)
    lCk = np.log(Cka)

    Sig1, rho1, Pi1, P1, vr1, Q1, Om1, ka1, H1, T1 = calc_gas_es(
                                                    r, GM, Mdot, Jdot, al, mu)
    rhoArr = np.empty(r.shape)
    TArr = np.empty(r.shape)

    for j in np.arange(len(r))[::-1]:

        if j == len(r)-1:
            lr = math.log(rho1[-1])
            lT = math.log(T1[-1])
            dr = np.inf
            dT = np.inf
        else:
            lr = math.log(rhoArr[j+1])
            lT = math.log(TArr[j+1])

        lCp = math.log(Cp[j])
        lCk = math.log(Cka[j])

        i = 0
        while True:
            rho = math.exp(lr)
            T = math.exp(lT)

            P, dPdr, dPdT = pressure(rho, T, pars, True)
            ka, dkadr, dkadT = opacity(rho, T, pars, True)

            lP = math.log(P)
            lk = math.log(ka)
            dlPdlr = rho*dPdr/P
            dlPdlT = T*dPdT/P
            dlkdlr = rho*dkadr/ka
            dlkdlT = T*dkadT/ka

            lP0 = lCp + lr/3.0
            lk0 = lCk - 2.0*lr/3.0 + 4*lT

            fa = lP - lP0
            fb = lk - lk0
            dfadr = dlPdlr - 1.0/3.0
            dfadT = dlPdlT
            dfbdr = dlkdlr + 2.0/3.0
            dfbdT = dlkdlT - 4.0
            det = dfadr*dfbdT - dfadT*dfbdr

            dr0 = dr
            dT0 = dT
            dr = -( dfbdT*fa - dfadT*fb) / det
            dT = -(-dfbdr*fa + dfadr*fb) / det

            deg = np.fabs(dr/dr0+1) + np.fabs(dT/dT0+1)

            if False:
                print("   {0:d}: ({1:.2g},{2:.2g}) {3:.2g} {4:.2g} ({5:.2g} {6:.2g} {7:.2g} {8:.2g})  ({9:.2g} {10:.2g}) {11:.2g}".format(
                    i, rho, T, fa, fb, 
                    dfadr, dfadT, dfbdr, dfbdT,
                    dr, dT, deg))

            damp = 1.0
            if deg < 1.0e-3:
                damp = 0.7

            lr += damp*dr
            lT += damp*dT

            i += 1

            if math.sqrt(dr*dr + dT*dT) < TOL or i > 100:
                break

        rhoArr[j] = np.exp(lr)
        TArr[j] = np.exp(lT)

    rho = rhoArr
    T = TArr

    P = pressure(rho, T, pars)
    ka = opacity(rho, T, pars)

    H = np.sqrt(P/rho) / omK
    Sig = rho*H
    Pi = P*H
    vr = -Mdot / (2*np.pi*r*Sig)

    return Sig, rho, Pi, P, vr, Q, Om, ka, H, T

def calc_gas_es(r, GM, Mdot, Jdot, al, mu):

    omK = np.sqrt(GM/(r*r*r))
    Mdotf = Mdot - Jdot/(r*r*omK)
    Q = 3.0/(8*np.pi) * Mdotf * omK*omK

    SigNu = Mdotf/(3*np.pi)
    Pi = SigNu * omK / al
    Om = omK

    Sig = np.power( 4*SB/(3*ka_es) * np.power(mu*mp*Pi/kB,4)/Q, 0.2)
    T = mu*mp*Pi/(kB*Sig)

    H = np.sqrt(Pi/Sig) / omK
    rho = Sig/H
    P = Pi/H

    vr = -Mdot / (2*np.pi*r*Sig)

    ka = ka_es * np.ones(r.shape)

    return Sig, rho, Pi, P, vr, Q, Om, ka, H, T

def calc_rad_es(r, GM, Mdot, Jdot, al, mu):

    omK = np.sqrt(GM/(r*r*r))
    Mdotf = Mdot - Jdot/(r*r*omK)
    Q = 3.0/(8*np.pi) * Mdotf * omK*omK

    SigNu = Mdotf/(3*np.pi)
    Pi = SigNu * omK / al
    Om = omK

    H = ka_es/c * Q / (omK*omK)
    Sig = Pi / (H*H*omK*omK)

    P = Pi/H
    rho = Sig/H

    T = np.power(3*c/(4*SB)*P, 0.25)

    vr = -Mdot / (2*np.pi*r*Sig)

    ka = ka_es * np.ones(r.shape)

    return Sig, rho, Pi, P, vr, Q, Om, ka, H, T

def calc_gas_ff(r, GM, Mdot, Jdot, al, mu):

    omK = np.sqrt(GM/(r*r*r))
    Mdotf = Mdot - Jdot/(r*r*omK)
    Q = 3.0/(8*np.pi) * Mdotf * omK*omK

    SigNu = Mdotf/(3*np.pi)
    Pi = SigNu * omK / al
    Om = omK

    psi0 = 5.0e24

    Q1fac = 4*SB/(3*psi0)*math.sqrt(kB/(mu*mp))/omK
    Q2fac = 9.0/8.0*al*omK*(kB/(mu*mp))
    T = np.power(Q*Q*Q / (Q1fac*Q2fac*Q2fac), 0.1)

    H = np.sqrt(kB*T/(mu*mp)) / omK
    P = Pi / H
    rho = mu*mp * P / (kB*T)
    Sig = rho * H

    vr = -Mdot / (2*np.pi*r*Sig)

    ka = psi0 * rho * np.power(T, -3.5)

    return Sig, rho, Pi, P, vr, Q, Om, ka, H, T

def plotTheDisks(r, GM, Mdot, Jdot, al, mu):

    Sig, rho, Pi, P, vr, Q, Om, ka, H, T = calc_thin_disk(
                                                r, GM, Mdot, Jdot, al, mu)
    Sig1, rho1, Pi1, P1, vr1, Q1, Om1, ka1, H1, T1 = calc_gas_es(
                                                r, GM, Mdot, Jdot, al, mu)
    Sig2, rho2, Pi2, P2, vr2, Q2, Om2, ka2, H2, T2 = calc_rad_es(
                                                r, GM, Mdot, Jdot, al, mu)
    Sig3, rho3, Pi3, P3, vr3, Q3, Om3, ka3, H3, T3 = calc_gas_ff(
                                                r, GM, Mdot, Jdot, al, mu)

    print("M:     {0:.3g} M_Solar".format(GM/GMSolar))
    print("Mdot:  {0:.3g} g/s".format(Mdot))
    print("Jdot:  {0:.3g} erg".format(Jdot))
    print("alpha: {0:.3g}".format(al))

    Mach = R*Om*np.sqrt(rho/P)
    Mach1 = R*Om1*np.sqrt(rho1/P1)
    Mach2 = R*Om2*np.sqrt(rho2/P2)
    Mach3 = R*Om3*np.sqrt(rho3/P3)

    Pg = pressureGas(rho, T, (mu,))
    Pr = pressureRad(rho, T, (mu,))
    kaES = opacityES(rho, T, (mu,))
    kaFF = opacityFF(rho, T, (mu,))

    fig, ax = plt.subplots(3,4, figsize=(16,9))

    ax[0,0].plot(R, Sig1, 'g')
    ax[0,0].plot(R, Sig2, 'r')
    ax[0,0].plot(R, Sig3, 'b')
    ax[0,0].plot(R, Sig, 'k', lw=2)
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlabel(r'$R$ (cm)')
    ax[0,0].set_ylabel(r'$\Sigma$ (g/cm$^2$)')
    
    ax[0,1].plot(R, rho1, 'g')
    ax[0,1].plot(R, rho2, 'r')
    ax[0,1].plot(R, rho3, 'b')
    ax[0,1].plot(R, rho, 'k', lw=2)
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel(r'$R$ (cm)')
    ax[0,1].set_ylabel(r'$\rho$ (g/cm$^3$)')

    ax[0,2].plot(R, P1, 'g')
    ax[0,2].plot(R, P2, 'r')
    ax[0,2].plot(R, P3, 'b')
    ax[0,2].plot(R, P, 'k', lw=2)
    ax[0,2].plot(R, Pg, 'k:', lw=2)
    ax[0,2].plot(R, Pr, 'k--', lw=2)
    ax[0,2].set_xscale('log')
    ax[0,2].set_yscale('log')
    ax[0,2].set_xlabel(r'$R$ (cm)')
    ax[0,2].set_ylabel(r'$P$ (erg/cm$^2$)')

    ax[1,0].plot(R, -vr1, 'g')
    ax[1,0].plot(R, -vr2, 'r')
    ax[1,0].plot(R, -vr3, 'b')
    ax[1,0].plot(R, -vr, 'k', lw=2)
    ax[1,0].set_xscale('log')
    ax[1,0].set_yscale('log')
    ax[1,0].set_xlabel(r'$R$ (cm)')
    ax[1,0].set_ylabel(r'$-v^r$ (cm/s)')
    
    ax[1,1].plot(R, Q1, 'g')
    ax[1,1].plot(R, Q2, 'r')
    ax[1,1].plot(R, Q3, 'b')
    ax[1,1].plot(R, Q, 'k', lw=2)
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')
    ax[1,1].set_xlabel(r'$R$ (cm)')
    ax[1,1].set_ylabel(r'$Q$ (erg/cm$^2$s)')

    ax[1,2].plot(R, Mach1, 'g')
    ax[1,2].plot(R, Mach2, 'r')
    ax[1,2].plot(R, Mach3, 'b')
    ax[1,2].plot(R, Mach, 'k', lw=2)
    ax[1,2].set_xscale('log')
    ax[1,2].set_yscale('log')
    ax[1,2].set_xlabel(r'$R$ (cm)')
    ax[1,2].set_ylabel(r'$\mathcal{M}$')

    ax[2,0].plot(R, ka1, 'g')
    ax[2,0].plot(R, ka2, 'r')
    ax[2,0].plot(R, ka3, 'b')
    ax[2,0].plot(R, ka, 'k', lw=2)
    ax[2,0].plot(R, kaES, 'k:', lw=2)
    ax[2,0].plot(R, kaFF, 'k--', lw=2)
    ax[2,0].set_xscale('log')
    ax[2,0].set_yscale('log')
    ax[2,0].set_xlabel(r'$R$ (cm)')
    ax[2,0].set_ylabel(r'$\kappa$ (cm$^2$/g)')
    
    ax[2,1].plot(R, H1, 'g')
    ax[2,1].plot(R, H2, 'r')
    ax[2,1].plot(R, H3, 'b')
    ax[2,1].plot(R, H, 'k', lw=2)
    ax[2,1].set_xscale('log')
    ax[2,1].set_yscale('log')
    ax[2,1].set_xlabel(r'$R$ (cm)')
    ax[2,1].set_ylabel(r'$H$ (cm)')

    ax[2,2].plot(R, T1, 'g')
    ax[2,2].plot(R, T2, 'r')
    ax[2,2].plot(R, T3, 'b')
    ax[2,2].plot(R, T, 'k', lw=2)
    ax[2,2].set_xscale('log')
    ax[2,2].set_yscale('log')
    ax[2,2].set_xlabel(r'$R$ (cm)')
    ax[2,2].set_ylabel(r'$T$ (K)')

    ax[0,3].plot(R, Sig1*ka1, 'g')
    ax[0,3].plot(R, Sig2*ka2, 'r')
    ax[0,3].plot(R, Sig3*ka3, 'b')
    ax[0,3].plot(R, Sig*ka, 'k', lw=2)
    ax[0,3].set_xscale('log')
    ax[0,3].set_yscale('log')
    ax[0,3].set_xlabel(r'$R$ (cm)')
    ax[0,3].set_ylabel(r'$\tau$')
    
    ax[1,3].plot(R, Om1*np.sqrt(P1/rho1)/(np.pi*G*Sig1), 'g')
    ax[1,3].plot(R, Om2*np.sqrt(P2/rho2)/(np.pi*G*Sig2), 'r')
    ax[1,3].plot(R, Om3*np.sqrt(P3/rho3)/(np.pi*G*Sig3), 'b')
    ax[1,3].plot(R, Om*np.sqrt(P/rho)/(np.pi*G*Sig), 'k', lw=2)
    ax[1,3].set_xscale('log')
    ax[1,3].set_yscale('log')
    ax[1,3].set_xlabel(r'$R$ (cm)')
    ax[1,3].set_ylabel(r'Toomre q')

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":

    Mdot = 1.0 * MSolar / Year
    #Mdot = 1.0e16
    GM = 1.0e8 * GMSolar
    Rs = 3.0 * (GM/GMSolar)*RsSolar
    #Rs = 0.0
    Ro = 1.0e5 * (GM/GMSolar)*RsSolar
    Rin = 3 * (GM/GMSolar)*RsSolar * 1.00001
    #Rin = 10.0*RsSolar

    Jdot = Mdot*math.sqrt(GM*Rs)
    R = np.logspace(math.log10(Rin), math.log10(Ro), base=10.0, num=200)
    mu = 0.615
    al = 1.0e-2

    plotTheDisks(R, GM, Mdot, Jdot, al, mu)
