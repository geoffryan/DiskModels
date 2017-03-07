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
Year = 3.15569252e7


GMSolar = 0.5*RsSolar*c*c

def ka_ff(rho, T):
    return 5.0e24 * rho * np.power(T, -3.5)

def calc_gas_es(r, GM, Mdot, Jdot, al, mu):

    omK = np.sqrt(GM/(r*r*r))
    Mdotf = Mdot - Jdot/(r*r*omK)
    Q = 3.0/(8*np.pi) * Mdotf * omK*omK

    SigNu = Mdotf/(3*np.pi)
    Pi = SigNu * omK / al
    Om = omK

    Sig = np.power( 4*SB/(3*ka_es) * np.power(mu*Pi/kB,4)/Q, 0.2)
    T = mu*Pi/(kB*Sig)

    H = r * np.sqrt(r*Pi/(GM*Sig))
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

    H = r * np.sqrt(r*Pi/(GM*Sig))
    rho = Sig/H
    P = Pi/H

    vr = -Mdot / (2*np.pi*r*Sig)

    ka = ka_es * np.ones(r.shape)

    return Sig, rho, Pi, P, vr, Q, Om, ka, H, T


if __name__ == "__main__":

    Mdot = 1.0e-10 * MSolar / Year
    GM = 3.0 * GMSolar
    Rs = 6.0 * RsSolar
    Ro = 1.0e4 * Rs

    Jdot = -2*np.pi*Mdot*math.sqrt(GM*Rs)
    R = np.logspace(math.log10(Rs), math.log10(Ro), base=10.0, num=200)
    mu = 1.0
    al = 0.1

    Sig1, rho1, Pi1, P1, vr1, Q1, Om1, ka1, H1, T1 = calc_gas_es(
                                                    R, GM, Mdot, Jdot, al, mu)
    Sig2, rho2, Pi2, P2, vr2, Q2, Om2, ka2, H2, T2 = calc_rad_es(
                                                    R, GM, Mdot, Jdot, al, mu)

    fig, ax = plt.subplots(3,3, figsize=(12,9))

    ax[0,0].plot(R, Sig1, 'g')
    ax[0,0].plot(R, Sig2, 'r')
    ax[0,0].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[0,0].set_xlabel(r'$R$ (cm)')
    ax[0,0].set_ylabel(r'$\Sigma$ (g/cm^2)')
    
    ax[0,1].plot(R, P1, 'g')
    ax[0,1].plot(R, P2, 'r')
    ax[0,1].set_xscale('log')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel(r'$R$ (cm)')
    ax[0,1].set_ylabel(r'$P$ (erg/cm^2)')

    print Mdot

    plt.show()

    

