import math, time
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from tensorflow import keras
import matplotlib.pyplot as plt

# Setting
miller_opt = 0

# Model paths
model_0d_path = '/models/model_0d'
model_1d_enc_path = '/models/model_1d_enc'
model_1d_dec_path = '/models/model_1d_dec'
model_2d_enc_path = '/models/model_2d_enc'
model_2d_dec_path = '/models/model_2d_dec'

# Find geometric axis from the boundary coordinates
def findaxis(rbnd, zbnd, mode=2, ntheta=64):
    if mode == 0:
        return 0.5 * (min(rbnd) + max(rbnd)), 0.5 * (min(zbnd) + max(zbnd))
    else:
        rbnd, zbnd = np.array(rbnd), np.array(zbnd)
        r0_tmp, z0_tmp = 0.5 * (min(rbnd) + max(rbnd)), 0.5 * (min(zbnd) + max(zbnd))
        lbnd_old = np.sqrt((rbnd - r0_tmp) ** 2 + (zbnd - z0_tmp) ** 2)
        theta_old = np.array([math.atan2((zbnd[j] - z0_tmp) / lbnd_old[j], (r0_tmp - rbnd[j]) / lbnd_old[j]) for j in range(len(rbnd))])
        f = interpolate.interp1d(theta_old, lbnd_old, fill_value='extrapolate')
    
        theta = np.array([(2 * np.pi * (i / ntheta) if i <= 0.5 * ntheta else 2 * np.pi * (i / ntheta - 1)) for i in range(ntheta)])
        lbnd = np.array([f(theta[i]) for i in range(ntheta)])
        r0, z0 = r0_tmp - np.mean(lbnd * np.cos(theta)), z0_tmp + np.mean(lbnd * np.sin(theta))
        if mode == 1:
            return r0, z0
        else:
            return 0.5 * (min(rbnd) + max(rbnd)), z0

# Generate the boundary coordinates according to Miller geometry
def bnd_gen(r0, z0, a0, kappa, deltau, deltal, nbnd):
    x = np.linspace(-np.pi, np.pi, nbnd)
    delta = np.heaviside(np.sin(x), 0) * deltau + np.heaviside(-np.sin(x), 1) * deltal
    delta2 = np.arcsin(delta) if miller_opt else delta
    rtmp = r0 + a0 * np.cos(x + delta2 * np.sin(x))
    ztmp = z0 + kappa * a0 * np.sin(x)
    return np.array(rtmp), np.array(ztmp)

# Find length from axis in Miller geometry
def lbnd_miller(x, amin, kappa, deltau, deltal):
    xtmp = np.linspace(-np.pi, np.pi, len(x))
    delta = np.heaviside(np.sin(xtmp), 0) * deltau + np.heaviside(-np.sin(xtmp), 1) * deltal
    delta2 = np.arcsin(delta) if miller_opt else delta
    rtmp = amin * np.cos(xtmp + delta2 * np.sin(xtmp))
    ztmp = kappa * amin * np.sin(xtmp)
    ltmp = np.sqrt(rtmp ** 2 + ztmp ** 2)
    xtmp = np.array([math.atan2(ztmp[i], rtmp[i]) for i in range(len(rtmp))])
    f = interpolate.interp1d(xtmp, ltmp, fill_value='extrapolate')
    lout = np.array([f(x[i]) for i in range(len(x))])
    return lout

# Miller-fitted geometric parameters from the boundary coordinates
def rz2miller(rbnd, zbnd):
    r0, z0 = findaxis(rbnd, zbnd, mode=2)
    ltheta0 = np.sqrt((rbnd - r0) ** 2 + (zbnd - z0) ** 2)
    theta0 = np.array([math.atan2(zbnd[i] - z0, rbnd[i] - r0) for i in range(len(rbnd))])
    popt, pcov = curve_fit(f=lbnd_miller, xdata=theta0, ydata=ltheta0, p0=(r0/3, 1.8, 0.3, 0.7))
    amin, kappa, deltau, deltal = popt
    return r0, z0, amin, kappa, deltau, deltal

# NN model to solve ideal MHD equilibrium quantities
class Ndmodels():
    def __init__(self, path='.'):
        # Load models
        self.model_0d = keras.models.load_model(path + model_0d_path, compile=False)
        self.model_1d_enc = keras.models.load_model(path + model_1d_enc_path, compile=False)
        self.model_1d_dec = keras.models.load_model(path + model_1d_dec_path, compile=False)
        self.model_2d_enc = keras.models.load_model(path + model_2d_enc_path, compile=False)
        self.model_2d_dec = keras.models.load_model(path + model_2d_dec_path, compile=False)
        #self.model_0d.compile(run_eagerly=True)
        #self.model_1d_enc.compile(run_eagerly=True)
        #self.model_1d_dec.compile(run_eagerly=True)
        #self.model_2d_enc.compile(run_eagerly=True)
        #self.model_2d_dec.compile(run_eagerly=True)

    def set_inputs(self, pstar_201, jstar_201, rbnd, zbnd): # pstar_201 & jstar_201 should be shape of (201)
        self.pstar, self.jstar = pstar_201, jstar_201
        self.r0, self.z0, self.amin, self.k, self.du, self.dl = rz2miller(rbnd, zbnd) # Miller-fitted geometry
        self.asp = self.r0 / self.amin # Aspect ratio
        self.data_in = np.zeros([1, 201, 6])
        self.data_in[0,:,0] = self.asp
        self.data_in[0,:,1] = self.k
        self.data_in[0,:,2] = self.du
        self.data_in[0,:,3] = self.dl
        self.data_in[0,:,4] = self.jstar
        self.data_in[0,:,5] = self.pstar
        
    # 0d extraction
    # betat, betap, betan, wmhd_star, q95, li, volume_star, area_star = self.get_0ds()
    def get_0ds(self):
        yy = self.model_0d.predict(self.data_in)[0]
        return yy

    # 1d extraction
    # psi_star, rho_star, f_star, dvdrho_star, dpsidrho_star, q, s, rout_star, rin_star, k1d, du1d, dl1d, ftrap, bavg_star, gm1_star, gm2_star, gm3_star, gm4_star, gm5_star, gm6_star, gm7_star = self.get_1ds()
    def get_1ds(self):
        z, _, _ = self.model_1d_enc.predict(self.data_in) # Encoding
        yy = self.model_1d_dec.predict(z) # Decoding
        return yy

    # 2d extraction
    # psi_star, phi_star = self.get_2ds()
    def get_2ds(self):
        z, _, _ = self.model_2d_enc.predict(self.data_in) # Encoding
        yy = self.model_2d_dec.predict(z) # Decoding
        return yy

if __name__ == '__main__':
    # Set inputs 
    # pstar: normalized pressure
    # jstar: normalized current density
    # rbnd, zbnd: boundary coordinates
    pstar = np.linspace(0.01, 0.0, 201) # pstar = p [MKS] * mu0 / B0 ** 2
    jstar = np.linspace(2.00, 0.0, 201) # jstar = j [MKS] * mu0 * R0 / B0
    rgeo, amin, kappa, delta = 1.8, 0.5, 1.7, 0.3
    rbnd, zbnd = bnd_gen(r0=rgeo, z0=0, a0=amin, kappa=kappa, deltau=delta, deltal=delta, nbnd=100)
    
    # Load models
    models = Ndmodels()
    models.set_inputs(pstar, jstar, rbnd, zbnd)

    # Predict N-D (0D-2D) physical quantities
    t0 = time.time()
    print('betat, betap, betan, wmhd_star, q95, li, volume_star, area_star:\n',list(models.get_0ds()),'\nTime for 0D:',time.time()-t0)
    t1 = time.time()
    a = models.get_1ds()
    print('Time for 1D:',time.time()-t1)
    t2 = time.time()
    b = models.get_2ds()
    print('Time for 2D:',time.time()-t2)
    
    # Plot 2d quantities
    rstar, zstar = np.linspace(-1-10/129, 1+10/129, 129), np.linspace(-1-10/129, 1+10/129, 129)
    r = rstar * amin + rgeo
    z = zstar * kappa * amin
    psistar, phistar = b[0,:,:,0].T, b[0,:,:,1].T

    fig = plt.figure(figsize = (8, 5))
    
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Predicted φ*')
    ax.contourf(r, z, phistar, levels=25)
    ax.plot(rbnd, zbnd, 'k', linewidth=1)
    ax.set_xlabel('r [m]')
    ax.set_ylabel('z [m]')
    plt.axis('scaled')
    
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Predicted ψ*')
    ax.contourf(r, z, psistar, levels=25)
    ax.plot(rbnd, zbnd, 'k', linewidth=1)
    ax.set_xlabel('r [m]')
    ax.set_ylabel('z [m]')
    plt.axis('scaled')
    
    plt.tight_layout()
    plt.show()
