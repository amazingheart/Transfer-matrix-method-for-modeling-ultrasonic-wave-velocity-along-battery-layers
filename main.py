import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy.optimize import root


def field_matrix(k1, h, rho, cl, cs):
    k2l = cmath.sqrt(omega**2 / cl**2 - k1**2)
    k2s = cmath.sqrt(omega**2 / cs**2 - k1**2)
    g2l = cmath.exp(1j*k2l*h)
    g2s = cmath.exp(1j*k2s*h)
    b = omega**2 - 2 * cs**2 * k1**2

    return np.array([[k1*g2l, k1/g2l, k2s*g2s, -k2s/g2s],
                     [k2l*g2l, -k2l/g2l, -k1*g2s, -k1/g2s],
                     [1j*rho*b*g2l, 1j*rho*b/g2l, -2j*rho*k1*cs**2 * k2s*g2s, 2j*rho*k1*cs**2 * k2s/g2s],
                     [2j*rho*k1*cs**2 * k2l*g2l, -2j*rho*k1*cs**2 * k2l/g2l, 1j*rho*b*g2s, 1j*rho*b/g2s]])


def layer_matrix(k1, h, rho, cl, cs):
    field_matrix0 = field_matrix(k1, 0, rho, cl, cs)
    field_matrix1 = field_matrix(k1, h, rho, cl, cs)

    return np.linalg.solve(field_matrix0.T, field_matrix1.T).T


def system_matrix(k1):
    l1 = layer_matrix(k1, al[0], al[1], al[2], al[3])

    l2 = layer_matrix(k1, cathode[0], cathode[1], cathode[2], cathode[3])

    l2[0] += l2[3] / Kt

    l3 = layer_matrix(k1, separator[0], separator[1], separator[2], separator[3])

    l3[0] += l3[3] / Kt

    l4 = layer_matrix(k1, anode[0], anode[1], anode[2], anode[3])

    l5 = layer_matrix(k1, cu[0], cu[1], cu[2], cu[3])

    return l5 @ l4 @ l3 @ l2 @ l1


def cal_velocities(k1_vec):
    k_real, k_imag = k1_vec
    k1 = k_real + 1j * k_imag

    s = system_matrix(k1)

    det = s[1, 0] * s[3, 2] - s[3, 0] * s[1, 2]

    return [det.real, det.imag]


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
px = 1/plt.rcParams['figure.dpi']
plt.rcParams['figure.figsize'] = [1063*px, 797*px]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

f = 2e5
omega = 2 * np.pi * f

velocities = np.arange(2000, 4000, 100)
k_reals = omega / velocities
k_imags = np.arange(-100, 101, 25)

initial_guesses = [(x, y) for x in k_reals for y in k_imags]

Kts = np.arange(2, 21, 1)*1e10

'''SoC = 0'''

# Thickness, density, P-wave velocity, S-wave velocity
al = [30.08e-6, 2700, (122.67e9 / 2700) ** (1 / 2), (29e9 / 2700) ** (1 / 2)]
cathode = [74.17e-6, 2522, 1612.9, 259.1]
separator = [15.88e-6, 1048, 1515.2, 370.4]
anode = [55.3e-6, 1864, 1626.0, 172.4]
cu = [4.79e-6, 8960, (199.67e9 / 8960) ** (1 / 2), (47e9 / 8960) ** (1 / 2)]

Kts1 = []
Kts2 = []
c1s1 = []
c1s2 = []
k_imags1 = []
k_imags2 = []
for Kt in Kts:
    roots = []
    for guess in initial_guesses:
        sol = root(cal_velocities, guess, method='hybr')
        if sol.success:
            x_sol, y_sol = sol.x
            z_sol = x_sol + 1j * y_sol
            if x_sol <= 0:
                continue
            if omega / x_sol > 4200:
                continue

            # Check uniqueness (avoid duplicate roots)
            if not any(np.isclose(z_sol, r, atol=1e-6) for r in roots):
                roots.append(z_sol)

    print(f"Found {len(roots)} unique roots at %e N/m^3:" % Kt)
    for i, r in enumerate(roots):
        print(f"Root {i + 1}: {r}")

    for i in range(len(roots)):
        if omega / roots[i].real > 3000:
            Kts1.append(Kt)
            c1s1.append(omega / roots[i].real)
            k_imags1.append(roots[i].imag)
        else:
            Kts2.append(Kt)
            c1s2.append(omega / roots[i].real)
            k_imags2.append(roots[i].imag)

paras = np.polyfit(c1s2, Kts2, 3)
fun = np.poly1d(paras)
Kt0 = fun(2500)

plt.plot(Kts1, c1s1, c='b', label='Faster wave')
plt.plot(Kts2, c1s2, c='r', label='Slower wave')
plt.annotate('(%.1e, %s)' % (Kt0, 2500), (Kt0, 2500), textcoords="offset points", xytext=(-20, 10), ha='center')
plt.scatter(Kt0, 2500, c='k')
plt.ylabel('Velocity [m/s]')
plt.xlabel(r'$K_t$ [N/m$^3$]')
plt.grid()
plt.tight_layout()
plt.savefig('multi_velocity_')
plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# ax.scatter(Kts_, c1s, k_imags_)
#
# ax.set_xlabel(r'$K_t$ [N/m$^3$]')
# ax.set_ylabel('Velocity [m/s]')
# ax.set_zlabel('k_image [1/m]')
#
# plt.show()

Kts_ = []
c1s = []
k_imags_ = []
for Kt in Kts:
    roots = []
    for guess in initial_guesses:
        sol = root(cal_velocities, guess, method='hybr')
        if sol.success:
            x_sol, y_sol = sol.x
            z_sol = x_sol + 1j * y_sol
            if x_sol <= 0:
                continue
            if omega / x_sol > 3000:
                continue

            # Check uniqueness (avoid duplicate roots)
            if not any(np.isclose(z_sol, r, atol=1e-6) for r in roots):
                roots.append(z_sol)

    print(f"Found {len(roots)} unique roots at %e N/m^3:" % Kt)
    for i, r in enumerate(roots):
        print(f"Root {i + 1}: {r}")

    for i in range(len(roots)):
        Kts_.append(Kt)
        c1s.append(omega / roots[i].real)
        k_imags_.append(roots[i].imag)

plt.plot(Kts_, c1s, c='r', label='SoC=0')

paras = np.polyfit(c1s, Kts_, 3)
fun = np.poly1d(paras)
Kt0 = fun(2500)

'''SoC = 1'''

# Thickness, density, P-wave velocity, S-wave velocity
cathode = [73.56e-6, 2471, 1587.3, 258.7]
anode = [56.61e-6, 1898, 1739.1, 178.0]

Kts_ = []
c1s = []
k_imags_ = []
for Kt in Kts:
    roots = []
    for guess in initial_guesses:
        sol = root(cal_velocities, guess, method='hybr')
        if sol.success:
            x_sol, y_sol = sol.x
            z_sol = x_sol + 1j * y_sol
            if x_sol <= 0:
                continue
            if omega / x_sol > 3000:
                continue

            # Check uniqueness (avoid duplicate roots)
            if not any(np.isclose(z_sol, r, atol=1e-6) for r in roots):
                roots.append(z_sol)

    print(f"Found {len(roots)} unique roots at %e N/m^3:" % Kt)
    for i, r in enumerate(roots):
        print(f"Root {i + 1}: {r}")

    for i in range(len(roots)):
        Kts_.append(Kt)
        c1s.append(omega / roots[i].real)
        k_imags_.append(roots[i].imag)

plt.plot(Kts_, c1s, c='g', label='SoC=1')

paras = np.polyfit(Kts_, c1s, 3)
fun = np.poly1d(paras)
vel1 = fun(Kt0)

paras = np.polyfit(c1s, Kts_, 3)
fun = np.poly1d(paras)
Kt1 = fun(2559)

plt.annotate('(%.1e, %s)' % (Kt0, 2500), (Kt0, 2500), textcoords="offset points", xytext=(-20, 10), ha='center')
plt.scatter(Kt0, 2500, c='k')

plt.annotate('(%.1e, %d)' % (Kt0, vel1), (Kt0, vel1), textcoords="offset points", xytext=(15, -10), ha='center')
plt.scatter(Kt0, vel1, c='k')

plt.annotate('(%.1e, %s)' % (Kt1, 2559), (Kt1, 2559), textcoords="offset points", xytext=(15, -10), ha='center')
plt.scatter(Kt1, 2559, c='k')

plt.xlabel(r'$K_t$ [N/m$^3$]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('multi_velocity')
plt.close()

