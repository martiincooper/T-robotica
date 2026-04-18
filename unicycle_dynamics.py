import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PARÁMETROS FÍSICOS DEL ROBOT
# ==========================================
m = 5.0      # Masa total del robot (kg)
r = 0.05     # Radio de las ruedas (m)
L = 0.30     # Distancia entre ruedas / Ancho de vía (m)
I = 0.056    # Momento de inercia alrededor del COM (kg*m^2)

# NUEVOS PARÁMETROS: Coeficientes de roce (fricción viscosa)
c_v = 0.45   # Coeficiente de roce lineal (N*s/m)
c_w = 0.08   # Coeficiente de roce rotacional (N*m*s/rad)
factor_reproduccion = 2.2  # >1 acelera la animación

dt = 0.05    # Paso de tiempo de integración (s)
t_max = 18.0  # Demo breve con cambios visibles

# Variables de estado iniciales [x, y, theta, v, omega]
x, y, theta = 0.0, 0.0, np.pi/2  # Empieza apuntando hacia arriba (eje Y)
v, omega = 0.0, 0.0              # Parte del reposo

# Listas para guardar el historial
hist_x, hist_y, hist_theta = [], [], []
hist_estado = []

# ==========================================
# 2. BUCLE DE DINÁMICA (Newton-Euler con Roce)
# ==========================================
v_tol = 0.03      # Umbral para considerar reposo lineal [m/s]
omega_tol = 0.05  # Umbral para considerar reposo angular [rad/s]

# Notación solicitada:
# - Adelante: tau_R > 0, tau_L > 0, tau_R = tau_L
# - Atrás: tau_R < 0, tau_L < 0, tau_R = tau_L
# - Sola rueda: tau_R > 0, tau_L = 0
# - Ruedas contrapuestas: tau_R = -tau_L
maneuvers = [
    {
        "tau_R": 0.06,
        "tau_L": 0.06,
        "dur": 2.0,
        "label": "Adelante: tau_R > 0, tau_L > 0, tau_R = tau_L",
    },
    {
        "tau_R": -0.06,
        "tau_L": -0.06,
        "dur": 2.0,
        "label": "Atras: tau_R < 0, tau_L < 0, tau_R = tau_L",
    },
    {
        "tau_R": 0.06,
        "tau_L": 0.00,
        "dur": 2.0,
        "label": "Sola rueda: tau_R > 0, tau_L = 0",
    },
    {
        "tau_R": 0.04,
        "tau_L": -0.04,
        "dur": 2.0,
        "label": "Ruedas contrapuestas: tau_R = -tau_L",
    },
]

mode_idx = 0
phase_time = 0.0
stopping_between_modes = False

for _ in range(int(t_max / dt)):
    if mode_idx >= len(maneuvers):
        break

    if stopping_between_modes:
        # Fase de alto completo antes de pasar a la siguiente maniobra
        tau_R, tau_L = 0.0, 0.0
        estado_txt = "Alto completo entre cambios"
        if abs(v) < v_tol and abs(omega) < omega_tol:
            mode_idx += 1
            phase_time = 0.0
            stopping_between_modes = False
            if mode_idx >= len(maneuvers):
                break
    else:
        current = maneuvers[mode_idx]
        tau_R = current["tau_R"]
        tau_L = current["tau_L"]
        estado_txt = current["label"]
        phase_time += dt
        if phase_time >= current["dur"]:
            if mode_idx < len(maneuvers) - 1:
                stopping_between_modes = True
            else:
                # Tras la ultima maniobra no hay siguiente cambio
                break
        
    # --- MODELO DINÁMICO CON ROCE ---
    
    # 1. Fuerzas y Torques de Tracción generados por los motores
    F_traccion = (tau_R + tau_L) / r
    Torque_traccion = (L * (tau_R - tau_L)) / (2 * r)
    
    # 2. Fuerzas y Torques de Roce (oponentes al movimiento)
    F_roce = c_v * v
    Torque_roce = c_w * omega
    
    # 3. Ecuaciones Dinámicas (Aceleraciones netas)
    v_dot = (F_traccion - F_roce) / m
    omega_dot = (Torque_traccion - Torque_roce) / I
    
    # Integración de Euler para Velocidades
    v += v_dot * dt
    omega += omega_dot * dt
    
    # Ecuaciones Cinemáticas
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = omega
    
    # Integración de Euler para Posiciones
    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt
    
    # Guardar registros
    hist_x.append(x)
    hist_y.append(y)
    hist_theta.append(theta)
    hist_estado.append(estado_txt)

# ==========================================
# 3. ANIMACIÓN
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title("Simulación Dinámica Uniciclo (Con inercia y roce del suelo)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# Ajustar vista (dinámico según el recorrido)
ax.set_xlim(min(hist_x) - 0.5, max(hist_x) + 0.5)
ax.set_ylim(min(hist_y) - 0.5, max(hist_y) + 0.5)

# Gráficos
linea_rastro, = ax.plot([], [], 'b--', alpha=0.5, label='Rastro del COM')
robot_chasis = plt.Circle((0, 0), L/2, color='green', alpha=0.5, zorder=4, label='Chasis')
ax.add_patch(robot_chasis)
linea_orientacion, = ax.plot([], [], 'k-', linewidth=3, zorder=5)

texto_maniobra = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                         fontsize=12, fontweight='bold', color='red',
                         verticalalignment='top')
ax.legend(loc="lower right")

def init():
    linea_rastro.set_data([], [])
    robot_chasis.center = (0, 0)
    linea_orientacion.set_data([], [])
    texto_maniobra.set_text('')
    return linea_rastro, robot_chasis, linea_orientacion, texto_maniobra

def update(frame):
    # Actualizar rastro
    linea_rastro.set_data(hist_x[:frame], hist_y[:frame])
    
    # Actualizar chasis y orientación
    cx, cy = hist_x[frame], hist_y[frame]
    robot_chasis.center = (cx, cy)
    
    dx = (L/2) * np.cos(hist_theta[frame])
    dy = (L/2) * np.sin(hist_theta[frame])
    linea_orientacion.set_data([cx, cx + dx], [cy, cy + dy])
    
    # Actualizar texto
    texto_maniobra.set_text(hist_estado[frame])
    
    return linea_rastro, robot_chasis, linea_orientacion, texto_maniobra

ani = animation.FuncAnimation(fig, update, frames=len(hist_x), 
                              init_func=init, blit=False, interval=(dt * 1000) / factor_reproduccion)

plt.show()