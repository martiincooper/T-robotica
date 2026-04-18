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

# Coeficientes de roce viscoso con el suelo
c_v = 0.8    # Roce lineal (N*s/m)
c_w = 0.15   # Roce rotacional (N*m*s/rad)

dt = 0.05    # Paso de tiempo de integración (s)
t_array = np.arange(0, 19, dt)  # 19 s con maniobras mas diferenciadas

# Variables de estado iniciales [x, y, theta, v, omega]
x, y, theta = 0.0, 0.0, np.pi/2  # Empieza apuntando hacia arriba (eje Y)
v, omega = 0.0, 0.0              # Parte del reposo

# Listas para guardar el historial
hist_x, hist_y, hist_theta = [], [], []
hist_estado = []  # Para mostrar el texto en el video

# ==========================================
# 2. BUCLE DE DINÁMICA (Newton-Euler)
# ==========================================
for t in t_array:
    # Definición de perfiles de Torque (tau_R, tau_L)
    if t < 2.5:
        # Tramo 1: Aceleración frontal
        tau_R, tau_L = 0.03, 0.03
        estado_txt = "Aceleracion: tau_R = tau_L > 0"
    elif t < 4.5:
        # Tramo 2: Planeo inercial sin torque
        tau_R, tau_L = 0.00, 0.00
        estado_txt = "Planeo inercial: tau_R = tau_L = 0"
    elif t < 8.5:
        # Tramo 3: Sola rueda marcada (curva asimétrica evidente)
        tau_R, tau_L = 0.05, 0.00
        estado_txt = "Sola rueda: tau_R > 0, tau_L = 0"
    elif t < 11.0:
        # Tramo 4: Frenado para reducir velocidad lineal antes del giro puro
        tau_R, tau_L = -0.03, -0.03
        estado_txt = "Frenado: tau_R = tau_L < 0"
    elif t < 15.5:
        # Tramo 5: Ruedas contrapuestas fuertes (giro sobre su eje)
        tau_R, tau_L = 0.03, -0.03
        estado_txt = "Contrapuestas: tau_R = -tau_L"
    else:
        # Tramo 6: Asentamiento final por disipación
        tau_R, tau_L = 0.00, 0.00
        estado_txt = "Asentamiento final por roce"

    # Modelo dinámico con fuerzas/torques de tracción y disipación por roce
    F_traccion = (tau_R + tau_L) / r
    T_traccion = (L * (tau_R - tau_L)) / (2 * r)
    F_roce = c_v * v
    T_roce = c_w * omega

    # Ecuaciones dinámicas netas (Newton-Euler)
    v_dot = (F_traccion - F_roce) / m
    omega_dot = (T_traccion - T_roce) / I
    
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
ax.set_title("Simulación Dinámica Directa - Uniciclo (Con roce del suelo)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# Ajustar vista
ax.set_xlim(min(hist_x) - 1, max(hist_x) + 1)
ax.set_ylim(min(hist_y) - 1, max(hist_y) + 1)

# Gráficos
linea_rastro, = ax.plot([], [], 'b--', alpha=0.5, label='Rastro del COM')
robot_chasis = plt.Circle((0, 0), L/2, color='green', alpha=0.5, zorder=4, label='Chasis')
ax.add_patch(robot_chasis)
linea_orientacion, = ax.plot([], [], 'k-', linewidth=3, zorder=5)

# Texto en pantalla para identificar la maniobra
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
    
    # Dibujar la línea de orientación para ver hacia dónde apunta
    dx = (L/2) * np.cos(hist_theta[frame])
    dy = (L/2) * np.sin(hist_theta[frame])
    linea_orientacion.set_data([cx, cx + dx], [cy, cy + dy])
    
    # Actualizar texto
    texto_maniobra.set_text(hist_estado[frame])
    
    return linea_rastro, robot_chasis, linea_orientacion, texto_maniobra

ani = animation.FuncAnimation(fig, update, frames=len(t_array), 
                              init_func=init, blit=False, interval=dt*1000)

plt.show()