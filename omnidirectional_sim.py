import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PARÁMETROS DE SIMULACIÓN Y DEL ROBOT
# ==========================================
v0 = 0.5           # Velocidad lineal de avance (m/s)
R_curva = 0.5      # Radio de la semicircunferencia (m)
dt = 0.02          # Paso de tiempo de simulación (s)

# Tiempos de cada tramo
t1 = 2.0 / v0                  # 2 metros recto
t2 = (np.pi * R_curva) / v0    # Semicircunferencia izquierda
t3 = 2.0 / v0                  # 2 metros recto
t4 = (np.pi * R_curva) / v0    # Semicircunferencia derecha

tiempo_total = t1 + t2 + t3 + t4
t_array = np.arange(0, tiempo_total, dt)

# Listas para guardar el historial de la trayectoria
hist_x, hist_y, hist_theta = [], [], []

# Condiciones iniciales globales (x, y, theta)
x, y, theta = 0.0, 0.0, 0.0

# ==========================================
# 2. BUCLE DE SIMULACIÓN (INTEGRACIÓN DE EULER)
# ==========================================
for t in t_array:
    # Determinar el perfil de velocidad local según el tiempo
    if t < t1:
        # Tramo 1: Recto
        vx_R, vy_R, omega_R = v0, 0.0, 0.0
    elif t < t1 + t2:
        # Tramo 2: Curva a la izquierda (omega positivo)
        vx_R, vy_R, omega_R = v0, 0.0, v0 / R_curva
    elif t < t1 + t2 + t3:
        # Tramo 3: Recto
        vx_R, vy_R, omega_R = v0, 0.0, 0.0
    else:
        # Tramo 4: Curva a la derecha (omega negativo)
        vx_R, vy_R, omega_R = v0, 0.0, -v0 / R_curva
        
    # Modelo cinemático global (Rotación del marco local al global)
    x_dot = vx_R * np.cos(theta) - vy_R * np.sin(theta)
    y_dot = vx_R * np.sin(theta) + vy_R * np.cos(theta)
    theta_dot = omega_R
    
    # Integración numérica de Euler
    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt
    
    # Guardar estado actual
    hist_x.append(x)
    hist_y.append(y)
    hist_theta.append(theta)

# ==========================================
# 3. ANIMACIÓN PARA EL VIDEO
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title("Simulación Robot Omnidireccional - Problema 2")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# Límites del gráfico dinámicos según la trayectoria
ax.set_xlim(min(hist_x) - 1, max(hist_x) + 1)
ax.set_ylim(min(hist_y) - 1, max(hist_y) + 1)

# Elementos a dibujar: trayectoria completa, rastro y el robot
linea_trayectoria, = ax.plot(hist_x, hist_y, 'k--', alpha=0.3, label="Trayectoria Ideal")
linea_rastro, = ax.plot([], [], 'b-', linewidth=2, label="Rastro del Robot")
robot_chasis = plt.Circle((0, 0), 0.15, color='orange', zorder=5)
ax.add_patch(robot_chasis)

# Vector para mostrar hacia dónde apunta el robot
quiver_orientacion = ax.quiver(0, 0, 0, 0, color='red', scale=15, zorder=6)

ax.legend(loc="upper left")

def init():
    linea_rastro.set_data([], [])
    robot_chasis.center = (0, 0)
    return linea_rastro, robot_chasis, quiver_orientacion

def update(frame):
    # Actualizar rastro
    linea_rastro.set_data(hist_x[:frame], hist_y[:frame])
    
    # Actualizar posición del chasis
    robot_chasis.center = (hist_x[frame], hist_y[frame])
    
    # Actualizar orientación (flecha roja)
    u = np.cos(hist_theta[frame])
    v = np.sin(hist_theta[frame])
    quiver_orientacion.set_offsets(np.c_[hist_x[frame], hist_y[frame]])
    quiver_orientacion.set_UVC(u, v)
    
    return linea_rastro, robot_chasis, quiver_orientacion

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=len(t_array), 
                              init_func=init, blit=False, interval=dt*1000)

plt.show()