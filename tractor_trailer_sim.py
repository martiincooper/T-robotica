import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PARÁMETROS DEL SISTEMA (Gorilla Carts a escala)
# ==========================================
L = 0.30     # Distancia entre ejes del tractor (m)
d = 0.10     # Distancia del eje trasero al enganche (m)
L1 = 0.50    # Distancia del enganche al eje del trailer (m)

v0 = 0.5     # Velocidad lineal de avance del tractor (m/s)
dt = 0.05    # Paso de tiempo (s)

# Tiempos de simulación
phi_giro = np.deg2rad(25)  # Magnitud máxima del ángulo de dirección
omega_giro = (v0 / L) * np.tan(phi_giro)  # Velocidad angular del tractor para giro constante
factor_reproduccion = 1.8  # >1 acelera la animación en pantalla

# Trayectoria para mostrar dinámica: recta + slalom + giros opuestos + recta final
t_recta_1 = 3.0
t_slalom = 8.0
t_izquierda = np.deg2rad(180) / omega_giro
t_derecha = np.deg2rad(140) / omega_giro
t_recta_2 = 5.0

t_fin_recta_1 = t_recta_1
t_fin_slalom = t_fin_recta_1 + t_slalom
t_fin_izquierda = t_fin_slalom + t_izquierda
t_fin_derecha = t_fin_izquierda + t_derecha
t_total = t_fin_derecha + t_recta_2
t_array = np.arange(0, t_total, dt)

# Condiciones iniciales [x0, y0, theta0, theta1]
x0, y0 = 0.0, 0.0
theta0, theta1 = 0.0, 0.0

# Historial para graficar
hist_x0, hist_y0 = [], []
hist_x1, hist_y1 = [], []  # Eje del remolque

# ==========================================
# 2. INTEGRACIÓN DEL MODELO CINEMÁTICO
# ==========================================
for t in t_array:
    # Definir entradas de control (perfil de la trayectoria)
    if t < t_fin_recta_1:
        phi = 0.0  # Recta inicial
    elif t < t_fin_slalom:
        # Slalom suave para evidenciar el retraso angular del remolque
        tau = (t - t_fin_recta_1) / t_slalom
        phi = 0.75 * phi_giro * np.sin(2 * np.pi * 1.5 * tau)
    elif t < t_fin_izquierda:
        phi = phi_giro  # Giro sostenido a la izquierda
    elif t < t_fin_derecha:
        phi = -phi_giro  # Giro sostenido a la derecha
    else:
        phi = 0.0  # Recta final para observar asentamiento
        
    # Ecuaciones del modelo cinemático
    x0_dot = v0 * np.cos(theta0)
    y0_dot = v0 * np.sin(theta0)
    theta0_dot = (v0 / L) * np.tan(phi)
    
    # Derivada del trailer
    gamma = theta0 - theta1
    theta1_dot = (v0 / L1) * (np.sin(gamma) - (d / L) * np.tan(phi) * np.cos(gamma))
    
    # Integración de Euler
    x0 += x0_dot * dt
    y0 += y0_dot * dt
    theta0 += theta0_dot * dt
    theta1 += theta1_dot * dt
    
    # Calcular coordenadas del remolque para el historial
    x_H = x0 - d * np.cos(theta0)
    y_H = y0 - d * np.sin(theta0)
    x1 = x_H - L1 * np.cos(theta1)
    y1 = y_H - L1 * np.sin(theta1)
    
    hist_x0.append(x0)
    hist_y0.append(y0)
    hist_x1.append(x1)
    hist_y1.append(y1)

# ==========================================
# 3. ANIMACIÓN
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title("Simulación TTWR (Tractor + Trailer) - Problema 3")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# Ajustar límites de la ventana
ax.set_xlim(min(hist_x1) - 1, max(hist_x0) + 1)
ax.set_ylim(min(hist_y1) - 1, max(hist_y0) + 1)

# Líneas de rastro
rastro_tractor, = ax.plot([], [], 'b--', alpha=0.5, label='Rastro Tractor')
rastro_trailer, = ax.plot([], [], 'r--', alpha=0.5, label='Rastro Trailer')

# Líneas que representan los cuerpos físicos
linea_tractor, = ax.plot([], [], 'b-', linewidth=4, label='Tractor (Chasis)')
linea_enganche, = ax.plot([], [], 'k-', linewidth=2, label='Enganche')
linea_trailer, = ax.plot([], [], 'r-', linewidth=4, label='Trailer (Tolva)')

ax.legend(loc="upper left")

def init():
    rastro_tractor.set_data([], [])
    rastro_trailer.set_data([], [])
    linea_tractor.set_data([], [])
    linea_enganche.set_data([], [])
    linea_trailer.set_data([], [])
    return rastro_tractor, rastro_trailer, linea_tractor, linea_enganche, linea_trailer

def update(frame):
    # Actualizar rastros
    rastro_tractor.set_data(hist_x0[:frame], hist_y0[:frame])
    rastro_trailer.set_data(hist_x1[:frame], hist_y1[:frame])
    
    # Extraer variables del instante actual
    hx0, hy0 = hist_x0[frame], hist_y0[frame]
    ht0 = np.arctan2(hist_y0[frame] - hist_y0[frame-1] if frame > 0 else 0, 
                     hist_x0[frame] - hist_x0[frame-1] if frame > 0 else 1)
    ht0 = theta0 # simplificación visual usando el estado real guardado costaría memoria, 
                 # recalcularemos las posiciones en base a los historiales si es necesario,
                 # pero es más fácil re-simular el frame actual:
    
    # Recalculamos theta0 y theta1 en el frame para dibujar (aproximación visual rápida)
    if frame == 0:
        ang_0 = 0.0
        ang_1 = 0.0
    else:
        ang_0 = np.arctan2(hist_y0[frame]-hist_y0[frame-1], hist_x0[frame]-hist_x0[frame-1])
        ang_1 = np.arctan2(hist_y1[frame]-hist_y1[frame-1], hist_x1[frame]-hist_x1[frame-1])

    # Geometría para dibujar en este frame
    # Tractor: eje trasero (hx0, hy0) a eje delantero
    x_front = hx0 + L * np.cos(ang_0)
    y_front = hy0 + L * np.sin(ang_0)
    linea_tractor.set_data([hx0, x_front], [hy0, y_front])
    
    # Enganche: eje trasero a punto H
    x_H = hx0 - d * np.cos(ang_0)
    y_H = hy0 - d * np.sin(ang_0)
    linea_enganche.set_data([hx0, x_H], [hy0, y_H])
    
    # Trailer: punto H al eje del trailer (hist_x1, hist_y1)
    linea_trailer.set_data([x_H, hist_x1[frame]], [y_H, hist_y1[frame]])
    
    return rastro_tractor, rastro_trailer, linea_tractor, linea_enganche, linea_trailer

ani = animation.FuncAnimation(fig, update, frames=len(t_array), 
                              init_func=init, blit=False, interval=(dt * 1000) / factor_reproduccion)

plt.show()