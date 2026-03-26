import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Metodos.input_parser import parse_real_or_default, parse_int_or_default


def dataset_prueba_pequeno():
    """Dataset pequeno para simular aprendizaje supervisado lineal."""
    x = np.array([0.5, 1.0, 1.8, 2.5, 3.2, 4.0, 4.8, 5.5], dtype=float)
    y = np.array([2.0, 2.8, 4.1, 5.0, 6.4, 7.2, 8.5, 9.3], dtype=float)
    return x, y


def entrenar_descenso_gradiente_lineal(
    x,
    y,
    alpha=0.03,
    epocas=120,
    semilla=7,
    w_inicial=None,
    b_inicial=None,
):
    """
    Entrena un modelo lineal y_hat = w*x + b con descenso de gradiente.

    Costo: MSE = (1/m) * sum((y_hat - y)^2)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x e y deben ser vectores unidimensionales")
    if x.size != y.size:
        raise ValueError("x e y deben tener igual cantidad de datos")
    if x.size < 2:
        raise ValueError("Se requieren al menos 2 puntos")
    if alpha <= 0:
        raise ValueError("alpha debe ser mayor que 0")
    if epocas <= 0:
        raise ValueError("epocas debe ser mayor que 0")

    rng = np.random.default_rng(int(semilla))
    w = float(rng.normal(0.0, 1.0) if w_inicial is None else w_inicial)
    b = float(rng.normal(0.0, 1.0) if b_inicial is None else b_inicial)

    m = x.size

    historial_w = [w]
    historial_b = [b]
    historial_costo = []

    for _ in range(int(epocas)):
        y_hat = w * x + b
        error = y_hat - y

        costo = float(np.mean(error ** 2))
        historial_costo.append(costo)

        # Derivadas parciales del MSE (Calculo 2):
        # dJ/dw = (2/m) * sum((y_hat - y) * x)
        # dJ/db = (2/m) * sum((y_hat - y))
        # Estas derivadas marcan la pendiente de J respecto de cada parametro.
        # El descenso de gradiente actualiza en direccion opuesta para bajar J.
        grad_w = float((2.0 / m) * np.sum(error * x))
        grad_b = float((2.0 / m) * np.sum(error))

        w = w - alpha * grad_w
        b = b - alpha * grad_b

        historial_w.append(w)
        historial_b.append(b)

    return {
        "w0": float(historial_w[0]),
        "b0": float(historial_b[0]),
        "w_final": float(historial_w[-1]),
        "b_final": float(historial_b[-1]),
        "hist_w": np.array(historial_w, dtype=float),
        "hist_b": np.array(historial_b, dtype=float),
        "hist_costo": np.array(historial_costo, dtype=float),
    }


def _mapa_contorno_costo(x, y, hist_w, hist_b, resolucion=110):
    margen_w = max(0.6, 0.2 * (np.max(hist_w) - np.min(hist_w) + 1e-12))
    margen_b = max(0.6, 0.2 * (np.max(hist_b) - np.min(hist_b) + 1e-12))

    w_vals = np.linspace(np.min(hist_w) - margen_w, np.max(hist_w) + margen_w, resolucion)
    b_vals = np.linspace(np.min(hist_b) - margen_b, np.max(hist_b) + margen_b, resolucion)

    W, B = np.meshgrid(w_vals, b_vals)
    pred = W[None, :, :] * x[:, None, None] + B[None, :, :]
    J = np.mean((pred - y[:, None, None]) ** 2, axis=0)
    return W, B, J


def figura_tres_subplots_descenso(x, y, entrenamiento, animar=True, interval_ms=120):
    hist_w = np.array(entrenamiento["hist_w"], dtype=float)
    hist_b = np.array(entrenamiento["hist_b"], dtype=float)
    hist_costo = np.array(entrenamiento["hist_costo"], dtype=float)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ax1, ax2, ax3 = axs

    x_line = np.linspace(float(np.min(x) - 0.6), float(np.max(x) + 0.6), 120)

    # Subplot 1: ajuste de la recta sobre los datos
    ax1.scatter(x, y, color="tab:red", label="Datos reales", zorder=3)
    (linea_pred,) = ax1.plot([], [], color="tab:blue", linewidth=2.2, label="Prediccion")
    titulo_1 = ax1.text(0.02, 0.96, "", transform=ax1.transAxes, va="top")
    ax1.set_title("El ajuste")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    # Subplot 2: curva de costo por epoca
    ax2.set_title("Curva de costo (MSE)")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Costo")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, max(1, len(hist_costo)))
    ax2.set_ylim(0, max(1e-12, float(np.max(hist_costo) * 1.08)))
    (linea_costo,) = ax2.plot([], [], color="tab:green", linewidth=2)

    # Subplot 3: mapa de contorno de J(w, b) y trayectoria de pesos
    W, B, J = _mapa_contorno_costo(x, y, hist_w, hist_b)
    niveles = np.geomspace(max(1e-6, np.min(J) + 1e-9), np.max(J), 20)
    cs = ax3.contour(W, B, J, levels=niveles, cmap="viridis")
    ax3.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax3.set_title("Mapa de contorno del costo")
    ax3.set_xlabel("w")
    ax3.set_ylabel("b")
    ax3.grid(alpha=0.25)

    (trayectoria,) = ax3.plot([], [], "o-", color="tab:orange", markersize=3, linewidth=1.6, label="Trayectoria")
    (punto_actual,) = ax3.plot([], [], "o", color="black", markersize=6)
    ax3.legend(loc="best")

    fig.tight_layout()

    if not animar:
        y_final = hist_w[-1] * x_line + hist_b[-1]
        linea_pred.set_data(x_line, y_final)
        titulo_1.set_text(
            f"w={hist_w[-1]:.4f}, b={hist_b[-1]:.4f}\nMSE={hist_costo[-1]:.6f}"
        )

        ep = np.arange(1, len(hist_costo) + 1)
        linea_costo.set_data(ep, hist_costo)

        trayectoria.set_data(hist_w, hist_b)
        punto_actual.set_data([hist_w[-1]], [hist_b[-1]])
        return fig, None

    epocas_totales = len(hist_costo)

    def init():
        linea_pred.set_data([], [])
        linea_costo.set_data([], [])
        trayectoria.set_data([], [])
        punto_actual.set_data([], [])
        titulo_1.set_text("")
        return linea_pred, linea_costo, trayectoria, punto_actual, titulo_1

    def update(frame):
        idx = int(frame)

        y_frame = hist_w[idx] * x_line + hist_b[idx]
        linea_pred.set_data(x_line, y_frame)

        costo_idx = min(idx, epocas_totales - 1)
        titulo_1.set_text(
            f"Epoca {costo_idx + 1}/{epocas_totales}\nw={hist_w[idx]:.4f}, b={hist_b[idx]:.4f}\nMSE={hist_costo[costo_idx]:.6f}"
        )

        ep = np.arange(1, costo_idx + 2)
        linea_costo.set_data(ep, hist_costo[: costo_idx + 1])

        trayectoria.set_data(hist_w[: idx + 1], hist_b[: idx + 1])
        punto_actual.set_data([hist_w[idx]], [hist_b[idx]])

        return linea_pred, linea_costo, trayectoria, punto_actual, titulo_1

    anim = FuncAnimation(
        fig,
        update,
        frames=len(hist_w),
        init_func=init,
        interval=int(interval_ms),
        blit=False,
        repeat=False,
    )

    return fig, anim


def ejecutar_red_neuronal_descenso_gradiente():
    print("\n" + "=" * 90)
    print("SIMULACION BASE DE APRENDIZAJE - RED NEURONAL (DESCENSO DE GRADIENTE)")
    print("=" * 90)

    x, y = dataset_prueba_pequeno()
    print(f"Dataset de prueba cargado con {len(x)} puntos")

    alpha = parse_real_or_default(input("Tasa de aprendizaje alpha (default 0.03): "), 0.03, "alpha")
    epocas = parse_int_or_default(input("Numero de epocas (default 120): "), 120, "epocas")
    semilla = parse_int_or_default(input("Semilla para pesos iniciales (default 7): "), 7, "semilla")

    entrenamiento = entrenar_descenso_gradiente_lineal(
        x,
        y,
        alpha=float(alpha),
        epocas=int(epocas),
        semilla=int(semilla),
    )

    print("\nResumen de entrenamiento")
    print(f"w inicial: {entrenamiento['w0']:.6f}")
    print(f"b inicial: {entrenamiento['b0']:.6f}")
    print(f"w final:   {entrenamiento['w_final']:.6f}")
    print(f"b final:   {entrenamiento['b_final']:.6f}")
    print(f"Costo final (MSE): {float(entrenamiento['hist_costo'][-1]):.8f}")

    fig, _ = figura_tres_subplots_descenso(x, y, entrenamiento, animar=True, interval_ms=110)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ejecutar_red_neuronal_descenso_gradiente()
