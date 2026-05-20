import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

try:
    from scipy.integrate import solve_ivp
except ImportError as exc:  # pragma: no cover
    solve_ivp = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


def _formatear_numero(valor, precision=6):
    valor = complex(valor)
    real = float(np.real(valor))
    imag = float(np.imag(valor))

    if abs(imag) < 1e-12:
        return f"{real:.{precision}g}"

    signo = "+" if imag >= 0 else "-"
    return f"{real:.{precision}g} {signo} {abs(imag):.{precision}g}i"


def _formatear_vector(vector):
    vector = np.asarray(vector)
    valores = ", ".join(_formatear_numero(valor) for valor in vector)
    return f"[{valores}]"


def _validar_matriz_A(A):
    matriz = np.asarray(A, dtype=float)
    if matriz.shape != (2, 2):
        raise ValueError("A debe ser una matriz 2x2")
    if not np.all(np.isfinite(matriz)):
        raise ValueError("A debe contener solo valores finitos")
    return matriz


def _validar_t_eval(t_span, t_eval):
    t_span = tuple(float(v) for v in t_span)
    if len(t_span) != 2:
        raise ValueError("t_span debe tener exactamente 2 valores")
    if t_span[0] == t_span[1]:
        raise ValueError("t_span no puede tener extremos iguales")

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
    else:
        t_eval = np.asarray(t_eval, dtype=float)

    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval debe ser un vector unidimensional con al menos 2 puntos")
    if not np.all(np.isfinite(t_eval)):
        raise ValueError("t_eval debe contener solo valores finitos")

    return t_span, t_eval


def construir_forzante_vectorial(f_t):
    if not callable(f_t):
        raise TypeError("f_t debe ser una funcion de Python que reciba t")

    def forzante(t):
        valor = np.asarray(f_t(float(t)), dtype=float).reshape(-1)
        if valor.size != 2:
            raise ValueError("f_t(t) debe devolver un vector de longitud 2")
        if not np.all(np.isfinite(valor)):
            raise ValueError("f_t(t) debe devolver valores finitos")
        return valor

    return forzante


def clasificar_estabilidad(eigvals, tol=1e-12):
    eigvals = np.asarray(eigvals, dtype=complex)
    re = np.real(eigvals)
    im = np.imag(eigvals)

    if np.all(np.abs(im) < tol):
        if np.all(re < -tol):
            return "Nodo estable"
        if np.all(re > tol):
            return "Nodo inestable"
        if re[0] * re[1] < 0:
            return "Punto silla"
        if np.all(np.abs(re) < tol):
            return "Centro lineal / neutro"
        return "Nodo degenerado"

    if np.all(re < -tol):
        return "Foco estable"
    if np.all(re > tol):
        return "Foco inestable"
    if np.all(np.abs(re) < tol):
        return "Centro"
    return "Estabilidad mixta"


def _es_forzante_constante(forzante, t_span, muestras=5, tol=1e-10):
    tiempos = np.linspace(float(t_span[0]), float(t_span[1]), int(max(2, muestras)))
    valores = np.array([forzante(t) for t in tiempos], dtype=float)
    return np.allclose(valores, valores[0], rtol=1e-8, atol=tol), valores[0]


def _construir_solucion_homogenea_matricial(matriz_A, eigvals, eigvecs):
    P = np.asarray(eigvecs, dtype=complex)
    D = np.diag(np.asarray(eigvals, dtype=complex))

    if np.linalg.matrix_rank(P) == 2 and np.linalg.det(P) != 0:
        return {
            "diagonalizable": True,
            "latex": (
                r"X_h(t) = P\,e^{Dt}\,P^{-1}X(0)"
                r" = P\,\begin{pmatrix}e^{\lambda_1 t} & 0\\0 & e^{\lambda_2 t}\end{pmatrix}P^{-1}X(0)"
            ),
            "texto": "X_h(t) = P e^{Dt} P^{-1} X(0), con D = diag(lambda_1, lambda_2)",
            "P": P,
            "D": D,
        }

    return {
        "diagonalizable": False,
        "latex": r"X_h(t) = e^{At}X(0)",
        "texto": "X_h(t) = exp(A t) X(0)",
        "P": P,
        "D": D,
    }

def _construir_solucion_componentes(eigvals, eigvecs, equilibrio_desplazado=None):
    eigvals = np.asarray(eigvals, dtype=complex).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=complex)

    if eigvecs.shape != (2, 2) or eigvals.size != 2:
        return {
            "disponible": False,
            "texto": "No se pudo construir la solucion por componentes.",
            "x_t": None,
            "y_t": None,
        }

    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]
    lambda1_txt = _formatear_numero(eigvals[0])
    lambda2_txt = _formatear_numero(eigvals[1])

    x_h = (
        f"{_formatear_numero(v1[0])}*C1*exp({lambda1_txt}*t)"
        f" + {_formatear_numero(v2[0])}*C2*exp({lambda2_txt}*t)"
    )
    y_h = (
        f"{_formatear_numero(v1[1])}*C1*exp({lambda1_txt}*t)"
        f" + {_formatear_numero(v2[1])}*C2*exp({lambda2_txt}*t)"
    )

    if equilibrio_desplazado is not None:
        xp_txt = _formatear_numero(equilibrio_desplazado[0])
        yp_txt = _formatear_numero(equilibrio_desplazado[1])
        x_t = f"x(t) = {xp_txt} + ({x_h})"
        y_t = f"y(t) = {yp_txt} + ({y_h})"
        texto = f"x(t) = {xp_txt} + ({x_h}); y(t) = {yp_txt} + ({y_h})"
    else:
        x_t = f"x(t) = {x_h}"
        y_t = f"y(t) = {y_h}"
        texto = f"x(t) = {x_h}; y(t) = {y_h}"

    return {
        "disponible": True,
        "texto": texto,
        "x_t": x_t,
        "y_t": y_t,
        "lambda1": lambda1_txt,
        "lambda2": lambda2_txt,
    }


def _integrar_trayectoria_sistema(sistema, t_span, x0, t_eval):
    if solve_ivp is not None:
        solucion = solve_ivp(
            sistema,
            t_span,
            x0,
            t_eval=t_eval,
            dense_output=False,
            rtol=1e-8,
            atol=1e-10,
        )
        if not solucion.success:
            raise RuntimeError(f"solve_ivp no pudo completar la trayectoria: {solucion.message}")
        return solucion

    t_eval = np.asarray(t_eval, dtype=float)
    y = np.zeros((2, t_eval.size), dtype=float)
    y[:, 0] = np.asarray(x0, dtype=float)

    for i in range(t_eval.size - 1):
        ax_fase.axvline(float(equilibrio_desplazado[0]), color="red", linestyle="--", alpha=0.28, linewidth=1.2)
        ax_fase.axhline(float(equilibrio_desplazado[1]), color="red", linestyle="--", alpha=0.28, linewidth=1.2)
        t0 = float(t_eval[i])
        t1 = float(t_eval[i + 1])
        h = t1 - t0
        yi = y[:, i]

        k1 = np.asarray(sistema(t0, yi), dtype=float)
        k2 = np.asarray(sistema(t0 + 0.5 * h, yi + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(sistema(t0 + 0.5 * h, yi + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(sistema(t1, yi + h * k3), dtype=float)

        y[:, i + 1] = yi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return SimpleNamespace(
        t=t_eval,
        y=y,
        success=True,
        message="integracion RK4 interna",
    )


def resolver_sistema_dinamico_no_homogeneo(A, f_vector, X0_list, t_span, t_eval=None, t_ref=None):
    matriz_A = _validar_matriz_A(A)
    if callable(f_vector):
        forzante = construir_forzante_vectorial(f_vector)
    else:
        vector_forzante = np.asarray(f_vector, dtype=float).reshape(-1)
        if vector_forzante.size != 2:
            raise ValueError("f_vector debe tener exactamente 2 componentes")
        if not np.all(np.isfinite(vector_forzante)):
            raise ValueError("f_vector debe contener solo valores finitos")

        def forzante(_t):
            return vector_forzante

    t_span, t_eval = _validar_t_eval(t_span, t_eval)

    if not X0_list:
        raise ValueError("X0_list no puede estar vacia")

    condiciones_iniciales = []
    for idx, x0 in enumerate(X0_list, start=1):
        punto = np.asarray(x0, dtype=float).reshape(-1)
        if punto.size != 2:
            raise ValueError(f"X0_list[{idx}] debe tener exactamente 2 componentes")
        if not np.all(np.isfinite(punto)):
            raise ValueError(f"X0_list[{idx}] debe contener solo valores finitos")
        condiciones_iniciales.append(punto)

    eigvals, eigvecs = np.linalg.eig(matriz_A)
    clasificacion = clasificar_estabilidad(eigvals)
    solucion_homogenea = _construir_solucion_homogenea_matricial(matriz_A, eigvals, eigvecs)

    es_constante, f_constante = True, np.asarray(forzante(t_span[0]), dtype=float).reshape(-1)
    equilibrio_desplazado = None
    if es_constante and abs(np.linalg.det(matriz_A)) > 1e-12:
        equilibrio_desplazado = -np.linalg.solve(matriz_A, f_constante)
    solucion_componentes = _construir_solucion_componentes(eigvals, eigvecs, equilibrio_desplazado)

    def sistema(t, x):
        return matriz_A @ np.asarray(x, dtype=float) + forzante(t)

    trayectorias = []
    for x0 in condiciones_iniciales:
        solucion = _integrar_trayectoria_sistema(sistema, t_span, x0, t_eval)
        trayectorias.append(solucion)

    t_visual = float(t_ref) if t_ref is not None else float(0.5 * (t_span[0] + t_span[1]))
    f_visual = forzante(t_visual)

    x_all = np.hstack([sol.y[0] for sol in trayectorias])
    y_all = np.hstack([sol.y[1] for sol in trayectorias])
    margen_x = max(0.5, 0.2 * (float(np.max(x_all)) - float(np.min(x_all)) + 1e-12))
    margen_y = max(0.5, 0.2 * (float(np.max(y_all)) - float(np.min(y_all)) + 1e-12))

    centro = np.array(
        equilibrio_desplazado if equilibrio_desplazado is not None else [np.mean(x_all), np.mean(y_all)],
        dtype=float,
    )
    span_x = max(1.0, 0.65 * max(abs(float(np.max(x_all)) - centro[0]), abs(centro[0] - float(np.min(x_all)))) + margen_x)
    span_y = max(1.0, 0.65 * max(abs(float(np.max(y_all)) - centro[1]), abs(centro[1] - float(np.min(y_all)))) + margen_y)

    x_grid = np.linspace(float(centro[0] - span_x), float(centro[0] + span_x), 32)
    y_grid = np.linspace(float(centro[1] - span_y), float(centro[1] + span_y), 32)
    X, Y = np.meshgrid(x_grid, y_grid)
    U = matriz_A[0, 0] * X + matriz_A[0, 1] * Y + f_visual[0]
    V = matriz_A[1, 0] * X + matriz_A[1, 1] * Y + f_visual[1]
    magnitud = np.hypot(U, V)
    U_norm = U / np.where(magnitud == 0, 1, magnitud)
    V_norm = V / np.where(magnitud == 0, 1, magnitud)

    fig, (ax_fase, ax_tiempo) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax_fase.streamplot(X, Y, U_norm, V_norm, color=magnitud, cmap="viridis", density=1.1, linewidth=1.0)
    colores = plt.cm.tab10(np.linspace(0, 1, max(3, len(trayectorias))))
    for idx, solucion in enumerate(trayectorias):
        color = colores[idx % len(colores)]
        ax_fase.plot(solucion.y[0], solucion.y[1], linewidth=2.0, color=color, label=f"X0 {idx + 1}")
        ax_fase.scatter([solucion.y[0][0]], [solucion.y[1][0]], color=color, s=32, edgecolors="black", linewidths=0.5)

    if equilibrio_desplazado is not None:
        ax_fase.scatter(
            [equilibrio_desplazado[0]],
            [equilibrio_desplazado[1]],
            color="red",
            s=90,
            marker="X",
            label="Equilibrio desplazado",
            zorder=5,
        )
        ax_fase.axvline(float(equilibrio_desplazado[0]), color="red", linestyle="--", alpha=0.28, linewidth=1.2)
        ax_fase.axhline(float(equilibrio_desplazado[1]), color="red", linestyle="--", alpha=0.28, linewidth=1.2)

    ax_fase.set_xlim(float(centro[0] - span_x), float(centro[0] + span_x))
    ax_fase.set_ylim(float(centro[1] - span_y), float(centro[1] + span_y))
    ax_fase.set_aspect("equal", adjustable="box")

    ax_fase.set_title(f"Retrato de fase en t = {t_visual:.3g}")
    ax_fase.set_xlabel("x")
    ax_fase.set_ylabel("y")
    ax_fase.grid(alpha=0.25)
    ax_fase.legend(loc="best")

    primera = trayectorias[0]
    ax_tiempo.plot(primera.t, primera.y[0], label="x(t)", linewidth=2.0)
    ax_tiempo.plot(primera.t, primera.y[1], label="y(t)", linewidth=2.0)
    ax_tiempo.set_title("Evolucion temporal")
    ax_tiempo.set_xlabel("t")
    ax_tiempo.set_ylabel("Estado")
    ax_tiempo.grid(alpha=0.25)
    ax_tiempo.legend(loc="best")

    fig.tight_layout()

    trayectorias_homogeneas = []
    for x0 in condiciones_iniciales:
        solucion_h = _integrar_trayectoria_sistema(
            lambda t, x: matriz_A @ np.asarray(x, dtype=float),
            t_span,
            x0,
            t_eval,
        )
        trayectorias_homogeneas.append(solucion_h)

    x_all_h = np.hstack([sol.y[0] for sol in trayectorias_homogeneas])
    y_all_h = np.hstack([sol.y[1] for sol in trayectorias_homogeneas])
    margen_x_h = max(0.5, 0.2 * (float(np.max(x_all_h)) - float(np.min(x_all_h)) + 1e-12))
    margen_y_h = max(0.5, 0.2 * (float(np.max(y_all_h)) - float(np.min(y_all_h)) + 1e-12))

    x_grid_h = np.linspace(float(np.min(x_all_h)) - margen_x_h, float(np.max(x_all_h)) + margen_x_h, 24)
    y_grid_h = np.linspace(float(np.min(y_all_h)) - margen_y_h, float(np.max(y_all_h)) + margen_y_h, 24)
    Xh, Yh = np.meshgrid(x_grid_h, y_grid_h)
    Uh = matriz_A[0, 0] * Xh + matriz_A[0, 1] * Yh
    Vh = matriz_A[1, 0] * Xh + matriz_A[1, 1] * Yh
    magnitud_h = np.hypot(Uh, Vh)
    Uh_norm = Uh / np.where(magnitud_h == 0, 1, magnitud_h)
    Vh_norm = Vh / np.where(magnitud_h == 0, 1, magnitud_h)

    fig_homogeneo, ax_h = plt.subplots(figsize=(7.0, 5.2))
    ax_h.streamplot(Xh, Yh, Uh_norm, Vh_norm, color=magnitud_h, cmap="plasma", density=1.1, linewidth=1.0)
    colores_h = plt.cm.tab10(np.linspace(0, 1, max(3, len(trayectorias_homogeneas))))
    for idx, solucion in enumerate(trayectorias_homogeneas):
        color = colores_h[idx % len(colores_h)]
        ax_h.plot(solucion.y[0], solucion.y[1], linewidth=2.0, color=color, label=f"X0 {idx + 1}")
        ax_h.scatter([solucion.y[0][0]], [solucion.y[1][0]], color=color, s=32, edgecolors="black", linewidths=0.5)

    ax_h.scatter([0.0], [0.0], color="red", s=90, marker="X", label="Equilibrio homogéneo")
    ax_h.set_title("Retrato de fase del sistema homogéneo")
    ax_h.set_xlabel("x")
    ax_h.set_ylabel("y")
    ax_h.grid(alpha=0.25)
    ax_h.legend(loc="best")
    fig_homogeneo.tight_layout()

    return {
        "A": matriz_A,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "clasificacion": clasificacion,
        "solucion_homogenea": solucion_homogenea,
        "solucion_componentes": solucion_componentes,
        "forzante_constante": bool(es_constante),
        "forzante_constante_vector": f_constante if es_constante else None,
        "equilibrio_desplazado": equilibrio_desplazado,
        "trayectorias": trayectorias,
        "trayectorias_homogeneas": trayectorias_homogeneas,
        "t_visual": t_visual,
        "fig": fig,
        "fig_homogeneo": fig_homogeneo,
    }


def ejecutar_sistema_dinamico_no_homogeneo():
    casos = [
        {
            "nombre": "Caso A - Preservacion por fuerza constante",
            "A": np.array([[0.0, -1.0], [-9.0, 0.0]], dtype=float),
            "f_t": lambda t: np.array([1.0, 0.0], dtype=float),
            "X0_list": [(1.0, 0.0), (0.5, 0.5), (-2.0, 1.0)],
            "t_span": (0.0, 10.0),
            "t_eval": np.linspace(0.0, 10.0, 600),
        },
        {
            "nombre": "Caso B - Ruptura por fuerza variable",
            "A": np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=float),
            "f_t": lambda t: np.array([np.cos(t), 0.0], dtype=float),
            "X0_list": [(1.0, 0.0), (0.5, 0.5), (-2.0, 1.0)],
            "t_span": (0.0, 10.0),
            "t_eval": np.linspace(0.0, 10.0, 600),
        },
    ]

    print("\n" + "=" * 100)
    print("SISTEMAS DINAMICOS NO HOMOGENEOS 2D".center(100))
    print("=" * 100)

    for caso in casos:
        print(f"\n{caso['nombre']}")
        print("-" * 100)
        resultado = resolver_sistema_dinamico_no_homogeneo(
            caso["A"],
            caso["f_t"],
            caso["X0_list"],
            caso["t_span"],
            caso["t_eval"],
        )

        print("Matriz A:")
        print(resultado["A"])
        print("Autovalores:")
        for idx, valor in enumerate(resultado["eigvals"], start=1):
            print(f"  lambda_{idx} = {_formatear_numero(valor)}")
        print("Autovectores (columnas de P):")
        print(resultado["eigvecs"])
        print(f"Clasificacion: {resultado['clasificacion']}")

        if resultado["forzante_constante"]:
            print(f"Forzante constante: {_formatear_vector(resultado['forzante_constante_vector'])}")
            if resultado["equilibrio_desplazado"] is not None:
                print(f"Equilibrio desplazado X_p = {_formatear_vector(resultado['equilibrio_desplazado'])}")
            else:
                print("No se pudo calcular equilibrio desplazado porque det(A) = 0")
        else:
            print("Forzante variable: no existe equilibrio desplazado constante")

        for idx, solucion in enumerate(resultado["trayectorias"], start=1):
            estado_final = solucion.y[:, -1]
            print(f"Trayectoria {idx}: estado final = {_formatear_vector(estado_final)}")

        plt.show()
        plt.close(resultado["fig"])


CONFIGURACION_BASE = {
    "A": np.array([[0.0, -1.0], [-9.0, 0.0]], dtype=float),
    "f_t": lambda t: np.array([1.0, 0.0], dtype=float),
    "X0_list": [(1.0, 0.0), (0.5, 0.5), (-2.0, 1.0)],
    "t_span": (0.0, 10.0),
    "t_eval": np.linspace(0.0, 10.0, 600),
}


if __name__ == "__main__":
    ejecutar_sistema_dinamico_no_homogeneo()