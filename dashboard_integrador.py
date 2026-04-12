import io
import contextlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sympy as sp

from Metodos.input_parser import build_numeric_function
from Metodos.metodo_aitken import metodo_aitken
from Metodos.metodo_de_biseccion import metodo_biseccion
from Metodos.metodo_lagrange_derivacion import (
    aproximar_derivada_tres_formas,
    polinomio_lagrange,
    polinomio_newton_desde_dd,
    tabla_diferencias_divididas,
)
from Metodos.metodo_newton_raphson import metodo_newton_raphson
from Metodos.metodo_punto_fijo import metodo_punto_fijo
from Metodos.metodo_integracion_numerica import (
    evaluar_funcion_robusta,
    regla_rectangulo,
    regla_trapecio,
    regla_simpson_13,
    regla_simpson_38,
    regla_montecarlo,
    regla_montecarlo_2d,
)
from Metodos.metodo_ajuste_curvas import regresion_lineal, regresion_polinomial
from Metodos.metodo_sistemas_lineales import gauss_jordan, gauss_seidel
from Metodos.metodo_edo import metodo_euler, metodo_rk4, construir_funcion_edo
from Metodos.metodo_red_neuronal_descenso_gradiente import (
    dataset_prueba_pequeno,
    entrenar_descenso_gradiente_lineal,
    figura_tres_subplots_descenso,
)


ALLOWED_LOCALS = {
    "e": sp.E,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


MACHETES_TEORICOS = {
    "Newton-Raphson": {
        "definicion": "Método iterativo que usa la derivada de f para encontrar raíces utilizando rectas tangentes.",
        "utilidad": "Hallar raíces con convergencia cuadrática rápida. Ideal para ecuaciones no lineales complejas.",
        "pasos": [
            "1. Elegir x_0 inicial",
            "2. Calcular f(x_n) y f'(x_n)",
            "3. Calcular nuevo valor: x_(n+1) = x_n - f(x_n)/f'(x_n)",
            "4. Calcular error: |x_(n+1) - x_n|",
            "5. Si error < tol => convergió",
            "6. Si no => repetir con x_(n+1)"
        ],
        "formulas": [
            r"$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$",
            r"$\text{Error: } e_n = |x_{n+1} - x_n|$"
        ],
        "requisitos": [
            "• f(x) y f'(x) calculables",
            "• f'(x_0) != 0",
            "• x_0 cercano a la raiz",
            "• Maximo de iteraciones"
        ]
    },
    "Aitken": {
        "definicion": "Técnica de aceleración que usa tres iteraciones sucesivas de punto fijo para obtener mejor aproximación sin cálculos adicionales.",
        "utilidad": "Mejorar velocidad de convergencia del método de punto fijo. Reduce iteraciones necesarias sin derivadas.",
        "pasos": [
            "1. Calcular x_0, x_1, x_2 usando punto fijo",
            "2. Aplicar fórmula de aceleración",
            "3. Usar x* como nueva aproximación",
            "4. Calcular error",
            "5. Si error < tol => convergió",
            "6. Si no => iterar nuevamente"
        ],
        "formulas": [
            r"$x^* = x_0 - \frac{(x_1 - x_0)^2}{x_2 - 2x_1 + x_0}$",
            r"$e_n = |x^* - x_0|$"
        ],
        "requisitos": [
            "• Punto fijo convergente",
            "• Denominador (x_2 - 2*x_1 + x_0) != 0",
            "• Tres iteraciones iniciales disponibles"
        ]
    },
    "Biseccion": {
        "definicion": "Método para hallar raíces de f(x)=0 dividiendo repetidamente un intervalo en dos mitades, descartando la que no contiene la raíz.",
        "utilidad": "Encontrar raíces de ecuaciones no lineales cuando f es continua. Garantiza convergencia si f(a)*f(b) < 0.",
        "pasos": [
            "1. Verificar que f(a)*f(b) < 0 (signos opuestos)",
            "2. Calcular punto medio: c = (a+b)/2",
            "3. Evaluar f(c)",
            "4. Si |f(c)| < tol => convergió",
            "5. Si f(a)*f(c) < 0 => nueva región [a,c]",
            "6. Si no => nueva región [c,b]",
            "7. Repetir hasta convergencia"
        ],
        "formulas": [
            r"$c = \frac{a + b}{2}$",
            r"$e_n = |c_{\text{nuevo}} - c_{\text{anterior}}|$"
        ],
        "requisitos": [
            "• f(x) continua en [a,b]",
            "• f(a)*f(b) < 0 (signos opuestos)",
            "• Tolerancia > 0",
            "• Maximo de iteraciones definido"
        ]
    },
    "Punto Fijo": {
        "definicion": "Método iterativo que transforma f(x)=0 en x=g(x) y aplica iteraciones para converger a un punto fijo de g.",
        "utilidad": "Hallar raíces reescribiendo la ecuación como x = g(x). Útil cuando la transformación es más simple que Newton-Raphson.",
        "pasos": [
            "1. Transformar f(x)=0 en x = g(x)",
            "2. Elegir x_0 inicial",
            "3. Iterar: x_(n+1) = g(x_n)",
            "4. Calcular error: |x_(n+1) - x_n|",
            "5. Si error < tol => convergió",
            "6. Si no => repetir paso 3"
        ],
        "formulas": [
            r"$x_{n+1} = g(x_n)$",
            r"$e_n = |x_{n+1} - x_n|$"
        ],
        "requisitos": [
            "• |g'(x)| < 1 en región de convergencia",
            "• g continua",
            "• x_0 suficientemente cercano a la raiz",
            "• Maximo de iteraciones"
        ]
    },
    "Lagrange + Derivacion": {
        "definicion": "Interpolación polinomial que construye un polinomio de grado n que pasa por n+1 puntos dados, permitiendo estimar valores intermedios y derivadas.",
        "utilidad": "Interpolar valores entre puntos conocidos. Aproximar derivadas numéricamente. Reconstruir funciones a partir de datos discretos.",
        "pasos": [
            "1. Obtener n+1 puntos (x_i, y_i) distintos en x",
            "2. Construir bases de Lagrange L_i(x)",
            "3. Polinomio: P(x) = suma de y_i * L_i(x)",
            "4. Evaluar P(x) en puntos deseados",
            "5. Derivar P(x) para obtener velocidades"
        ],
        "formulas": [
            r"$L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$",
            r"$P(x) = \sum_{i=0}^{n} y_i \cdot L_i(x)$"
        ],
        "requisitos": [
            "• Minimo 2 puntos",
            "• Puntos con x distintos",
            "• Valores y finitos",
            "• Se asume continuidad entre puntos"
        ]
    },
    "Integracion Numerica": {
        "definicion": "Aproximar integral definida ∫_a^b f(x)dx mediante sumas de áreas de figuras geométricas simples.",
        "utilidad": "Integrar cuando hay solución analítica compleja. Integrar datos tabulados sin función explícita.",
        "pasos": [
            "1. Dividir [a,b] en n subintervalos de ancho h = (b-a)/n",
            "2. Evaluar f en puntos de la malla",
            "3. Aplicar regla específica (rectángulos/trapecios/Simpson)",
            "4. Sumar para obtener integral aproximada"
        ],
        "formulas": [
            r"$I \approx h \sum_{i} f(x_i) \quad \text{(Rectángulos)}$",
            r"$I \approx \frac{h}{2} [f(x_0) + 2\sum f(x_i) + f(x_n)] \quad \text{(Trapecios)}$",
            r"$I \approx \frac{h}{3} [f(x_0) + 4\sum f(x_{\text{impar}}) + 2\sum f(x_{\text{par}}) + f(x_n)] \quad \text{(Simpson 1/3)}$"
        ],
        "requisitos": [
            "• a < b",
            "• n > 0 (subintervalos)",
            "• f continua en [a,b]",
            "• n par (para Simpson)"
        ]
    },
    "Ajuste de Curvas": {
        "definicion": "Técnica para encontrar una función que mejor aproxima un conjunto de datos discretos.",
        "utilidad": "Ajuste lineal: y = mx + b (relaciones lineales). Ajuste polinomial: y = a_0 + a_1*x + a_2*x^2 + ... (relaciones curvas).",
        "pasos": [
            "1. Organizar datos (x, y)",
            "2. Elegir modelo (lineal/polinomial)",
            "3. Resolver sistema de ecuaciones normales (minimos cuadrados)",
            "4. Calcular coeficientes",
            "5. Obtener ecuación ajustada",
            "6. Calcular R^2 (bondad de ajuste)"
        ],
        "formulas": [
            r"$\hat{y} = mx + b \quad \text{(lineal)}$",
            r"$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$"
        ],
        "requisitos": [
            "• Minimo 2 puntos (lineal), n+1 puntos (grado n)",
            "• Puntos con x distintos",
            "• Valores finitos"
        ]
    },
    "Sistemas Lineales": {
        "definicion": "Métodos para resolver sistemas de ecuaciones lineales de la forma Ax=b. Gauss-Jordan (directo) y Gauss-Seidel (iterativo).",
        "utilidad": "Resolver sistemas de ecuaciones lineales n×n.",
        "pasos": [
            "Gauss-Jordan: 1) Matriz aumentada [A|b] 2) Pivoteo 3) Normalización 4) Matriz identidad => solución",
            "Gauss-Seidel: 1) x_0 inicial 2) Actualizar cada x_i usando valores nuevos y viejos 3) Iterar hasta convergencia"
        ],
        "formulas": [
            r"$A \mathbf{x} = \mathbf{b} \quad \text{(sistema lineal)}$",
            r"$x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\right)$"
        ],
        "requisitos": [
            "• Gauss-Jordan: Matriz A cuadrada, Det(A) != 0",
            "• Gauss-Seidel: A diagonalmente dominante"
        ]
    },
    "EDO": {
        "definicion": "Métodos para resolver ecuaciones diferenciales ordinarias dy/dx = f(x,y) numéricamente a partir de condición inicial.",
        "utilidad": "Resolver EDO sin solución analítica. Modelar sistemas dinámicos (física, biología, economía).",
        "pasos": [
            "Euler: 1) Condición inicial (x_0, y_0) 2) y_(i+1) = y_i + h*f(x_i, y_i) 3) Completar n iteraciones",
            "RK4: 1) Calcular k_1, k_2, k_3, k_4 en puntos estratégicos 2) y_(i+1) = y_i + (h/6)*(k_1+2*k_2+2*k_3+k_4)"
        ],
        "formulas": [
            r"$y_{i+1} = y_i + h \cdot f(x_i, y_i) \quad \text{(Euler)}$",
            r"$y_{i+1} = y_i + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4) \quad \text{(RK4)}$"
        ],
        "requisitos": [
            "• f(x,y) calculable",
            "• h > 0 (tamanio paso)",
            "• n > 0 (iteraciones)",
            "• RK4 es mas preciso que Euler"
        ]
    },
    "Red Neuronal GD": {
        "definicion": "Algoritmo de optimización iterativo que ajusta parámetros (w, b) moviéndose en dirección opuesta al gradiente para minimizar el costo.",
        "utilidad": "Entrenar modelos lineales y = w*x + b. Base de métodos de aprendizaje automático. Minimizar función objetivo (MSE, pérdida).",
        "pasos": [
            "1. Inicializar w y b aleatoriamente",
            "2. Para cada época: a) Calcular predicciones y_hat = w*x + b",
            "   b) Calcular costo J = (1/m)*suma(y_hat - y)^2",
            "   c) Calcular gradientes dJ/dw, dJ/db",
            "   d) Actualizar w := w - alpha*dJ/dw",
            "   e) Actualizar b := b - alpha*dJ/db",
            "3. Registrar historial"
        ],
        "formulas": [
            r"$J = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$",
            r"$\frac{\partial J}{\partial w} = \frac{2}{m} \sum_{i} (\hat{y}_i - y_i) \cdot x_i$",
            r"$w := w - \alpha \cdot \frac{\partial J}{\partial w}$"
        ],
        "requisitos": [
            "• alpha (learning rate) > 0 y pequeño (0.01-0.1)",
            "• Epocas > 0",
            "• Datos normalizados o centrados (recomendado)",
            "• Inicialización de semilla para reproducibilidad"
        ]
    }
}


def mostrar_machete(metodo_nombre):
    """Muestra el machete teórico de un método en un expander."""
    if metodo_nombre not in MACHETES_TEORICOS:
        return
    
    machete = MACHETES_TEORICOS[metodo_nombre]
    with st.expander(f"📚 Machete Teórico: {metodo_nombre}", expanded=False):
        st.markdown(f"**DEFINICIÓN**  \n{machete['definicion']}")
        st.markdown(f"**UTILIDAD**  \n{machete['utilidad']}")
        
        st.markdown("**PASOS**")
        for paso in machete['pasos']:
            st.markdown(f"- {paso}")
        
        st.markdown("**FÓRMULAS**")
        for formula in machete['formulas']:
            # Remover los signos $ al inicio y final si existen
            formula_clean = formula.strip()
            if formula_clean.startswith("$") and formula_clean.endswith("$"):
                formula_clean = formula_clean[1:-1]
            elif formula_clean.startswith(r"$") and formula_clean.endswith(r"$"):
                formula_clean = formula_clean[1:-1]
            st.latex(formula_clean)
        
        st.markdown("**REQUISITOS**")
        for req in machete['requisitos']:
            st.markdown(f"- {req}")


PALETAS_DASHBOARD = {
    "Viva": {
        "accent_1": "#00E5FF",
        "accent_2": "#FF6B35",
        "accent_3": "#9BFF00",
        "series": [
            "#00E5FF",
            "#FF6B35",
            "#9BFF00",
            "#FFD400",
            "#FF3D81",
            "#7A7CFF",
            "#00C16E",
            "#FF8A00",
        ],
    },
    "Pastel oscuro": {
        "accent_1": "#7AD3FF",
        "accent_2": "#FFAE8A",
        "accent_3": "#C9F7A5",
        "series": [
            "#7AD3FF",
            "#FFAE8A",
            "#C9F7A5",
            "#FFE59A",
            "#F6A5C0",
            "#B8B7FF",
            "#9EE7C9",
            "#FFD3A8",
        ],
    },
}


def paleta_activa():
    preset = st.session_state.get("palette_preset", "Viva")
    return PALETAS_DASHBOARD.get(preset, PALETAS_DASHBOARD["Viva"])


def aplicar_tema_visual_dashboard(paleta):
    """Mejora contraste y distincion de colores sin modificar el fondo negro general."""
    css = """
        <style>
        :root {
            --accent-1: __ACCENT1__;
            --accent-2: __ACCENT2__;
            --accent-3: __ACCENT3__;
            --line-soft: rgba(255,255,255,0.22);
        }

        .stTabs [data-baseweb="tab-list"] button[role="tab"] {
            border: 1px solid var(--line-soft);
            border-radius: 10px;
            margin-right: 6px;
            padding: 8px 12px;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-color: var(--accent-1);
            box-shadow: 0 0 0 1px var(--accent-1) inset;
            color: #FFFFFF;
            background: rgba(255,255,255,0.08);
        }

        .stButton > button {
            border: 1px solid var(--accent-1);
            color: #FFFFFF;
        }

        .stButton > button:hover {
            border-color: var(--accent-2);
            box-shadow: 0 0 0 1px var(--accent-2) inset;
        }

        [data-testid="stMetric"] {
            border: 1px solid var(--line-soft);
            border-radius: 12px;
            padding: 8px 10px;
            background: rgba(255,255,255,0.02);
        }
        </style>
    """
    css = css.replace("__ACCENT1__", paleta["accent_1"])
    css = css.replace("__ACCENT2__", paleta["accent_2"])
    css = css.replace("__ACCENT3__", paleta["accent_3"])
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def run_silent(func, *args, **kwargs):
    """Ejecuta funciones de metodos que imprimen en consola, silenciando su salida."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return func(*args, **kwargs)


def safe_eval_expr(expr_text, variable="x"):
    x = sp.Symbol(variable)
    expr = sp.sympify(expr_text, locals=ALLOWED_LOCALS)
    invalid = expr.free_symbols - {x}
    if invalid:
        raise ValueError(f"Variables no permitidas: {invalid}")
    return expr


def parse_numeric_expr(expr_text, field_name, allow_infinite=False):
    try:
        expr = sp.sympify(expr_text, locals=ALLOWED_LOCALS)
    except Exception as exc:
        raise ValueError(f"{field_name} invalido: {exc}") from exc

    if expr.free_symbols:
        raise ValueError(f"{field_name} no debe contener variables")

    if allow_infinite:
        if expr in {sp.oo, sp.zoo}:
            return np.inf
        if expr == -sp.oo:
            return -np.inf

    value = float(sp.N(expr))
    if not np.isfinite(value):
        if allow_infinite and np.isinf(value):
            return value
        raise ValueError(f"{field_name} debe ser un numero finito")
    return value


def estimate_max_abs_derivative(expr, variable, order, a, b, points=2001):
    deriv = sp.diff(expr, variable, order)
    f_deriv = sp.lambdify(variable, deriv, modules=["numpy"])
    x_grid = np.linspace(float(a), float(b), int(points))
    vals = np.array(f_deriv(x_grid), dtype=float)
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return None
    return float(np.max(np.abs(vals[finite_mask])))


def cota_truncamiento_integracion(nombre_metodo, a, b, n, max_f2=None, max_f4=None):
    h = (float(b) - float(a)) / int(n)
    longitud = float(b) - float(a)

    if nombre_metodo == "Rectangulo":
        if max_f2 is None:
            return np.nan
        return (longitud / 24.0) * (h ** 2) * float(max_f2)

    if nombre_metodo == "Trapecio":
        if max_f2 is None:
            return np.nan
        return (longitud / 12.0) * (h ** 2) * float(max_f2)

    if nombre_metodo == "Simpson 1/3":
        if max_f4 is None or int(n) % 2 != 0:
            return np.nan
        return (longitud / 180.0) * (h ** 4) * float(max_f4)

    if nombre_metodo == "Simpson 3/8":
        if max_f4 is None or int(n) % 3 != 0:
            return np.nan
        return (longitud / 80.0) * (h ** 4) * float(max_f4)

    if nombre_metodo == "Monte Carlo":
        # Monte Carlo no tiene error de truncamiento determinista
        return np.nan

    return np.nan


def detalle_cota_truncamiento_integracion(nombre_metodo, a, b, n, max_f2=None, max_f4=None):
    a = float(a)
    b = float(b)
    n = int(n)
    h = (b - a) / n
    longitud = b - a

    detalle = {
        "ok": False,
        "cota": np.nan,
        "pasos": [
            f"1) h = (b-a)/n = ({b:.7g} - {a:.7g})/{n} = {h:.7g}",
            f"2) Longitud del intervalo: (b-a) = {longitud:.7g}",
        ],
        "latex_formula": None,
        "latex_sustitucion": None,
    }

    if nombre_metodo == "Rectangulo":
        if max_f2 is None:
            detalle["pasos"].append("3) No se pudo estimar max|f''(x)| en el intervalo.")
            return detalle
        detalle["pasos"].append(f"3) max|f''(x)| ≈ {float(max_f2):.7g}")
        detalle["latex_formula"] = r"|E_T| \leq \frac{(b-a)}{24}h^2\max_{x\in[a,b]}|f''(x)|"
        detalle[
            "latex_sustitucion"
        ] = rf"|E_T| \leq \frac{{{longitud:.7g}}}{{24}}({h:.7g})^2({float(max_f2):.7g})"

    elif nombre_metodo == "Trapecio":
        if max_f2 is None:
            detalle["pasos"].append("3) No se pudo estimar max|f''(x)| en el intervalo.")
            return detalle
        detalle["pasos"].append(f"3) max|f''(x)| ≈ {float(max_f2):.7g}")
        detalle["latex_formula"] = r"|E_T| \leq \frac{(b-a)}{12}h^2\max_{x\in[a,b]}|f''(x)|"
        detalle[
            "latex_sustitucion"
        ] = rf"|E_T| \leq \frac{{{longitud:.7g}}}{{12}}({h:.7g})^2({float(max_f2):.7g})"

    elif nombre_metodo == "Simpson 1/3":
        if n % 2 != 0:
            detalle["pasos"].append("3) Simpson 1/3 requiere n par.")
            return detalle
        if max_f4 is None:
            detalle["pasos"].append("3) No se pudo estimar max|f^(4)(x)| en el intervalo.")
            return detalle
        detalle["pasos"].append(f"3) max|f^(4)(x)| ≈ {float(max_f4):.7g}")
        detalle["latex_formula"] = r"|E_T| \leq \frac{(b-a)}{180}h^4\max_{x\in[a,b]}|f^{(4)}(x)|"
        detalle[
            "latex_sustitucion"
        ] = rf"|E_T| \leq \frac{{{longitud:.7g}}}{{180}}({h:.7g})^4({float(max_f4):.7g})"

    elif nombre_metodo == "Simpson 3/8":
        if n % 3 != 0:
            detalle["pasos"].append("3) Simpson 3/8 requiere n multiplo de 3.")
            return detalle
        if max_f4 is None:
            detalle["pasos"].append("3) No se pudo estimar max|f^(4)(x)| en el intervalo.")
            return detalle
        detalle["pasos"].append(f"3) max|f^(4)(x)| ≈ {float(max_f4):.7g}")
        detalle["latex_formula"] = r"|E_T| \leq \frac{(b-a)}{80}h^4\max_{x\in[a,b]}|f^{(4)}(x)|"
        detalle[
            "latex_sustitucion"
        ] = rf"|E_T| \leq \frac{{{longitud:.7g}}}{{80}}({h:.7g})^4({float(max_f4):.7g})"

    elif nombre_metodo == "Monte Carlo":
        detalle["pasos"].append("3) Monte Carlo es un metodo estocastico; no aplica cota de truncamiento determinista.")
        return detalle

    else:
        detalle["pasos"].append("3) Metodo no reconocido para cota de truncamiento.")
        return detalle

    cota = cota_truncamiento_integracion(nombre_metodo, a, b, n, max_f2=max_f2, max_f4=max_f4)
    detalle["cota"] = cota
    detalle["ok"] = bool(np.isfinite(cota))
    detalle["pasos"].append(f"4) Cota final: |E_T| <= {float(cota):.7g}" if np.isfinite(cota) else "4) No se pudo calcular la cota final.")
    return detalle


def integrar_numerica_soporte_infinito(funcion_str, a, b, n, metodo, eps=1e-6):
    n = int(n)
    a = float(a)
    b = float(b)

    if np.isfinite(a) and np.isfinite(b):
        if metodo == "Rectangulo":
            valor, x_nodes, y_nodes = regla_rectangulo(funcion_str, a, b, n)
        elif metodo == "Trapecio":
            valor, x_nodes, y_nodes = regla_trapecio(funcion_str, a, b, n)
        elif metodo == "Simpson 1/3":
            valor, x_nodes, y_nodes = regla_simpson_13(funcion_str, a, b, n)
        elif metodo == "Simpson 3/8":
            valor, x_nodes, y_nodes = regla_simpson_38(funcion_str, a, b, n)
        else:
            raise ValueError("Metodo no valido")

        return {
            "valor": float(valor),
            "x_nodes": x_nodes,
            "y_nodes": y_nodes,
            "impropia": False,
            "u_nodes": None,
            "g_nodes": None,
            "tipo": "finita",
        }

    if np.isfinite(a) and np.isposinf(b):
        tipo = "[a, +infinito)"
        u0, u1 = 0.0, 1.0 - eps
        map_x = lambda u: a + (u / (1.0 - u))
        jac = lambda u: 1.0 / ((1.0 - u) ** 2)
    elif np.isneginf(a) and np.isfinite(b):
        tipo = "(-infinito, b]"
        u0, u1 = 0.0, 1.0 - eps
        map_x = lambda u: b - (u / (1.0 - u))
        jac = lambda u: 1.0 / ((1.0 - u) ** 2)
    elif np.isneginf(a) and np.isposinf(b):
        tipo = "(-infinito, +infinito)"
        u0, u1 = eps, 1.0 - eps
        map_x = lambda u: np.tan(np.pi * (u - 0.5))
        jac = lambda u: np.pi / (np.cos(np.pi * (u - 0.5)) ** 2)
    else:
        raise ValueError("Combinacion de limites no soportada")

    if metodo == "Simpson 1/3" and n % 2 != 0:
        raise ValueError("Simpson 1/3 requiere n par")
    if metodo == "Simpson 3/8" and n % 3 != 0:
        raise ValueError("Simpson 3/8 requiere n multiplo de 3")

    def _integrar_desde_valores(y_vals, h_local, nombre_metodo):
        if nombre_metodo == "Rectangulo":
            return h_local * np.sum(y_vals)
        if nombre_metodo == "Trapecio":
            return h_local * (0.5 * y_vals[0] + np.sum(y_vals[1:-1]) + 0.5 * y_vals[-1])
        if nombre_metodo == "Simpson 1/3":
            return (h_local / 3.0) * (
                y_vals[0] + y_vals[-1] + 4.0 * np.sum(y_vals[1:-1:2]) + 2.0 * np.sum(y_vals[2:-1:2])
            )
        if nombre_metodo == "Simpson 3/8":
            pesos = np.ones_like(y_vals)
            pesos[1:-1] = 3
            pesos[3:-1:3] = 2
            return (3.0 * h_local / 8.0) * np.sum(pesos * y_vals)
        raise ValueError("Metodo no valido")

    u_nodes = np.linspace(u0, u1, n + 1)
    x_nodes = map_x(u_nodes)
    fx_nodes = np.array(evaluar_funcion_robusta(funcion_str, x_nodes), dtype=float)
    g_nodes = fx_nodes * jac(u_nodes)

    if not np.all(np.isfinite(g_nodes)):
        raise ValueError("La transformacion impropia produjo valores no finitos")

    h_u = (u1 - u0) / n
    if metodo == "Rectangulo":
        u_mid = u0 + (np.arange(n) + 0.5) * h_u
        x_mid = map_x(u_mid)
        fx_mid = np.array(evaluar_funcion_robusta(funcion_str, x_mid), dtype=float)
        g_mid = fx_mid * jac(u_mid)
        valor = _integrar_desde_valores(g_mid, h_u, metodo)
    else:
        valor = _integrar_desde_valores(g_nodes, h_u, metodo)

    return {
        "valor": float(valor),
        "x_nodes": x_nodes,
        "y_nodes": fx_nodes,
        "impropia": True,
        "u_nodes": u_nodes,
        "g_nodes": g_nodes,
        "tipo": tipo,
    }


def plot_integracion_impropia_transformada(u_nodes, g_nodes, metodo, tipo):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(u_nodes, g_nodes, linewidth=2, label="Integrando transformado g(u)")
    ax.fill_between(u_nodes, g_nodes, 0.0, alpha=0.2, label="Area en dominio transformado")
    ax.set_title(f"Integracion impropia por {metodo} ({tipo})")
    ax.set_xlabel("u (variable transformada)")
    ax.set_ylabel("g(u)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_integracion_visual(funcion_str, a, b, n, metodo):
    x_nodes = np.linspace(float(a), float(b), int(n) + 1)
    y_nodes = np.array(evaluar_funcion_robusta(funcion_str, x_nodes), dtype=float)

    x_plot = np.linspace(float(a), float(b), 1200)
    y_plot = np.array(evaluar_funcion_robusta(funcion_str, x_plot), dtype=float)

    h = (float(b) - float(a)) / int(n)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x_plot, y_plot, linewidth=2, color="tab:blue", label=f"f(x) = {funcion_str}")

    if metodo == "Rectangulo":
        x_left = x_nodes[:-1]
        x_mid = x_left + 0.5 * h
        y_mid = np.array(evaluar_funcion_robusta(funcion_str, x_mid), dtype=float)

        ax.bar(
            x_left,
            y_mid,
            width=h,
            align="edge",
            alpha=0.25,
            color="tab:orange",
            edgecolor="tab:orange",
            linewidth=1.0,
            label="Rectangulos de aproximacion",
        )
        ax.scatter(x_mid, y_mid, color="tab:orange", s=20, zorder=3, label="Puntos medios")

    elif metodo == "Trapecio":
        for i in range(int(n)):
            xi = x_nodes[i]
            xi1 = x_nodes[i + 1]
            yi = y_nodes[i]
            yi1 = y_nodes[i + 1]
            label = "Trapecios de aproximacion" if i == 0 else None
            ax.fill([xi, xi, xi1, xi1], [0.0, yi, yi1, 0.0], alpha=0.2, color="tab:green", edgecolor="tab:green", label=label)
        ax.plot(x_nodes, y_nodes, "o-", color="tab:green", linewidth=1.2, markersize=4, label="Nodos")

    elif metodo == "Simpson 1/3":
        if int(n) % 2 != 0:
            raise ValueError("Simpson 1/3 requiere n par")
        for i in range(0, int(n), 2):
            x_block = x_nodes[i : i + 3]
            y_block = y_nodes[i : i + 3]
            coef = np.polyfit(x_block, y_block, 2)
            x_local = np.linspace(x_block[0], x_block[-1], 120)
            y_local = np.polyval(coef, x_local)
            label = "Parabolas de Simpson 1/3" if i == 0 else None
            ax.plot(x_local, y_local, color="tab:purple", linewidth=1.3, alpha=0.95, label=label)
            ax.fill_between(x_local, y_local, 0.0, color="tab:purple", alpha=0.18)
        ax.scatter(x_nodes, y_nodes, color="tab:purple", s=22, zorder=3, label="Nodos")

    elif metodo == "Simpson 3/8":
        if int(n) % 3 != 0:
            raise ValueError("Simpson 3/8 requiere n multiplo de 3")
        for i in range(0, int(n), 3):
            x_block = x_nodes[i : i + 4]
            y_block = y_nodes[i : i + 4]
            coef = np.polyfit(x_block, y_block, 3)
            x_local = np.linspace(x_block[0], x_block[-1], 140)
            y_local = np.polyval(coef, x_local)
            label = "Cubicas de Simpson 3/8" if i == 0 else None
            ax.plot(x_local, y_local, color="tab:brown", linewidth=1.3, alpha=0.95, label=label)
            ax.fill_between(x_local, y_local, 0.0, color="tab:brown", alpha=0.18)
        ax.scatter(x_nodes, y_nodes, color="tab:brown", s=22, zorder=3, label="Nodos")

    else:
        raise ValueError("Metodo no valido")

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(f"Integracion por {metodo}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def parse_expr_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Debes ingresar al menos un valor")
    return [sp.sympify(p, locals=ALLOWED_LOCALS) for p in parts]


def to_float_array(values):
    return np.array([float(sp.N(v)) for v in values], dtype=float)


def parse_matrix_text(text, n):
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if len(rows) != n:
        raise ValueError(f"La matriz debe tener exactamente {n} filas")

    out = []
    for i, row in enumerate(rows):
        parts = [p.strip() for p in row.split(",") if p.strip()]
        if len(parts) != n:
            raise ValueError(f"La fila {i + 1} debe tener exactamente {n} coeficientes")
        out.append([float(sp.N(sp.sympify(p, locals=ALLOWED_LOCALS))) for p in parts])

    return np.array(out, dtype=float)


def parse_vector_text(text, n):
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if len(rows) != n:
        raise ValueError(f"El vector b debe tener exactamente {n} valores")
    vals = [float(sp.N(sp.sympify(p, locals=ALLOWED_LOCALS))) for p in rows]
    return np.array(vals, dtype=float)


def parse_optional_vector_csv(text, n):
    if not text.strip():
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"x0 debe tener exactamente {n} valores")
    return np.array([float(sp.N(sp.sympify(p, locals=ALLOWED_LOCALS))) for p in parts], dtype=float)


def _format_decimal(value, max_decimals=7):
    txt = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return "0" if txt in {"", "-0"} else txt


def _is_terminating_denominator(den):
    """Un racional termina en decimal solo si su denominador tiene factores 2 y/o 5."""
    d = abs(int(den))
    if d == 0:
        return False
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1


def _format_number_hybrid(value_expr, max_decimals=7, latex=False):
    """
    Formato mixto:
    - Fraccion para racionales pequenos o decimales periodicos/no terminantes.
    - Decimal para enteros y racionales con decimal terminante "comodo".
    """
    v = sp.nsimplify(value_expr, rational=True)

    if v.is_Rational:
        v_rat = sp.Rational(v)
        den = int(v_rat.q)
        num = int(v_rat.p)

        if den == 1:
            return str(num)

        use_fraction = (den <= 20) or (not _is_terminating_denominator(den))
        if use_fraction:
            if latex:
                return f"\\frac{{{num}}}{{{den}}}"
            return f"{num}/{den}"

        return _format_decimal(float(v_rat), max_decimals)

    return _format_decimal(float(sp.N(v)), max_decimals)


def polynomial_to_decimal_text(expr, variable="x", max_decimals=7):
    x = sp.Symbol(variable)
    expr = sp.expand(expr)

    try:
        poly = sp.Poly(expr, x)
    except sp.PolynomialError:
        return _format_number_hybrid(expr, max_decimals=max_decimals, latex=False)

    terms = []
    eps = 10 ** (-(max_decimals + 2))

    for (power,), coef in sorted(poly.terms(), key=lambda t: t[0][0], reverse=True):
        coef_expr = sp.nsimplify(coef, rational=True)
        coef_val = float(sp.N(coef_expr))
        if abs(coef_val) < eps:
            continue

        sign = "-" if coef_val < 0 else "+"
        coef_abs_expr = abs(coef_expr)
        coef_abs_txt = _format_number_hybrid(coef_abs_expr, max_decimals=max_decimals, latex=False)
        is_one = sp.simplify(coef_abs_expr - 1) == 0

        if power == 0:
            body = coef_abs_txt
        elif power == 1:
            body = variable if is_one else f"{coef_abs_txt}{variable}"
        else:
            body = f"{variable}^{power}" if is_one else f"{coef_abs_txt}{variable}^{power}"

        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    out = f"-{first_body}" if first_sign == "-" else first_body

    for sign, body in terms[1:]:
        out += f" {sign} {body}"

    return out


def polynomial_to_decimal_latex(expr, variable="x", max_decimals=7):
    x = sp.Symbol(variable)
    expr = sp.expand(expr)

    try:
        poly = sp.Poly(expr, x)
    except sp.PolynomialError:
        return _format_number_hybrid(expr, max_decimals=max_decimals, latex=True)

    terms = []
    eps = 10 ** (-(max_decimals + 2))

    for (power,), coef in sorted(poly.terms(), key=lambda t: t[0][0], reverse=True):
        coef_expr = sp.nsimplify(coef, rational=True)
        coef_val = float(sp.N(coef_expr))
        if abs(coef_val) < eps:
            continue

        sign = "-" if coef_val < 0 else "+"
        coef_abs_expr = abs(coef_expr)
        coef_abs_txt = _format_number_hybrid(coef_abs_expr, max_decimals=max_decimals, latex=True)
        is_one = sp.simplify(coef_abs_expr - 1) == 0

        if power == 0:
            body = coef_abs_txt
        elif power == 1:
            body = variable if is_one else f"{coef_abs_txt}{variable}"
        else:
            body = f"{variable}^{{{power}}}" if is_one else f"{coef_abs_txt}{variable}^{{{power}}}"

        terms.append((sign, body))

    if not terms:
        return "0"

    first_sign, first_body = terms[0]
    out = f"-{first_body}" if first_sign == "-" else first_body

    for sign, body in terms[1:]:
        out += f" {sign} {body}"

    return out


def build_func_plot(expr_text, x_values):
    _, f_num = build_numeric_function(expr_text)
    y = np.array(f_num(x_values), dtype=float)
    return y


def plot_function_with_root(func_text, root, xmin, xmax, title):
    x = np.linspace(xmin, xmax, 800)
    y = build_func_plot(func_text, x)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, y, linewidth=2, label=f"f(x) = {func_text}")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.plot([root], [0], "ro", markersize=8, label=f"Raiz aprox: {root:.8g}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_error_curve(errors, title, y_label="Error absoluto"):
    idx = np.arange(1, len(errors) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))

    errors = np.array(errors, dtype=float)
    valid = np.all(errors > 0)

    if valid:
        ax.semilogy(idx, errors, "o-", linewidth=2)
    else:
        ax.plot(idx, errors, "o-", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Iteracion")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3, which="both")
    return fig


def sugerir_x0_primera_raiz_positiva(func_text, x0_usuario, muestras=4000):
    """Busca un cambio de signo en x>0 para proponer un x0 cercano a la primera raiz positiva."""
    _, f_num = build_numeric_function(func_text)

    x_min = 1e-6
    x_max = max(10.0, abs(float(x0_usuario)) * 2.0)
    x_grid = np.linspace(x_min, x_max, int(muestras))

    try:
        y_grid = np.array(f_num(x_grid), dtype=float)
    except Exception:
        return None

    for i in range(len(x_grid) - 1):
        yi = y_grid[i]
        yj = y_grid[i + 1]
        if not (np.isfinite(yi) and np.isfinite(yj)):
            continue

        if abs(yi) < 1e-8:
            if x_grid[i] > x_min:
                return float(x_grid[i])

        if yi * yj < 0:
            return float(0.5 * (x_grid[i] + x_grid[i + 1]))

    return None


def render_chart(fig):
    """Renderiza Matplotlib en modo interactivo (Plotly) cuando esta disponible."""
    interactive = st.session_state.get("interactive_charts", True)
    paleta_series = paleta_activa()["series"]

    # Refuerza contraste en figuras Matplotlib antes de convertir/renderizar.
    for ax in fig.get_axes():
        ax.tick_params(axis="both", colors="black", labelcolor="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_alpha(1.0)
            for txt in legend.get_texts():
                txt.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")

        # Resalta ejes en cero para mejorar visibilidad.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if ylim[0] <= 0 <= ylim[1]:
            ax.axhline(0, color="black", linewidth=2.2, alpha=0.95, zorder=4)
        if xlim[0] <= 0 <= xlim[1]:
            ax.axvline(0, color="black", linewidth=2.2, alpha=0.95, zorder=4)

    # Evita que etiquetas y leyendas queden recortadas en el render final.
    fig.tight_layout()

    # mpl_to_plotly suele deformar heatmaps/imshow (ejes gigantes o imagen invisible).
    has_image_axes = any(len(ax.images) > 0 for ax in fig.get_axes())

    if interactive and not has_image_axes:
        try:
            mpl_to_plotly = __import__("plotly.tools", fromlist=["mpl_to_plotly"]).mpl_to_plotly

            fig_plotly = mpl_to_plotly(fig)
            fig_plotly.update_layout(
                margin=dict(l=18, r=18, t=60, b=28),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                template="plotly_white",
                colorway=paleta_series,
                legend=dict(font=dict(color="black"), title=dict(font=dict(color="black"))),
            )
            fig_plotly.update_xaxes(
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
                showline=True,
                linecolor="black",
                zeroline=True,
                zerolinewidth=2.2,
                zerolinecolor="black",
                gridcolor="rgba(0,0,0,0.15)",
            )
            fig_plotly.update_yaxes(
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
                showline=True,
                linecolor="black",
                zeroline=True,
                zerolinewidth=2.2,
                zerolinecolor="black",
                gridcolor="rgba(0,0,0,0.15)",
            )
            st.plotly_chart(fig_plotly, use_container_width=True)
            return
        except Exception:
            if not st.session_state.get("_interactive_chart_warned", False):
                st.info(
                    "Algunos graficos no se pudieron convertir a interactivos. "
                    "Se mostraran en modo estatico en esos casos."
                )
                st.session_state["_interactive_chart_warned"] = True

    st.pyplot(fig, use_container_width=True)


def _build_newton_plotly_function(func_text, root, xmin, xmax, title):
    go = __import__("plotly.graph_objects", fromlist=["Figure"]) 
    paleta_series = paleta_activa()["series"]

    x = np.linspace(xmin, xmax, 800)
    y = build_func_plot(func_text, x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"f(x) = {func_text}",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[root],
            y=[0],
            mode="markers",
            name=f"Raiz aprox: {root:.8g}",
            marker=dict(color="#FF3D81", size=10),
        )
    )

    fig.add_hline(y=0, line_width=2.4, line_color="black", opacity=0.95)
    fig.add_vline(x=0, line_width=2.4, line_color="black", opacity=0.95)
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        margin=dict(l=18, r=18, t=60, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        template="plotly_white",
        colorway=paleta_series,
        legend=dict(font=dict(color="black"), title=dict(font=dict(color="black"))),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        color="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        color="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        showline=True,
        linecolor="black",
    )
    return fig


def _build_newton_plotly_error(errors, title, y_label="Error absoluto"):
    go = __import__("plotly.graph_objects", fromlist=["Figure"]) 
    paleta_series = paleta_activa()["series"]

    errors = np.array(errors, dtype=float)
    idx = np.arange(1, len(errors) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=errors,
            mode="lines+markers",
            name=y_label,
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Iteracion",
        yaxis_title=y_label,
        margin=dict(l=18, r=18, t=60, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        template="plotly_white",
        colorway=paleta_series,
        legend=dict(font=dict(color="black"), title=dict(font=dict(color="black"))),
    )
    if np.all(errors > 0):
        fig.update_yaxes(type="log")

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        color="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        showline=True,
        linecolor="black",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        color="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        showline=True,
        linecolor="black",
    )
    return fig


def render_newton_charts(func, root, x0, errors):
    """Newton usa graficos Plotly nativos para asegurar interactividad."""
    interactive = st.session_state.get("interactive_charts", True)

    if interactive:
        try:
            margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
            fig_f = _build_newton_plotly_function(
                func,
                float(root),
                float(root) - margin,
                float(root) + margin,
                "Funcion y raiz aproximada (Newton)",
            )
            st.plotly_chart(fig_f, use_container_width=True)

            fig_e = _build_newton_plotly_error(errors, "Error por iteracion (Newton)")
            st.plotly_chart(fig_e, use_container_width=True)
            return
        except Exception:
            pass

    margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
    fig_f = plot_function_with_root(
        func,
        float(root),
        float(root) - margin,
        float(root) + margin,
        "Funcion y raiz aproximada (Newton)",
    )
    st.pyplot(fig_f)
    plt.close(fig_f)

    fig_e = plot_error_curve(errors, "Error por iteracion (Newton)")
    st.pyplot(fig_e)
    plt.close(fig_e)


def mostrar_imagen_encabezado(nombre_archivo, ancho=120, texto_alternativo="Imagen no cargada"):
    """Muestra una imagen del encabezado si existe dentro de Metodos/."""
    shield_path = Path(__file__).resolve().parent / "Metodos" / nombre_archivo

    if shield_path.exists():
        st.image(str(shield_path), width=ancho)
    else:
        st.markdown(
            f"<div style='width:{ancho}px;height:{ancho}px;border:1px dashed rgba(255,255,255,0.35);"
            "border-radius:10px;display:flex;align-items:center;justify-content:center;"
            "color:rgba(255,255,255,0.65);font-size:12px;text-align:center;padding:8px;'>"
            f"{texto_alternativo}</div>",
            unsafe_allow_html=True,
        )


def render_panel_formulas(
    titulo,
    formulas,
    simbolos=None,
    condiciones=None,
    pasos=None,
    mostrar_pasos=False,
    desglose_completo=None,
    apartado_clave=None,
):
    """Muestra un panel estético con formulas, leyenda, condiciones y paso a paso opcional."""
    def _latex_expr(expr):
        if expr is None:
            return ""
        txt = str(expr).strip()
        if txt.startswith("$") and txt.endswith("$") and len(txt) >= 2:
            txt = txt[1:-1].strip()
        return txt

    def _latex_text(texto):
        txt = str(texto)
        txt = txt.replace("\\", r"\\")
        txt = txt.replace("{", r"\{").replace("}", r"\}")
        txt = txt.replace("_", r"\_")
        return r"\text{" + txt + "}"

    st.markdown("---")
    with st.container(border=True):
        st.markdown(f"### {titulo}")
        st.caption("Desarrollo matematico y condiciones del metodo")

        if simbolos:
            st.markdown("**Leyenda de simbolos**")
            for item in simbolos:
                st.markdown(f"- {item}")

        if condiciones:
            st.markdown("**Condiciones de aplicacion / convergencia**")
            for item in condiciones:
                st.markdown(f"- {item}")

        st.markdown("**Formulas**")
        for formula in formulas:
            st.latex(formula)

        if mostrar_pasos and pasos:
            st.markdown("**Paso a paso completo (letras y numeros)**")
            for i, paso in enumerate(pasos, start=1):
                st.markdown(f"**Paso {i}: {paso['titulo']}**")
                st.latex(_latex_expr(paso["simbolico"]))
                st.latex(_latex_text(paso["numerico"]))

            if desglose_completo:
                st.markdown("**Secuencia completa del procedimiento**")
                for i, linea in enumerate(desglose_completo, start=1):
                    st.latex(_latex_text(f"{i}) {linea}"))

        if mostrar_pasos and apartado_clave:
            cuentas = st.session_state.get("cuentas_paso_a_paso", {}).get(apartado_clave, [])
            st.markdown("**Cuentas realizadas (con tus datos)**")
            if cuentas:
                for linea in cuentas:
                    st.latex(linea)
            else:
                st.info("Ejecuta este metodo para ver aqui el paso a paso numerico real de las cuentas.")


def _num(v, dec=8):
    try:
        return f"{float(v):.{dec}g}"
    except Exception:
        return str(v)


def guardar_cuentas(apartado_clave, lineas):
    if "cuentas_paso_a_paso" not in st.session_state:
        st.session_state["cuentas_paso_a_paso"] = {}
    st.session_state["cuentas_paso_a_paso"][apartado_clave] = list(lineas)


FORMULAS_POR_APARTADO = {
    "Newton-Raphson": [
        r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}",
        r"e_n=|x_{n+1}-x_n|",
        r"\text{criterio: }e_n<\varepsilon",
    ],
    "Aitken": [
        r"x_{n+1}=g(x_n)",
        r"\hat{x}_n=x_n-\frac{(x_{n+1}-x_n)^2}{x_{n+2}-2x_{n+1}+x_n}",
        r"e_n=|\hat{x}_n-\hat{x}_{n-1}|",
    ],
    "Biseccion": [
        r"c_n=\frac{a_n+b_n}{2}",
        r"\text{si }f(a_n)f(c_n)<0\Rightarrow b_{n+1}=c_n,\ a_{n+1}=a_n",
        r"\text{si }f(a_n)f(c_n)>0\Rightarrow a_{n+1}=c_n,\ b_{n+1}=b_n",
        r"e_n\approx\frac{b_n-a_n}{2}",
    ],
    "Punto Fijo": [
        r"x_{n+1}=g(x_n)",
        r"e_n=|x_{n+1}-x_n|",
        r"\text{condicion local: }|g'(x^*)|<1",
    ],
    "Comparativa": [
        r"\text{Error relativo}=\frac{|x_{aprox}-x_{ref}|}{|x_{ref}|}",
        r"\text{Metrica de iteraciones: }N_{iter}",
        r"\text{Convergencia} \Rightarrow e_n<\varepsilon",
    ],
    "Lagrange + Derivacion": [
        r"P_n(x)=\sum_{i=0}^{n} y_i L_i(x)",
        r"L_i(x)=\prod_{j\neq i}\frac{x-x_j}{x_i-x_j}",
        r"f'(x)\approx\frac{f(x+h)-f(x)}{h},\ \frac{f(x)-f(x-h)}{h},\ \frac{f(x+h)-f(x-h)}{2h}",
        r"E(x)=|f(x)-P_n(x)|",
    ],
    "Integracion Numerica": [
        r"I\approx h\sum_{i=1}^{n} f\!\left(a+\left(i-\frac12\right)h\right)\ \text{(Rectangulo)}",
        r"I\approx h\left[\frac{f(a)+f(b)}{2}+\sum_{i=1}^{n-1}f(x_i)\right]\ \text{(Trapecio)}",
        r"I\approx\frac{h}{3}\left[f(x_0)+f(x_n)+4\sum f(x_{impar})+2\sum f(x_{par})\right]\ \text{(Simpson 1/3)}",
        r"I\approx\frac{3h}{8}\left[f(x_0)+f(x_n)+3\sum f(x_{i\not\equiv0\ (3)})+2\sum f(x_{i\equiv0\ (3)})\right]\ \text{(Simpson 3/8)}",
    ],
    "Monte Carlo": [
        r"I\approx(b-a)\,\frac{1}{n}\sum_{i=1}^{n}f(x_i),\quad x_i\sim U(a,b)",
        r"\widehat{\sigma}_I=\sqrt{\frac{(b-a)^2}{n}\,\widehat{\mathrm{Var}}(f(x_i))}",
        r"IC_{1-\alpha}:\ I\pm z_{1-\alpha/2}\,\widehat{\sigma}_I",
        r"\widehat{e}_{MC}=\mathcal{O}(n^{-1/2})",
    ],
    "Monte Carlo 2D": [
        r"I\approx(b-a)(d-c)\,\frac{1}{n}\sum_{i=1}^{n}f(x_i,y_i),\ (x_i,y_i)\sim U([a,b]\times[c,d])",
        r"\widehat{\sigma}_I=\sqrt{\frac{[(b-a)(d-c)]^2}{n}\,\widehat{\mathrm{Var}}(f(x_i,y_i))}",
        r"IC_{1-\alpha}:\ I\pm z_{1-\alpha/2}\,\widehat{\sigma}_I",
        r"\widehat{e}_{MC}=\mathcal{O}(n^{-1/2})",
    ],
    "Ajuste de Curvas": [
        r"\hat{y}=a x+b\ \text{(lineal)}",
        r"\hat{y}=\sum_{k=0}^{m} a_k x^k\ \text{(polinomial)}",
        r"\text{MSE}=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2",
        r"R^2=1-\frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}",
    ],
    "Sistemas Lineales": [
        r"A\mathbf{x}=\mathbf{b}",
        r"\mathbf{x}^{(k+1)}=D^{-1}(\mathbf{b}-(L+U)\mathbf{x}^{(k)})\ \text{(iterativa base)}",
        r"x_i^{(k+1)}=\frac{1}{a_{ii}}\left(b_i-\sum_{j<i}a_{ij}x_j^{(k+1)}-\sum_{j>i}a_{ij}x_j^{(k)}\right)\ \text{(Gauss-Seidel)}",
        r"\|e^{(k)}\|_{\infty}=\|x^{(k)}-x^{(k-1)}\|_{\infty}",
    ],
    "EDO": [
        r"y_{n+1}=y_n+h f(x_n,y_n)\ \text{(Euler)}",
        r"k_1=f(x_n,y_n),\ k_2=f\!\left(x_n+\frac h2,y_n+\frac h2k_1\right)",
        r"k_3=f\!\left(x_n+\frac h2,y_n+\frac h2k_2\right),\ k_4=f(x_n+h,y_n+hk_3)",
        r"y_{n+1}=y_n+\frac h6(k_1+2k_2+2k_3+k_4)\ \text{(RK4)}",
    ],
    "Red Neuronal GD": [
        r"\hat{y}=wx+b",
        r"J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2",
        r"w\leftarrow w-\alpha\frac{\partial J}{\partial w},\quad b\leftarrow b-\alpha\frac{\partial J}{\partial b}",
        r"\frac{\partial J}{\partial w}=\frac{2}{m}\sum x_i(\hat{y}_i-y_i),\quad \frac{\partial J}{\partial b}=\frac{2}{m}\sum(\hat{y}_i-y_i)",
    ],
}


SIMBOLOS_POR_APARTADO = {
    "Newton-Raphson": [
        r"$x_n$: aproximacion en iteracion $n$",
        r"$f(x)$: funcion objetivo",
        r"$f'(x)$: derivada de $f$",
        r"$\varepsilon$: tolerancia",
        r"$e_n$: error iterativo",
    ],
    "Aitken": [
        r"$g(x)$: funcion de punto fijo",
        r"$x_n$: iteracion base",
        r"$\hat{x}_n$: aceleracion de Aitken",
        r"$e_n$: error entre aceleraciones consecutivas",
    ],
    "Biseccion": [
        r"$a_n,b_n$: extremos del intervalo",
        r"$c_n$: punto medio",
        r"$f(c_n)$: evaluacion en el punto medio",
        r"$e_n$: semiancho del intervalo",
    ],
    "Punto Fijo": [
        r"$x_n$: iteracion actual",
        r"$g(x)$: funcion iterativa",
        r"$x^*$: punto fijo (raiz)",
        r"$e_n$: error iterativo",
    ],
    "Comparativa": [
        r"$x_{aprox}$: raiz aproximada por metodo",
        r"$x_{ref}$: raiz de referencia",
        r"$N_{iter}$: numero de iteraciones",
        r"$e_n$: error de parada",
    ],
    "Lagrange + Derivacion": [
        r"$P_n(x)$: polinomio interpolante de grado $n$",
        r"$L_i(x)$: base de Lagrange",
        r"$x_i,y_i$: nodos de interpolacion",
        r"$h$: paso para diferencias finitas",
        r"$E(x)$: error de interpolacion",
    ],
    "Integracion Numerica": [
        r"$I$: integral aproximada",
        r"$a,b$: limites de integracion",
        r"$n$: numero de subintervalos",
        r"$h=(b-a)/n$: paso",
        r"$x_i=a+ih$: nodos",
    ],
    "Monte Carlo": [
        r"$a,b$: limites del intervalo",
        r"$x_i$: muestra uniforme en $[a,b]$",
        r"$n$: cantidad de muestras aleatorias",
        r"$\widehat{\sigma}_I$: desviacion estandar del estimador",
        r"$z$: cuantil normal para el nivel de confianza",
    ],
    "Monte Carlo 2D": [
        r"$a,b$: limites en eje $x$",
        r"$c,d$: limites en eje $y$",
        r"$(x_i,y_i)$: muestra uniforme en el rectangulo",
        r"$A=(b-a)(d-c)$: area de integracion",
        r"$\widehat{\sigma}_I$: desviacion estandar del estimador",
    ],
    "Ajuste de Curvas": [
        r"$y_i$: valor observado",
        r"$\hat{y}_i$: valor ajustado",
        r"$a,b,a_k$: parametros del modelo",
        r"$m$: grado del polinomio",
        r"$R^2$, MSE: metricas de ajuste",
    ],
    "Sistemas Lineales": [
        r"$A$: matriz de coeficientes",
        r"$\mathbf{x}$: vector desconocido",
        r"$\mathbf{b}$: vector termino independiente",
        r"$D,L,U$: descomposicion diagonal, inferior y superior",
        r"$\|\cdot\|_\infty$: norma infinito",
    ],
    "EDO": [
        r"$y'=f(x,y)$: ecuacion diferencial",
        r"$(x_0,y_0)$: condicion inicial",
        r"$h$: paso",
        r"$k_1,k_2,k_3,k_4$: pendientes de RK4",
        r"$y_n$: aproximacion en el nodo $x_n$",
    ],
    "Red Neuronal GD": [
        r"$x,y$: datos de entrada y salida",
        r"$w,b$: peso y sesgo",
        r"$\hat{y}$: prediccion del modelo",
        r"$J(w,b)$: funcion de costo",
        r"$\alpha$: tasa de aprendizaje",
    ],
}


CONDICIONES_POR_APARTADO = {
    "Newton-Raphson": [
        "f debe ser derivable cerca de la raiz.",
        "f'(x_n) no debe ser cero (o casi cero).",
        "x0 debe elegirse cerca de la raiz buscada.",
    ],
    "Aitken": [
        "La sucesion base x_{n+1}=g(x_n) debe converger.",
        "El denominador x_{n+2}-2x_{n+1}+x_n no debe anularse.",
        "Idealmente |g'(x*)|<1 en torno al punto fijo.",
    ],
    "Biseccion": [
        "Requiere intervalo inicial [a,b] con f(a)f(b)<0.",
        "f debe ser continua en [a,b].",
        "Garantiza convergencia si se cumplen las dos condiciones anteriores.",
    ],
    "Punto Fijo": [
        "Se debe reescribir como x=g(x).",
        "Condicion local de convergencia: |g'(x*)|<1.",
        "x0 debe estar en una zona de atraccion del punto fijo.",
    ],
    "Comparativa": [
        "Usar la misma tolerancia y max_iter para comparar metodos de forma justa.",
        "Interpretar errores con la misma referencia (exacta o numerica).",
    ],
    "Lagrange + Derivacion": [
        "Los nodos x_i deben ser distintos.",
        "Para error exacto se requiere disponer de f(x) real.",
        "En diferencias finitas, h no debe ser demasiado grande ni demasiado pequeño.",
    ],
    "Integracion Numerica": [
        "Rectangulo/Trapecio: n entero positivo.",
        "Simpson 1/3: n par.",
        "Simpson 3/8: n multiplo de 3.",
        "f(x) debe ser integrable en [a,b] (singularidades removibles se tratan con limite).",
    ],
    "Monte Carlo": [
        "Se requiere b > a y n >= 1.",
        "f(x) debe evaluarse de forma finita en la mayor parte del intervalo.",
        "La precision mejora al aumentar n (convergencia estadistica).",
        "Para reproducibilidad, usar semilla fija.",
    ],
    "Monte Carlo 2D": [
        "Se requiere b > a, d > c y n >= 1.",
        "f(x,y) debe poder evaluarse en los puntos muestreados.",
        "La region de integracion debe ser rectangular [a,b]x[c,d].",
        "La precision mejora al aumentar n (orden Monte Carlo).",
    ],
    "Ajuste de Curvas": [
        "x e y deben tener la misma cantidad de datos.",
        "En ajuste polinomial, grado <= numero de datos - 1.",
        "Evitar extrapolaciones amplias fuera del rango de datos.",
    ],
    "Sistemas Lineales": [
        "A debe ser cuadrada y compatible con b.",
        "Gauss-Jordan requiere pivotes no nulos (o pivoteo).",
        "Gauss-Seidel converge mejor con diagonal dominante o matriz SPD.",
    ],
    "EDO": [
        "f(x,y) debe estar bien definida en el dominio de integracion.",
        "El paso h debe ser positivo y acorde a la estabilidad del metodo.",
        "RK4 suele ofrecer mayor precision que Euler con el mismo h.",
    ],
    "Red Neuronal GD": [
        "Elegir una tasa de aprendizaje alpha estable (ni muy grande ni muy chica).",
        "Escalar datos puede mejorar la convergencia.",
        "Requiere epocas suficientes para estabilizar el costo.",
    ],
}


PASOS_POR_APARTADO = {
    "Newton-Raphson": [
        {
            "titulo": "Iteracion principal",
            "simbolico": r"$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$",
            "numerico": "Si x_n=2, f(x_n)=1 y f'(x_n)=5, entonces x_{n+1}=2-1/5=1.8.",
        },
        {
            "titulo": "Error iterativo",
            "simbolico": r"$e_n=|x_{n+1}-x_n|$",
            "numerico": "Con x_{n+1}=1.8 y x_n=2, e_n=0.2.",
        },
        {
            "titulo": "Criterio de parada",
            "simbolico": r"$e_n<\varepsilon$",
            "numerico": "Si eps=1e-6, se detiene cuando e_n sea menor que 0.000001.",
        },
    ],
    "Aitken": [
        {
            "titulo": "Secuencia base",
            "simbolico": r"$x_{n+1}=g(x_n)$",
            "numerico": "Si g(x)=cos(x) y x0=0.5, se calcula x1=cos(0.5).",
        },
        {
            "titulo": "Aceleracion Delta-cuadrado",
            "simbolico": r"$\hat{x}_n=x_n-\frac{(x_{n+1}-x_n)^2}{x_{n+2}-2x_{n+1}+x_n}$",
            "numerico": "Se usan tres iteraciones consecutivas para estimar una raiz mejorada.",
        },
        {
            "titulo": "Error",
            "simbolico": r"$e_n=|\hat{x}_n-\hat{x}_{n-1}|$",
            "numerico": "Si xhat_n=0.7391 y xhat_{n-1}=0.7390, entonces e_n=0.0001.",
        },
    ],
    "Biseccion": [
        {
            "titulo": "Punto medio",
            "simbolico": r"$c_n=\frac{a_n+b_n}{2}$",
            "numerico": "Si a=1 y b=2, c=1.5.",
        },
        {
            "titulo": "Seleccion de subintervalo",
            "simbolico": r"$f(a_n)f(c_n)<0\Rightarrow[a_n,c_n]$, si no $[c_n,b_n]$",
            "numerico": "Si f(1)=-1 y f(1.5)=0.2, hay cambio de signo en [1,1.5].",
        },
        {
            "titulo": "Error por intervalo",
            "simbolico": r"$e_n\approx\frac{b_n-a_n}{2}$",
            "numerico": "Si b-a=0.01, error aprox=0.005.",
        },
    ],
    "Punto Fijo": [
        {
            "titulo": "Iterar g(x)",
            "simbolico": r"$x_{n+1}=g(x_n)$",
            "numerico": "Con x0=1 y g(x)=(x+2/x)/2, se obtiene x1=1.5.",
        },
        {
            "titulo": "Error",
            "simbolico": r"$e_n=|x_{n+1}-x_n|$",
            "numerico": "Si x2=1.4167 y x1=1.5, e=0.0833.",
        },
        {
            "titulo": "Convergencia local",
            "simbolico": r"$|g'(x^*)|<1$",
            "numerico": "Si |g'(x*)|=0.3, normalmente converge.",
        },
    ],
    "Comparativa": [
        {
            "titulo": "Error relativo",
            "simbolico": r"$e_r=\frac{|x_{aprox}-x_{ref}|}{|x_{ref}|}$",
            "numerico": "Si x_aprox=1.41 y x_ref=1.4142, e_r aprox=0.00297.",
        },
        {
            "titulo": "Costo iterativo",
            "simbolico": r"$N_{iter}$ por metodo",
            "numerico": "Newton puede requerir menos iteraciones que Biseccion.",
        },
    ],
    "Lagrange + Derivacion": [
        {
            "titulo": "Interpolacion",
            "simbolico": r"$P_n(x)=\sum_{i=0}^{n}y_iL_i(x)$",
            "numerico": "Con 3 puntos se construye P2(x) que pasa por todos los nodos.",
        },
        {
            "titulo": "Base de Lagrange",
            "simbolico": r"$L_i(x)=\prod_{j\neq i}\frac{x-x_j}{x_i-x_j}$",
            "numerico": "Cada L_i vale 1 en x_i y 0 en el resto de nodos.",
        },
        {
            "titulo": "Derivada numerica",
            "simbolico": r"$f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}$",
            "numerico": "Si h=0.1, usar valores en x+0.1 y x-0.1.",
        },
    ],
    "Integracion Numerica": [
        {
            "titulo": "Paso",
            "simbolico": r"$h=(b-a)/n$",
            "numerico": "Si a=0, b=1, n=4, entonces h=0.25.",
        },
        {
            "titulo": "Trapecio compuesto",
            "simbolico": r"$I\approx h[\frac{f(a)+f(b)}{2}+\sum f(x_i)]$",
            "numerico": "Se evalua f en nodos y se suman con pesos 1/2 en extremos.",
        },
        {
            "titulo": "Cota de truncamiento",
            "simbolico": r"$|E_T|\le C\,h^p\max|f^{(k)}(x)|$",
            "numerico": "Se reemplaza h y la derivada maxima estimada en el intervalo.",
        },
    ],
    "Monte Carlo": [
        {
            "titulo": "Muestreo uniforme",
            "simbolico": r"$x_i\sim U(a,b),\ i=1,\dots,n$",
            "numerico": "Se generan n puntos aleatorios en [a,b].",
        },
        {
            "titulo": "Estimador de la integral",
            "simbolico": r"$I\approx (b-a)\,\bar f,\ \bar f=\frac{1}{n}\sum f(x_i)$",
            "numerico": "Se calcula el promedio de f(x_i) y se multiplica por (b-a).",
        },
        {
            "titulo": "Incertidumbre e intervalo",
            "simbolico": r"$\widehat{\sigma}_I=\sqrt{\frac{(b-a)^2}{n}\widehat{Var}(f)}\ ,\ IC=I\pm z\widehat{\sigma}_I$",
            "numerico": "Con confianza elegida se calcula z y luego el margen z*sigma.",
        },
    ],
    "Monte Carlo 2D": [
        {
            "titulo": "Muestreo en area rectangular",
            "simbolico": r"$(x_i,y_i)\sim U([a,b]\times[c,d])$",
            "numerico": "Se generan n pares aleatorios dentro del rectangulo.",
        },
        {
            "titulo": "Estimador de integral doble",
            "simbolico": r"$I\approx A\,\bar f,\ A=(b-a)(d-c),\ \bar f=\frac{1}{n}\sum f(x_i,y_i)$",
            "numerico": "Se promedia f(x_i,y_i) y se multiplica por el area A.",
        },
        {
            "titulo": "Incertidumbre e intervalo",
            "simbolico": r"$\widehat{\sigma}_I=\sqrt{\frac{A^2}{n}\widehat{Var}(f)}\ ,\ IC=I\pm z\widehat{\sigma}_I$",
            "numerico": "Se reporta desviacion del estimador y el intervalo de confianza.",
        },
    ],
    "Ajuste de Curvas": [
        {
            "titulo": "Modelo",
            "simbolico": r"$\hat y=ax+b$ o $\hat y=\sum a_kx^k$",
            "numerico": "Con datos (x,y), se obtienen coeficientes por minimos cuadrados.",
        },
        {
            "titulo": "Residuo",
            "simbolico": r"$r_i=y_i-\hat y_i$",
            "numerico": "Si y=5 y yhat=4.7, residuo=0.3.",
        },
        {
            "titulo": "Metricas",
            "simbolico": r"$MSE, RMSE, R^2$",
            "numerico": "MSE promedio de errores cuadrados y R^2 cercano a 1 indica mejor ajuste.",
        },
    ],
    "Sistemas Lineales": [
        {
            "titulo": "Planteo",
            "simbolico": r"$A\mathbf{x}=\mathbf{b}$",
            "numerico": "Ejemplo 3x3: A por vector x igual a vector b.",
        },
        {
            "titulo": "Gauss-Jordan",
            "simbolico": r"$[A|b]\to[I|x]$",
            "numerico": "Se aplican operaciones elementales hasta matriz identidad.",
        },
        {
            "titulo": "Gauss-Seidel",
            "simbolico": r"$x_i^{k+1}=\frac{1}{a_{ii}}(b_i-\sum_{j<i}a_{ij}x_j^{k+1}-\sum_{j>i}a_{ij}x_j^k)$",
            "numerico": "Cada variable se actualiza usando valores nuevos y viejos en cada iteracion.",
        },
    ],
    "EDO": [
        {
            "titulo": "Euler",
            "simbolico": r"$y_{n+1}=y_n+h f(x_n,y_n)$",
            "numerico": "Con h=0.1, x0=0, y0=1 se avanza paso a paso.",
        },
        {
            "titulo": "RK4",
            "simbolico": r"$y_{n+1}=y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4)$",
            "numerico": "Se calculan cuatro pendientes intermedias por cada paso.",
        },
        {
            "titulo": "Error",
            "simbolico": r"$e_n=|y_{num}(x_n)-y_{exacta}(x_n)|$",
            "numerico": "Si hay solucion exacta, se compara punto a punto.",
        },
    ],
    "Red Neuronal GD": [
        {
            "titulo": "Prediccion lineal",
            "simbolico": r"$\hat y=wx+b$",
            "numerico": "Si w=1.2, b=0.3, x=2 => yhat=2.7.",
        },
        {
            "titulo": "Costo",
            "simbolico": r"$J=\frac{1}{m}\sum(\hat y_i-y_i)^2$",
            "numerico": "Promedio de errores cuadrados sobre el dataset.",
        },
        {
            "titulo": "Actualizacion",
            "simbolico": r"$w\leftarrow w-\alpha\partial J/\partial w,\ b\leftarrow b-\alpha\partial J/\partial b$",
            "numerico": "Con alpha=0.03, se corrigen w y b en cada epoca.",
        },
    ],
}


DESGLOSE_COMPLETO_POR_APARTADO = {
    "Newton-Raphson": [
        "Definir f(x), tolerancia eps, max_iter y valor inicial x0.",
        "Derivar simbolicamente f'(x).",
        "Evaluar f(x_n) y f'(x_n) en cada iteracion.",
        "Verificar que f'(x_n) no sea cercano a cero.",
        "Actualizar con x_(n+1)=x_n-f(x_n)/f'(x_n).",
        "Calcular error e_n=|x_(n+1)-x_n|.",
        "Guardar fila completa de iteracion (x_n, f, f', x_(n+1), error).",
        "Repetir hasta e_n<eps o alcanzar max_iter.",
        "Reportar raiz, convergencia e historial completo.",
    ],
    "Aitken": [
        "Definir g(x), eps, max_iter y x0.",
        "Generar la secuencia base x1=g(x0), x2=g(x1), ...",
        "Aplicar aceleracion Delta-cuadrado en cada bloque de tres puntos.",
        "Verificar que el denominador no sea cero.",
        "Calcular estimado acelerado xhat_n.",
        "Calcular error entre aceleraciones sucesivas.",
        "Guardar todas las iteraciones y errores.",
        "Detener por tolerancia o maximo de iteraciones.",
    ],
    "Biseccion": [
        "Definir f(x), intervalo [a,b], eps y max_iter.",
        "Verificar continuidad y cambio de signo f(a)f(b)<0.",
        "Calcular c=(a+b)/2.",
        "Evaluar f(a), f(b), f(c).",
        "Seleccionar nuevo subintervalo segun signo de f(a)f(c).",
        "Actualizar error con semiancho (b-a)/2.",
        "Registrar cada iteracion en tabla.",
        "Repetir hasta cumplir tolerancia.",
    ],
    "Punto Fijo": [
        "Definir g(x), x0, eps y max_iter.",
        "Iterar x_(n+1)=g(x_n).",
        "Calcular error e_n=|x_(n+1)-x_n|.",
        "Registrar cada iteracion (x_n, x_(n+1), error).",
        "Verificar criterio de paro por tolerancia.",
        "Concluir con raiz aproximada y tabla completa.",
    ],
    "Comparativa": [
        "Ejecutar Newton con los parametros comunes.",
        "Ejecutar Aitken con los parametros comunes.",
        "Ejecutar Biseccion con intervalo [a,b].",
        "Ejecutar Punto Fijo con g(x).",
        "Recolectar para cada metodo: raiz, iteraciones, error final.",
        "Construir tabla comparativa y graficos de rendimiento.",
    ],
    "Lagrange + Derivacion": [
        "Ingresar nodos x_i e y_i (manual o desde f exacta).",
        "Construir bases L_i(x) y luego el polinomio P_n(x).",
        "Expandir y mostrar P_n(x).",
        "Evaluar P_n en x* si se solicita.",
        "Graficar datos, interpolante y f exacta (si existe).",
        "Calcular error puntual y error global en intervalo.",
        "Para derivacion: aplicar formula adelante/atras/centrada.",
        "Comparar con derivada exacta cuando este disponible.",
    ],
    "Integracion Numerica": [
        "Definir f(x), a, b, n y metodo.",
        "Calcular paso h=(b-a)/n.",
        "Evaluar nodos requeridos por el metodo.",
        "Aplicar suma ponderada segun Rectangulo/Trapecio/Simpson.",
        "Obtener integral aproximada.",
        "Si hay referencia, calcular error absoluto y relativo.",
        "Estimar cota de truncamiento y mostrar sustitucion numerica.",
        "Graficar aproximacion geometrica del metodo elegido.",
    ],
    "Monte Carlo": [
        "Definir f(x), intervalo [a,b], n y nivel de confianza.",
        "Generar n muestras uniformes x_i en [a,b].",
        "Evaluar y_i=f(x_i) y calcular promedio muestral.",
        "Estimar integral con I=(b-a)*promedio(y_i).",
        "Estimar varianza muestral de y_i y sigma del estimador.",
        "Construir intervalo de confianza I +- z*sigma.",
        "Reportar tabla de puntos y grafico con muestras aleatorias.",
    ],
    "Monte Carlo 2D": [
        "Definir f(x,y), rectangulo [a,b]x[c,d], n y confianza.",
        "Generar muestras uniformes (x_i,y_i) en la region.",
        "Evaluar z_i=f(x_i,y_i) y calcular promedio de z_i.",
        "Estimar integral con I=A*promedio(z_i), A=(b-a)(d-c).",
        "Estimar dispersion y construir intervalo de confianza.",
        "Mostrar tabla de muestras y nube de puntos coloreada por f(x,y).",
    ],
    "Ajuste de Curvas": [
        "Ingresar pares (x_i,y_i) y elegir tipo de regresion.",
        "Construir modelo lineal o polinomial por minimos cuadrados.",
        "Calcular y ajustada y residuos r_i=y_i-yhat_i.",
        "Calcular metricas MSE, RMSE y R^2.",
        "Mostrar ecuacion final y tabla completa.",
        "Graficar puntos, curva ajustada y analisis de residuos.",
    ],
    "Sistemas Lineales": [
        "Ingresar A y b del sistema A x = b.",
        "Analizar estructura (heatmap y dominancia diagonal).",
        "Si es Gauss-Jordan: aplicar operaciones hasta [I|x].",
        "Si es Gauss-Seidel: actualizar variables por iteracion.",
        "Calcular error infinito por iteracion en Seidel.",
        "Mostrar solucion final y pasos/iteraciones completas.",
    ],
    "EDO": [
        "Definir y'=f(x,y), condicion inicial (x0,y0), h y n.",
        "Euler: avanzar con y_(n+1)=y_n+h f(x_n,y_n).",
        "RK4: calcular k1,k2,k3,k4 y combinar.",
        "Construir tabla de nodos (x_n,y_n).",
        "Si hay solucion exacta, calcular errores por nodo.",
        "Graficar trayectorias y campo de pendientes opcional.",
    ],
    "Red Neuronal GD": [
        "Definir dataset, alpha, epocas y semilla.",
        "Inicializar parametros w y b.",
        "Por epoca: calcular predicciones yhat=wx+b.",
        "Calcular costo MSE.",
        "Calcular gradientes dJ/dw y dJ/db.",
        "Actualizar parametros con descenso de gradiente.",
        "Guardar historial completo de costo y parametros.",
        "Mostrar resultados finales y evolucion en graficos.",
    ],
}


def section_newton():
    st.subheader("Metodo de Newton-Raphson")

    with st.form("form_newton"):
        c1, c2 = st.columns(2)
        with c1:
            func = st.text_input("f(x)", value="x**3 - 2*x - 5")
            x0 = st.number_input("x0", value=2.0)
        with c2:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1)
        buscar_primera_positiva = st.checkbox(
            "Priorizar primera raiz positiva automaticamente",
            value=True,
            help="Si hay multiples raices para x>0, ajusta x0 para intentar converger a la primera.",
        )
        run_btn = st.form_submit_button("Ejecutar Newton")

    if run_btn:
        try:
            x0_run = float(x0)
            if buscar_primera_positiva:
                x0_sugerido = sugerir_x0_primera_raiz_positiva(func, x0_run)
                if x0_sugerido is not None and abs(x0_sugerido - x0_run) > 1e-7:
                    x0_run = float(x0_sugerido)
                    st.info(f"Se ajusto x0 automaticamente a {x0_run:.7g} para priorizar la primera raiz positiva.")

            root, iterations, converged = run_silent(
                metodo_newton_raphson, func, x0_run, float(tol), int(max_iter)
            )
            if not iterations:
                st.error("No se generaron iteraciones.")
                return

            df = pd.DataFrame(iterations)
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Convergencia", "Si" if converged else "No")
            st.metric("Iteraciones", len(iterations))

            err_col = "Error |x_(n+1) - x_n|"
            errors = df[err_col].astype(float).to_numpy()

            render_newton_charts(func, root, x0_run, errors)

            cuentas = []
            for i, fila in enumerate(iterations[:5], start=1):
                cuentas.append(
                    rf"x_{{{i}}} = x_{{{i-1}}} - \frac{{f(x_{{{i-1}}})}}{{f'(x_{{{i-1}}})}} = {_num(fila['x_n'])} - \frac{{{_num(fila['f(x_n)'])}}}{{{_num(fila["f'(x_n)"])}}} = {_num(fila['x_(n+1)'])}"
                )
                cuentas.append(rf"e_{{{i}}}=|x_{{{i}}}-x_{{{i-1}}}|={_num(fila['Error |x_(n+1) - x_n|'])}")
            cuentas.append(rf"x^* \approx {_num(root, 12)}")
            guardar_cuentas("Newton-Raphson", cuentas)

        except Exception as exc:
            st.error(f"Error al ejecutar Newton: {exc}")


def section_aitken():
    st.subheader("Metodo de Aitken")

    with st.form("form_aitken"):
        c1, c2 = st.columns(2)
        with c1:
            g = st.text_input("g(x)", value="cos(x)")
            x0 = st.number_input("x0", value=0.5, key="aitken_x0")
        with c2:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="aitken_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="aitken_max")
        run_btn = st.form_submit_button("Ejecutar Aitken")

    if run_btn:
        try:
            root, iterations, converged = run_silent(
                metodo_aitken, g, float(x0), float(tol), int(max_iter)
            )
            if root is None or not iterations:
                st.error("No se pudo obtener resultado para Aitken.")
                return

            df = pd.DataFrame(iterations)
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Convergencia", "Si" if converged else "No")
            st.metric("Iteraciones", len(iterations))

            errors = df["Error"].astype(float).to_numpy()
            fig_e = plot_error_curve(errors, "Error por iteracion (Aitken)")
            render_chart(fig_e)
            plt.close(fig_e)

            margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
            x = np.linspace(float(root) - margin, float(root) + margin, 800)
            y_g = build_func_plot(g, x)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(x, y_g, linewidth=2, label=f"g(x) = {g}")
            ax.plot(x, x, "--", linewidth=1.5, label="y = x")
            ax.plot([root], [root], "ro", label=f"Punto fijo: {root:.8g}")
            ax.set_title("Diagrama de punto fijo (Aitken)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.3)
            ax.legend()
            render_chart(fig)
            plt.close(fig)

            cuentas = []
            for i, fila in enumerate(iterations[:5], start=1):
                x_n = fila.get("x_n", np.nan)
                x_n1 = fila.get("x_n1", fila.get("x_(n+1)", np.nan))
                err = fila.get("Error", np.nan)
                cuentas.append(rf"x_{{{i}}}=g(x_{{{i-1}}})={_num(x_n1)}")
                cuentas.append(rf"e_{{{i}}}=|x_{{{i}}}-x_{{{i-1}}}|={_num(err)}")
            cuentas.append(rf"x^* \approx {_num(root, 12)}")
            guardar_cuentas("Aitken", cuentas)

        except Exception as exc:
            st.error(f"Error al ejecutar Aitken: {exc}")


def section_biseccion():
    st.subheader("Metodo de Biseccion")

    with st.form("form_biseccion"):
        c1, c2 = st.columns(2)
        with c1:
            func = st.text_input("f(x)", value="x**3 - x - 2", key="bis_f")
            a = st.number_input("a", value=1.0, key="bis_a")
        with c2:
            b = st.number_input("b", value=2.0, key="bis_b")
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="bis_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="bis_max")
        run_btn = st.form_submit_button("Ejecutar Biseccion")

    if run_btn:
        try:
            root, rows = run_silent(
                metodo_biseccion,
                func,
                float(a),
                float(b),
                float(tol),
                int(max_iter),
            )
            if not rows:
                st.error("No se generaron iteraciones para Biseccion.")
                return

            cols = ["Iteracion", "a", "b", "c", "f(a)", "f(b)", "f(c)"]
            df = pd.DataFrame(rows, columns=cols)
            df["Error_f(c)"] = df["f(c)"].abs()
            df["Semiancho_intervalo"] = (df["b"] - df["a"]).abs() / 2.0
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Iteraciones", len(df))
            st.metric("Error final |f(c)|", f"{float(df['Error_f(c)'].iloc[-1]):.7f}")

            fig_f = plot_function_with_root(func, float(root), float(a), float(b), "Funcion y raiz aproximada (Biseccion)")
            render_chart(fig_f)
            plt.close(fig_f)

            fig_e = plot_error_curve(df["Error_f(c)"].to_numpy(), "Error por iteracion (Biseccion)", "|f(c)|")
            render_chart(fig_e)
            plt.close(fig_e)

            cuentas = []
            for i, (_, fila) in enumerate(df.head(5).iterrows(), start=1):
                a_i = _num(fila["a"])
                b_i = _num(fila["b"])
                cuentas.append(
                    rf"c_{{{i}}}=\frac{{a_{{{i}}}+b_{{{i}}}}}{{2}}=\frac{{{a_i}+{b_i}}}{{2}}={_num(fila['c'])}"
                )
                cuentas.append(rf"|f(c_{{{i}}})|={_num(abs(fila['f(c)']))}")
            cuentas.append(rf"x^* \approx {_num(root, 12)}")
            guardar_cuentas("Biseccion", cuentas)

        except Exception as exc:
            st.error(f"Error al ejecutar Biseccion: {exc}")


def section_punto_fijo():
    st.subheader("Metodo de Punto Fijo")

    with st.form("form_punto_fijo"):
        c1, c2 = st.columns(2)
        with c1:
            f_plot = st.text_input("f(x) para graficar", value="x**2 - 2")
            g = st.text_input("g(x)", value="(x + 2/x)/2")
            x0 = st.number_input("x0", value=1.0, key="pf_x0")
        with c2:
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="pf_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="pf_max")
            xmin = st.number_input("x min grafico", value=-2.0, key="pf_xmin")
            xmax = st.number_input("x max grafico", value=2.0, key="pf_xmax")
        run_btn = st.form_submit_button("Ejecutar Punto Fijo")

    if run_btn:
        try:
            root, rows = run_silent(
                metodo_punto_fijo,
                g,
                float(x0),
                float(tol),
                int(max_iter),
            )
            if not rows:
                st.error("No se generaron iteraciones para Punto Fijo.")
                return

            cols = ["Iteracion", "x_n", "x_n1", "Error"]
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df, use_container_width=True)

            st.metric("Raiz aproximada", f"{root:.12g}")
            st.metric("Iteraciones", len(df))
            st.metric("Error final", f"{float(df['Error'].iloc[-1]):.7f}")

            if xmax <= xmin:
                st.warning("Para graficar f(x), se requiere x max > x min.")
            else:
                fig_f = plot_function_with_root(
                    f_plot,
                    float(root),
                    float(xmin),
                    float(xmax),
                    "Funcion y raiz aproximada (Punto Fijo)",
                )
                render_chart(fig_f)
                plt.close(fig_f)

            fig_e = plot_error_curve(df["Error"].to_numpy(), "Error por iteracion (Punto Fijo)")
            render_chart(fig_e)
            plt.close(fig_e)

            margin = max(2.0, abs(float(root) - float(x0)) + 1.0)
            x = np.linspace(float(root) - margin, float(root) + margin, 800)
            y_g = build_func_plot(g, x)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(x, y_g, linewidth=2, label=f"g(x) = {g}")
            ax.plot(x, x, "--", linewidth=1.5, label="y = x")
            ax.plot([root], [root], "ro", label=f"Punto fijo: {root:.8g}")
            ax.set_title("Diagrama de punto fijo")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.3)
            ax.legend()
            render_chart(fig)
            plt.close(fig)

            cuentas = []
            for i, (_, fila) in enumerate(df.head(5).iterrows(), start=1):
                cuentas.append(rf"x_{{{i}}}=g(x_{{{i-1}}})={_num(fila['x_n1'])}")
                cuentas.append(rf"e_{{{i}}}=|x_{{{i}}}-x_{{{i-1}}}|={_num(fila['Error'])}")
            cuentas.append(rf"x^* \approx {_num(root, 12)}")
            guardar_cuentas("Punto Fijo", cuentas)

        except Exception as exc:
            st.error(f"Error al ejecutar Punto Fijo: {exc}")


def section_comparativa():
    st.subheader("Comparativa de los 4 metodos")

    with st.form("form_comparativa"):
        c1, c2 = st.columns(2)
        with c1:
            func = st.text_input("f(x)", value="x**3 - 2*x - 5", key="cmp_f")
            g = st.text_input("g(x)", value="cos(x)", key="cmp_g")
            x0 = st.number_input("x0", value=1.0, key="cmp_x0")
        with c2:
            a = st.number_input("a", value=1.0, key="cmp_a")
            b = st.number_input("b", value=2.0, key="cmp_b")
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e", key="cmp_tol")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1, key="cmp_max")
        run_btn = st.form_submit_button("Ejecutar comparativa")

    if run_btn:
        results = []

        try:
            root, iters, conv = run_silent(metodo_newton_raphson, func, float(x0), float(tol), int(max_iter))
            if iters:
                results.append(
                    {
                        "Metodo": "Newton-Raphson",
                        "Raiz": float(root),
                        "Iteraciones": len(iters),
                        "Error_final": float(iters[-1]["Error |x_(n+1) - x_n|"]),
                        "Convergencia": bool(conv),
                    }
                )
        except Exception as exc:
            st.warning(f"Newton no se pudo ejecutar: {exc}")

        try:
            root, iters, conv = run_silent(metodo_aitken, g, float(x0), float(tol), int(max_iter))
            if root is not None and iters:
                results.append(
                    {
                        "Metodo": "Aitken",
                        "Raiz": float(root),
                        "Iteraciones": len(iters),
                        "Error_final": float(iters[-1]["Error"]),
                        "Convergencia": bool(conv),
                    }
                )
        except Exception as exc:
            st.warning(f"Aitken no se pudo ejecutar: {exc}")

        try:
            root, rows = run_silent(metodo_biseccion, func, float(a), float(b), float(tol), int(max_iter))
            if rows:
                results.append(
                    {
                        "Metodo": "Biseccion",
                        "Raiz": float(root),
                        "Iteraciones": len(rows),
                        "Error_final": abs(float(rows[-1][6])),
                        "Convergencia": True,
                    }
                )
        except Exception as exc:
            st.warning(f"Biseccion no se pudo ejecutar: {exc}")

        try:
            root, rows = run_silent(metodo_punto_fijo, g, float(x0), float(tol), int(max_iter))
            if rows:
                results.append(
                    {
                        "Metodo": "Punto Fijo",
                        "Raiz": float(root),
                        "Iteraciones": len(rows),
                        "Error_final": float(rows[-1][3]),
                        "Convergencia": True,
                    }
                )
        except Exception as exc:
            st.warning(f"Punto Fijo no se pudo ejecutar: {exc}")

        if not results:
            st.error("No hubo resultados para mostrar en la comparativa.")
            return

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        fig1, ax1 = plt.subplots(figsize=(8, 4.2))
        ax1.bar(df["Metodo"], df["Iteraciones"])
        ax1.set_title("Iteraciones por metodo")
        ax1.set_ylabel("Cantidad de iteraciones")
        ax1.grid(axis="y", alpha=0.3)
        render_chart(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 4.2))
        values = np.clip(df["Error_final"].to_numpy(dtype=float), 1e-18, None)
        ax2.bar(df["Metodo"], values)
        ax2.set_yscale("log")
        ax2.set_title("Error final por metodo (escala log)")
        ax2.set_ylabel("Error final")
        ax2.grid(axis="y", alpha=0.3, which="both")
        render_chart(fig2)
        plt.close(fig2)

        cuentas = []
        for _, row in df.iterrows():
            cuentas.append(
                rf"\text{{{row['Metodo']}}}:\ x^*={_num(row['Raiz'], 10)},\ N_{{iter}}={int(row['Iteraciones'])},\ e_f={_num(row['Error_final'])}"
            )
        guardar_cuentas("Comparativa", cuentas)


def section_lagrange():
    st.subheader("Lagrange, derivacion y error")

    st.markdown("Ingresa datos para interpolacion. Puedes cargar y manualmente o desde f(x).")

    mode = st.radio(
        "Modo de carga",
        ["y manual", "y desde f(x)"],
        horizontal=True,
        key="lag_mode",
    )
    x_text = st.text_input("x (separados por coma)", value="0, 1, 2, 3", key="lag_x_text")

    if mode == "y manual":
        y_text = st.text_input("y (separados por coma)", value="1, 2, 0, 5", key="lag_y_text")
        f_exact_text = st.text_input("f(x) exacta opcional", value="", key="lag_f_exact_text")
    else:
        y_text = ""
        f_exact_text = st.text_input("f(x) exacta", value="sin(x)", key="lag_f_exact_text")

    build_btn = st.button("Construir interpolacion")

    if build_btn:
        try:
            x_vals = parse_expr_list(x_text)
            if mode == "y manual":
                y_vals = parse_expr_list(y_text)
            else:
                x_sym = sp.Symbol("x")
                f_expr = safe_eval_expr(f_exact_text, "x")
                y_vals = [sp.simplify(f_expr.subs(x_sym, xv)) for xv in x_vals]

            f_exact_expr = None
            if f_exact_text.strip():
                f_exact_expr = safe_eval_expr(f_exact_text, "x")

            st.session_state["lag_x"] = x_vals
            st.session_state["lag_y"] = y_vals
            st.session_state["lag_f_exact"] = f_exact_expr
            st.success("Datos cargados correctamente.")

        except Exception as exc:
            st.error(f"Error en los datos de interpolacion: {exc}")
            return

    if "lag_x" not in st.session_state:
        return

    x_vals = st.session_state["lag_x"]
    y_vals = st.session_state["lag_y"]
    f_exact_expr = st.session_state.get("lag_f_exact")

    try:
        p_lagr, bases_lagr = polinomio_lagrange(x_vals, y_vals)
        p_lagr = sp.expand(p_lagr)
        st.write("Polinomio de Lagrange:")
        st.latex(f"P(x) = {polynomial_to_decimal_latex(p_lagr, max_decimals=7)}")

        st.write("Bases de Lagrange:")
        with st.expander("Ver bases L_i(x)", expanded=True):
            for i, li in enumerate(bases_lagr):
                li_expandida = sp.expand(li)
                st.latex(f"L_{{{i}}}(x) = {sp.latex(li_expandida)}")

        c_eval_1, c_eval_2 = st.columns([2, 1])
        x_eval_text = c_eval_1.text_input(
            "Evaluar P(x) en x*",
            value="1",
            key="lag_eval_xstar",
        )
        eval_btn = c_eval_2.button("Evaluar polinomio", key="lag_eval_btn")

        if eval_btn:
            try:
                x_eval = float(sp.N(sp.sympify(x_eval_text, locals=ALLOWED_LOCALS)))
                x = sp.Symbol("x")
                y_eval = float(sp.N(p_lagr.subs(x, x_eval)))
                st.write(f"P({x_eval:.7f}) = {y_eval:.7f}")
            except Exception as exc:
                st.error(f"No se pudo evaluar el polinomio en x*: {exc}")

        x_num = to_float_array(x_vals)
        y_num = to_float_array(y_vals)

        x_min = float(np.min(x_num))
        x_max = float(np.max(x_num))
        margin = max(1.0, 0.2 * (x_max - x_min + 1e-12))
        x_plot = np.linspace(x_min - margin, x_max + margin, 900)

        x = sp.Symbol("x")
        p_fun = sp.lambdify(x, p_lagr, "numpy")
        y_plot = np.array(p_fun(x_plot), dtype=float)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(x_plot, y_plot, linewidth=2, label="P(x) Lagrange")
        ax.scatter(x_num, y_num, color="red", zorder=3, label="Datos")

        if f_exact_expr is not None:
            f_fun = sp.lambdify(x, f_exact_expr, "numpy")
            y_real = np.array(f_fun(x_plot), dtype=float)
            ax.plot(x_plot, y_real, "--", linewidth=1.8, label="f(x) exacta")

        ax.set_title("Interpolacion de Lagrange")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        ax.legend()
        render_chart(fig)
        plt.close(fig)

        if f_exact_expr is not None:
            f_fun = sp.lambdify(x, f_exact_expr, "numpy")
            y_real = np.array(f_fun(x_plot), dtype=float)
            err = np.abs(y_real - y_plot)

            fig_e = plot_error_curve(err, "Error |f(x) - P(x)| en el intervalo")
            render_chart(fig_e)
            plt.close(fig_e)

            # Error global maximo en el intervalo de interpolacion [min(x_i), max(x_i)].
            x_global = np.linspace(x_min, x_max, 2000)
            y_global_real = np.array(f_fun(x_global), dtype=float)
            y_global_interp = np.array(p_fun(x_global), dtype=float)
            err_global_arr = np.abs(y_global_real - y_global_interp)
            idx_global_max = int(np.argmax(err_global_arr))
            err_global_max = float(err_global_arr[idx_global_max])
            x_global_max = float(x_global[idx_global_max])

            st.markdown("---")
            st.markdown("Error local en punto especifico y comparacion con error global")

            c_local_1, c_local_2 = st.columns([2, 1])
            x_local_text = c_local_1.text_input(
                "Punto x* para error local",
                value=str(round((x_min + x_max) / 2.0, 6)),
                key="lag_local_xstar",
            )
            calc_local_btn = c_local_2.button("Calcular error local", key="lag_local_btn")

            st.write(
                f"Error global maximo en [{x_min:.7g}, {x_max:.7g}] = {err_global_max:.7f} "
                f"(en x = {x_global_max:.7f})"
            )

            if calc_local_btn:
                try:
                    x_star = float(sp.N(sp.sympify(x_local_text, locals=ALLOWED_LOCALS)))
                    y_star_real = float(sp.N(f_exact_expr.subs(x, x_star)))
                    y_star_interp = float(sp.N(p_lagr.subs(x, x_star)))
                    err_local = abs(y_star_real - y_star_interp)

                    razon = err_local / err_global_max if err_global_max > 1e-15 else np.nan
                    diferencia = err_global_max - err_local

                    st.write(f"f(x*) real = {y_star_real:.7f}")
                    st.write(f"P(x*) = {y_star_interp:.7f}")
                    st.write(f"Error local |f(x*) - P(x*)| = {err_local:.7f}")
                    st.write(f"Error global maximo = {err_global_max:.7f}")

                    if np.isnan(razon):
                        st.write("Comparacion local/global: no definida (error global ~ 0)")
                    else:
                        st.write(f"Relacion local/global = {razon:.7f}")
                        st.write(f"Diferencia (global - local) = {diferencia:.7f}")

                except Exception as exc:
                    st.error(f"No se pudo calcular el error local en x*: {exc}")
        else:
            st.markdown("---")
            st.info("Para calcular y comparar error local vs global, debes ingresar f(x) exacta.")

    except Exception as exc:
        st.error(f"Error al construir la interpolacion: {exc}")
        return

    st.markdown("---")
    st.markdown("Derivacion por aproximacion")

    c1, c2, c3 = st.columns(3)
    forma = c1.selectbox("Forma", ["adelante", "atras", "centrada"])
    x_obj_text = c2.text_input("x objetivo (opcional)", value="")
    deriv_btn = c3.button("Calcular derivada")

    if deriv_btn:
        try:
            x_obj = None
            if x_obj_text.strip():
                x_obj = float(sp.N(sp.sympify(x_obj_text, locals=ALLOWED_LOCALS)))

            result = aproximar_derivada_tres_formas(x_vals, y_vals, forma, x_obj)
            x_eval = float(result["x_evaluacion"])
            d_aprox = float(result["derivada"])

            # Si el usuario ingresa x objetivo, evaluamos la derivada del polinomio local
            # en ese punto para que el valor responda al input de la interfaz.
            if x_obj is not None:
                x = sp.Symbol("x")
                d_poly_local = result["derivada_polinomio_local"]
                d_aprox = float(sp.N(d_poly_local.subs(x, x_obj)))
                x_eval = float(x_obj)

            st.write(f"Derivada aproximada en x = {x_eval:.12g}")
            st.write(f"f'(x) aprox = {d_aprox:.12g}")
            st.write(f"Puntos usados para aproximar: {list(result['x_sub'])}")

            if f_exact_expr is not None:
                x = sp.Symbol("x")
                d_real_expr = sp.diff(f_exact_expr, x)
                d_real = float(sp.N(d_real_expr.subs(x, x_eval)))
                err_abs = abs(d_aprox - d_real)
                err_rel = err_abs / abs(d_real) if abs(d_real) > 1e-15 else np.nan

                st.write(f"f'(x) real = {d_real:.12g}")
                st.write(f"Error absoluto = {err_abs:.7f}")
                if np.isnan(err_rel):
                    st.write("Error relativo = no definido (derivada real cercana a 0)")
                else:
                    st.write(f"Error relativo = {err_rel:.7f}")
            else:
                st.info("No hay f(x) exacta, por eso no se puede calcular el error con la derivada real.")

        except Exception as exc:
            st.error(f"Error al aproximar derivada: {exc}")

    st.markdown("---")
    st.markdown("Diferencias divididas y polinomio de Newton")

    if st.button("Construir tabla DD"):
        try:
            x_dd, table = tabla_diferencias_divididas(x_vals, y_vals)

            n = len(x_dd)
            columns = ["x_i", "f[x_i]"] + [f"DD orden {k}" for k in range(1, n)]
            rows = []
            for i in range(n):
                row = [x_dd[i]]
                for j in range(n):
                    if i <= n - j - 1:
                        row.append(table[i, j])
                    else:
                        row.append(np.nan)
                rows.append(row)

            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, use_container_width=True)

            p_newton = sp.expand(polinomio_newton_desde_dd(x_dd, table))
            st.code(f"P_N(x) = {polynomial_to_decimal_text(p_newton, max_decimals=7)}")

            cuentas = [
                rf"P_n(x) = {sp.latex(p_lagr)}",
                rf"P_n(x^*)\ \text{{evaluado en datos del usuario}}",
            ]
            guardar_cuentas("Lagrange + Derivacion", cuentas)

        except Exception as exc:
            st.error(f"Error en diferencias divididas: {exc}")


def section_integracion_numerica():
    st.subheader("Integracion numerica")

    with st.form("form_integracion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            func = st.text_input("f(x)", value="sin(x)")
            a_text = st.text_input("Limite inferior a", value="0", help="Admite: numero, pi, oo, -oo, inf, -inf")
            b_text = st.text_input("Limite superior b", value="pi", help="Admite: numero, pi, oo, -oo, inf, -inf")
        with c2:
            n = st.number_input("Cantidad de intervalos n", value=6, min_value=1, step=1)
            metodo = st.selectbox("Metodo", ["Rectangulo", "Trapecio", "Simpson 1/3", "Simpson 3/8"])
            comparar_metodos = st.checkbox("Comparar los 4 metodos", value=True)
        with c3:
            analizar_convergencia = st.checkbox("Mostrar convergencia (error vs n)", value=True)
            n_max = st.number_input("n max para convergencia", value=30, min_value=6, step=2)
            referencia_exacta = st.checkbox("Intentar integral exacta con Sympy", value=True)
        run_btn = st.form_submit_button("Calcular integral")

    if run_btn:
        try:
            a_val = parse_numeric_expr(a_text, "a", allow_infinite=True)
            b_val = parse_numeric_expr(b_text, "b", allow_infinite=True)

            es_impropia = not (np.isfinite(a_val) and np.isfinite(b_val))
            if np.isfinite(a_val) and np.isfinite(b_val) and b_val <= a_val:
                st.error("Se requiere b > a.")
                return
            if np.isposinf(a_val):
                st.error("El limite inferior no puede ser +infinito.")
                return
            if np.isneginf(b_val):
                st.error("El limite superior no puede ser -infinito.")
                return

            x_sym = sp.Symbol("x")
            exact_val = None
            exact_label = "Integral exacta"
            expr = None
            if referencia_exacta:
                try:
                    expr = safe_eval_expr(func, "x")
                    a_sym = -sp.oo if np.isneginf(a_val) else (sp.oo if np.isposinf(a_val) else float(a_val))
                    b_sym = -sp.oo if np.isneginf(b_val) else (sp.oo if np.isposinf(b_val) else float(b_val))
                    exact_expr = sp.integrate(expr, (x_sym, a_sym, b_sym))
                    exact_eval = sp.N(exact_expr, 30)
                    exact_candidate = float(exact_eval)
                    if np.isfinite(exact_candidate):
                        exact_val = exact_candidate
                        exact_label = "Integral exacta (Sympy)"
                except Exception:
                    exact_val = None

                # Si no hay forma cerrada o falla Sympy, usa una referencia numerica robusta.
                if exact_val is None and expr is not None:
                    try:
                        mp = __import__("mpmath")
                        mp.mp.dps = 50
                        f_mp = sp.lambdify(x_sym, expr, modules=["mpmath"])
                        exact_num = mp.quad(lambda t: f_mp(t), [a_val, b_val])
                        exact_candidate = float(exact_num)
                        if np.isfinite(exact_candidate):
                            exact_val = exact_candidate
                            exact_label = "Integral de referencia (alta precision)"
                    except Exception:
                        exact_val = None

            if expr is None:
                try:
                    expr = safe_eval_expr(func, "x")
                except Exception:
                    expr = None

            max_f2 = None
            max_f4 = None
            if expr is not None and (not es_impropia):
                try:
                    max_f2 = estimate_max_abs_derivative(expr, x_sym, 2, a_val, b_val)
                except Exception:
                    max_f2 = None
                try:
                    max_f4 = estimate_max_abs_derivative(expr, x_sym, 4, a_val, b_val)
                except Exception:
                    max_f4 = None

            resultado_int = integrar_numerica_soporte_infinito(func, a_val, b_val, int(n), metodo)
            valor = resultado_int["valor"]
            x_nodes = resultado_int["x_nodes"]
            y_nodes = resultado_int["y_nodes"]
            es_impropia = resultado_int["impropia"]

            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Resultado", f"{valor:.12g}")
            c_m2.metric("Metodo", metodo)
            c_m3.metric("Intervalos n", int(n))

            cota_sel = np.nan if es_impropia else cota_truncamiento_integracion(
                metodo,
                a_val,
                b_val,
                int(n),
                max_f2=max_f2,
                max_f4=max_f4,
            )
            if np.isfinite(cota_sel):
                st.metric("Cota teorica de truncamiento", f"{float(cota_sel):.7f}")
            else:
                if es_impropia:
                    st.info("La cota teorica de truncamiento implementada aplica a intervalos finitos.")
                else:
                    st.info("No se pudo estimar la cota teorica de truncamiento para este metodo.")

            if not es_impropia:
                detalle_cota = detalle_cota_truncamiento_integracion(
                    metodo,
                    a_val,
                    b_val,
                    int(n),
                    max_f2=max_f2,
                    max_f4=max_f4,
                )
                with st.expander("Ver paso a paso del error de truncamiento"):
                    for paso in detalle_cota["pasos"]:
                        st.write(paso)
                    if detalle_cota["latex_formula"]:
                        st.latex(detalle_cota["latex_formula"])
                    if detalle_cota["latex_sustitucion"]:
                        st.latex(detalle_cota["latex_sustitucion"])
                    if detalle_cota["ok"]:
                        st.latex(rf"|E_T| \leq {float(detalle_cota['cota']):.7g}")
            else:
                st.info("Se utilizo transformacion de variable para resolver integral impropia.")

            if exact_val is not None:
                err_abs = abs(float(valor) - float(exact_val))
                err_rel = err_abs / abs(exact_val) if abs(exact_val) > 1e-15 else np.nan
                c_e1, c_e2, c_e3 = st.columns(3)
                c_e1.metric(exact_label, f"{exact_val:.12g}")
                c_e2.metric("Error absoluto", f"{err_abs:.7f}")
                c_e3.metric("Error relativo", "no definido" if np.isnan(err_rel) else f"{err_rel:.7f}")
            elif referencia_exacta:
                st.warning(
                    "No se pudo obtener integral exacta ni referencia numerica de alta precision para esta funcion e intervalo."
                )

            if es_impropia:
                df_nodes = pd.DataFrame(
                    {
                        "u_i": resultado_int["u_nodes"],
                        "x(u_i)": x_nodes,
                        "g(u_i)": resultado_int["g_nodes"],
                    }
                )
            else:
                df_nodes = pd.DataFrame({"x_i": x_nodes, "f(x_i)": y_nodes})
            st.dataframe(
                df_nodes,
                use_container_width=True,
                column_config={k: st.column_config.NumberColumn(k, format="%.7f") for k in df_nodes.columns},
            )

            if es_impropia:
                fig = plot_integracion_impropia_transformada(
                    resultado_int["u_nodes"],
                    resultado_int["g_nodes"],
                    metodo,
                    resultado_int["tipo"],
                )
            else:
                fig = plot_integracion_visual(func, a_val, b_val, int(n), metodo)
            render_chart(fig)
            plt.close(fig)

            if comparar_metodos:
                comp_rows = []
                for nombre in ["Rectangulo", "Trapecio", "Simpson 1/3", "Simpson 3/8"]:
                    try:
                        v = integrar_numerica_soporte_infinito(func, a_val, b_val, int(n), nombre)["valor"]

                        row = {"Metodo": nombre, "Integral": float(v)}
                        cota = np.nan if es_impropia else cota_truncamiento_integracion(
                            nombre,
                            a_val,
                            b_val,
                            int(n),
                            max_f2=max_f2,
                            max_f4=max_f4,
                        )
                        row["Cota_trunc"] = cota if np.isfinite(cota) else np.nan
                        if exact_val is not None:
                            row["Error_abs"] = abs(float(v) - exact_val)
                        comp_rows.append(row)
                    except Exception as exc:
                        comp_rows.append({"Metodo": nombre, "Integral": np.nan, "Cota_trunc": np.nan, "Estado": str(exc)})

                df_comp = pd.DataFrame(comp_rows)
                st.dataframe(
                    df_comp,
                    use_container_width=True,
                    column_config={
                        "Integral": st.column_config.NumberColumn("Integral", format="%.7f"),
                        "Error_abs": st.column_config.NumberColumn("Error_abs", format="%.7f"),
                        "Cota_trunc": st.column_config.NumberColumn("Cota_trunc", format="%.7f"),
                    },
                )

                fig_c, ax_c = plt.subplots(figsize=(8.5, 4.2))
                ok_mask = df_comp["Integral"].notna()
                df_ok = df_comp.loc[ok_mask, ["Metodo", "Integral"]].copy()
                posiciones = np.arange(len(df_ok))
                ax_c.bar(posiciones, df_ok["Integral"].astype(float), width=0.6)
                ax_c.set_xticks(posiciones)
                ax_c.set_xticklabels(df_ok["Metodo"], rotation=20, ha="right")
                ax_c.set_title("Comparativa de integrales por metodo")
                ax_c.set_xlabel("Metodo")
                ax_c.set_ylabel("Valor de integral")
                ax_c.grid(axis="y", alpha=0.3)
                fig_c.tight_layout()
                render_chart(fig_c)
                plt.close(fig_c)

            if analizar_convergencia:
                n_vals = np.arange(2, int(n_max) + 1)
                curves = {"Rectangulo": [], "Trapecio": [], "Simpson 1/3": [], "Simpson 3/8": []}

                if exact_val is None:
                    n_ref = max(800, int(n_max) * 20)
                    ref_val = integrar_numerica_soporte_infinito(func, a_val, b_val, n_ref, "Trapecio")["valor"]
                else:
                    ref_val = exact_val

                for ni in n_vals:
                    try:
                        v_r = integrar_numerica_soporte_infinito(func, a_val, b_val, int(ni), "Rectangulo")["valor"]
                        curves["Rectangulo"].append(abs(float(v_r) - float(ref_val)))
                    except Exception:
                        curves["Rectangulo"].append(np.nan)

                    try:
                        v_t = integrar_numerica_soporte_infinito(func, a_val, b_val, int(ni), "Trapecio")["valor"]
                        curves["Trapecio"].append(abs(float(v_t) - float(ref_val)))
                    except Exception:
                        curves["Trapecio"].append(np.nan)

                    if ni % 2 == 0:
                        try:
                            v_s13 = integrar_numerica_soporte_infinito(func, a_val, b_val, int(ni), "Simpson 1/3")["valor"]
                            curves["Simpson 1/3"].append(abs(float(v_s13) - float(ref_val)))
                        except Exception:
                            curves["Simpson 1/3"].append(np.nan)
                    else:
                        curves["Simpson 1/3"].append(np.nan)

                    if ni % 3 == 0:
                        try:
                            v_s38 = integrar_numerica_soporte_infinito(func, a_val, b_val, int(ni), "Simpson 3/8")["valor"]
                            curves["Simpson 3/8"].append(abs(float(v_s38) - float(ref_val)))
                        except Exception:
                            curves["Simpson 3/8"].append(np.nan)
                    else:
                        curves["Simpson 3/8"].append(np.nan)

                fig_conv, ax_conv = plt.subplots(figsize=(8.8, 4.4))
                for nombre, vals in curves.items():
                    arr = np.array(vals, dtype=float)
                    ok = np.isfinite(arr)
                    if np.any(ok):
                        ax_conv.semilogy(n_vals[ok], arr[ok], "o-", linewidth=1.8, label=nombre)
                ax_conv.set_title("Convergencia del error vs n")
                ax_conv.set_xlabel("Numero de intervalos n")
                ax_conv.set_ylabel("Error absoluto")
                ax_conv.grid(alpha=0.3, which="both")
                ax_conv.legend()
                render_chart(fig_conv)
                plt.close(fig_conv)

            cuentas = [rf"I_{{aprox}}={_num(valor, 12)}"]
            if np.isfinite(a_val) and np.isfinite(b_val):
                h_val = (float(b_val) - float(a_val)) / int(n)
                cuentas.insert(0, rf"h=\frac{{b-a}}{{n}}=\frac{{{_num(b_val)}-{_num(a_val)}}}{{{int(n)}}}={_num(h_val)}")
            else:
                cuentas.insert(0, rf"\text{{Integral impropia en }}{resultado_int['tipo']}\text{{, via transformacion de variable}}")
            if exact_val is not None:
                cuentas.append(rf"I_{{ref}}={_num(exact_val, 12)}")
                cuentas.append(rf"|E|=|I_{{aprox}}-I_{{ref}}|={_num(abs(float(valor)-float(exact_val)))}")
            if np.isfinite(cota_sel):
                cuentas.append(rf"|E_T|\le {_num(cota_sel)}")
            guardar_cuentas("Integracion Numerica", cuentas)

        except Exception as exc:
            st.error(f"Error en integracion numerica: {exc}")


def section_montecarlo():
    st.subheader("Integracion por Monte Carlo")

    with st.form("form_montecarlo"):
        c1, c2, c3 = st.columns(3)
        with c1:
            func = st.text_input("f(x)", value="x**2")
            a_text = st.text_input("Limite inferior a", value="0")
            b_text = st.text_input("Limite superior b", value="1")
        with c2:
            n = st.number_input("Cantidad de puntos n", value=1000, min_value=10, step=10)
            confianza = st.number_input("Intervalo de confianza (%)", value=95.0, min_value=80.0, max_value=99.9, step=0.1)
            usar_seed = st.checkbox("Usar semilla personalizada", value=False)
            seed = st.number_input("Semilla (seed)", value=42, min_value=0, step=1) if usar_seed else None
        with c3:
            usar_error_max = st.checkbox("Usar error máximo permitido", value=False)
            error_max = st.number_input("Error máximo permitido", value=0.01, min_value=0.0001, format="%.6f") if usar_error_max else None

        run_btn = st.form_submit_button("Calcular integral")

    if run_btn:
        try:
            a_val = parse_numeric_expr(a_text, "a")
            b_val = parse_numeric_expr(b_text, "b")

            if b_val <= a_val:
                st.error("Se requiere b > a.")
                return

            # Calcular integral por Monte Carlo
            if usar_error_max and error_max is not None:
                # Modo iterativo: continuar hasta alcanzar el error máximo
                integral = 0.0
                y_all = np.array([])
                x_all = np.array([])
                total_puntos = 0
                iteracion = 0
                current_seed = seed if usar_seed else None
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while True:
                    iteracion += 1
                    if current_seed is not None:
                        current_seed += iteracion - 1
                    
                    integral_iter, std_iter, x_nodes, y_nodes = regla_montecarlo(
                        func, a_val, b_val, int(n), seed=current_seed
                    )
                    
                    # Acumular puntos
                    x_all = np.append(x_all, x_nodes)
                    y_all = np.append(y_all, y_nodes)
                    total_puntos += len(x_nodes)
                    
                    # Recalcular la integral y el error con todos los puntos acumulados
                    integral = (b_val - a_val) * np.mean(y_all)
                    var_f = np.var(y_all, ddof=1)
                    std = np.sqrt(((b_val - a_val) ** 2 / total_puntos) * var_f)
                    
                    progress = min(std / error_max, 1.0)
                    progress_bar.progress(progress)
                    status_text.info(f"Iteración {iteracion}: {total_puntos} puntos, Error = {std:.7f}, Objetivo = {error_max:.7f}")
                    
                    if std <= error_max:
                        status_text.success(f"✓ Convergencia alcanzada en iteración {iteracion} con {total_puntos} puntos")
                        break
                    
                    if iteracion > 100:
                        st.warning(f"Se alcanzó el límite de 100 iteraciones. Error actual: {std:.7f} (objetivo: {error_max:.7f})")
                        break
                
                # Ordenar puntos para visualizar
                sort_idx = np.argsort(x_all)
                x_nodes = x_all[sort_idx]
                y_nodes = y_all[sort_idx]
            else:
                # Modo estándar: usar n puntos
                integral, std, x_nodes, y_nodes = regla_montecarlo(func, a_val, b_val, int(n), seed=seed if usar_seed else None)

            # Calcular error estándar
            if usar_error_max and error_max is not None:
                num_puntos = total_puntos
            else:
                num_puntos = int(n)
            
            error_estandar = std / np.sqrt(num_puntos)
            
            # Calcular intervalo de confianza
            # Usando distribución normal, z para el nivel de confianza
            alpha = (100 - confianza) / 100
            # Aproximación simple para z
            if abs(confianza - 95) < 0.1:
                z = 1.96
            elif abs(confianza - 99) < 0.1:
                z = 2.576
            elif abs(confianza - 90) < 0.1:
                z = 1.645
            else:
                # Aproximación general: z ≈ sqrt(2) * erfinv(1 - alpha)
                # Pero usar scipy
                from scipy.special import erfinv
                from math import sqrt
                z = sqrt(2) * erfinv(1 - alpha)
            margin = z * std
            lower = integral - margin
            upper = integral + margin

            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("Integral aproximada", f"{integral:.7g}")
            c_m2.metric("Desviacion estandar", f"{std:.7g}")
            c_m3.metric("Error estándar", f"{error_estandar:.7g}")
            c_m4.metric(f"IC {confianza}%", f"[{lower:.7g}, {upper:.7g}]")
            
            # Mostrar IC en formato ± 
            st.write(f"### IC {confianza}% (formato ±): {integral:.7g} ± {margin:.7g}")

            # Mostrar puntos
            df_puntos = pd.DataFrame({"x": x_nodes, "f(x)": y_nodes})
            st.dataframe(
                df_puntos.head(50),  # Mostrar solo primeros 50 para no sobrecargar
                use_container_width=True,
                column_config={k: st.column_config.NumberColumn(k, format="%.7f") for k in df_puntos.columns},
            )
            if len(x_nodes) > 50:
                st.info(f"Mostrando 50 de {len(x_nodes)} puntos. Todos los puntos se usaron en el calculo.")

            # Graficar
            fig, ax = plt.subplots(figsize=(9, 4.8))
            x_plot = np.linspace(float(a_val), float(b_val), 1000)
            y_plot = build_func_plot(func, x_plot)
            ax.plot(x_plot, y_plot, linewidth=2, label=f"f(x) = {func}")
            ax.scatter(x_nodes, y_nodes, color="red", s=10, alpha=0.5, label="Puntos aleatorios")
            ax.axhline(np.mean(y_nodes), color="blue", linestyle="--", label=f"Promedio f(x) ≈ {np.mean(y_nodes):.4f}")
            ax.set_title("Integracion por Monte Carlo")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(alpha=0.3)
            ax.legend()
            render_chart(fig)
            plt.close(fig)

            promedio_fx = float(np.mean(y_nodes))
            cuentas = [
                rf"I_{{MC}}=(b-a)\,\overline{{f}}",
                rf"a={_num(a_val)},\ b={_num(b_val)},\ n={int(num_puntos)}",
                rf"\overline{{f}}={_num(promedio_fx, 12)}",
                rf"I_{{MC}}={_num(integral, 12)}",
                rf"\hat{{\sigma}}_I={_num(std, 12)}",
                rf"IC_{{{_num(confianza, 6)}\%}}=[{_num(lower, 12)},\ {_num(upper, 12)}]",
            ]
            if usar_error_max and error_max is not None:
                cuentas.append(rf"n_{{efectivo}}={int(num_puntos)}\ \text{{(acumulado por iteraciones)}}")
                cuentas.append(rf"\text{{objetivo de error}}={_num(error_max, 12)}")
            guardar_cuentas("Monte Carlo", cuentas)

        except Exception as exc:
            st.error(f"Error en Monte Carlo: {exc}")


def section_montecarlo_2d():
    st.subheader("Integracion Doble por Monte Carlo")

    with st.form("form_montecarlo_2d"):
        c1, c2, c3 = st.columns(3)
        with c1:
            func = st.text_input("f(x,y)", value="x**2 + y**2")
            a_text = st.text_input("Limite inferior a (x)", value="0")
            b_text = st.text_input("Limite superior b (x)", value="1")
        with c2:
            c_text = st.text_input("Limite inferior c (y)", value="0")
            d_text = st.text_input("Limite superior d (y)", value="1")
            n = st.number_input("Cantidad de puntos n", value=1000, min_value=10, step=10)
        with c3:
            confianza = st.number_input("Intervalo de confianza (%)", value=95.0, min_value=80.0, max_value=99.9, step=0.1)
            usar_seed = st.checkbox("Usar semilla personalizada", value=False)
            seed = st.number_input("Semilla (seed)", value=42, min_value=0, step=1) if usar_seed else None
        
        run_btn = st.form_submit_button("Calcular integral doble")

    if run_btn:
        try:
            a_val = parse_numeric_expr(a_text, "a")
            b_val = parse_numeric_expr(b_text, "b")
            c_val = parse_numeric_expr(c_text, "c")
            d_val = parse_numeric_expr(d_text, "d")

            if b_val <= a_val or d_val <= c_val:
                st.error("Se requiere b > a y d > c.")
                return

            # Calcular integral doble por Monte Carlo
            integral, std, x_nodes, y_nodes, z_nodes = regla_montecarlo_2d(
                func, a_val, b_val, c_val, d_val, int(n), seed=seed if usar_seed else None
            )

            # Calcular error estándar
            num_puntos = int(n)
            error_estandar = std / np.sqrt(num_puntos)

            # Calcular intervalo de confianza
            alpha = (100 - confianza) / 100
            if abs(confianza - 95) < 0.1:
                z = 1.96
            elif abs(confianza - 99) < 0.1:
                z = 2.576
            elif abs(confianza - 90) < 0.1:
                z = 1.645
            else:
                from scipy.special import erfinv
                from math import sqrt
                z = sqrt(2) * erfinv(1 - alpha)
            margin = z * std
            lower = integral - margin
            upper = integral + margin

            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            c_m1.metric("Integral doble aproximada", f"{integral:.7g}")
            c_m2.metric("Desviacion estandar", f"{std:.7g}")
            c_m3.metric("Error estándar", f"{error_estandar:.7g}")
            c_m4.metric(f"IC {confianza}%", f"[{lower:.7g}, {upper:.7g}]")

            st.write(f"### IC {confianza}% (formato ±): {integral:.7g} ± {margin:.7g}")

            # Mostrar puntos
            df_puntos = pd.DataFrame({"x": x_nodes, "y": y_nodes, "f(x,y)": z_nodes})
            st.dataframe(
                df_puntos.head(50),
                use_container_width=True,
                column_config={k: st.column_config.NumberColumn(k, format="%.7f") for k in df_puntos.columns},
            )
            if len(x_nodes) > 50:
                st.info(f"Mostrando 50 de {len(x_nodes)} puntos. Todos los puntos se usaron en el calculo.")

            # Graficar puntos en el plano xy
            fig, ax = plt.subplots(figsize=(9, 4.8))
            scatter = ax.scatter(x_nodes, y_nodes, c=z_nodes, cmap="viridis", s=10, alpha=0.6)
            ax.set_title(f"Puntos aleatorios en [${a_val},{b_val}$] × [${c_val},{d_val}$]")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f"f(x,y)")
            ax.grid(alpha=0.3)
            render_chart(fig)
            plt.close(fig)

            area = (b_val - a_val) * (d_val - c_val)
            promedio_fxy = float(np.nanmean(z_nodes))
            cuentas = [
                rf"A=(b-a)(d-c)=({_num(b_val)}-{_num(a_val)})({_num(d_val)}-{_num(c_val)})={_num(area, 12)}",
                rf"I_{{MC2D}}=A\,\overline{{f}},\ \overline{{f}}=\frac{{1}}{{n}}\sum_{{i=1}}^n f(x_i,y_i)",
                rf"\overline{{f}}={_num(promedio_fxy, 12)}",
                rf"I_{{MC2D}}={_num(integral, 12)}",
                rf"\hat{{\sigma}}_I={_num(std, 12)}",
                rf"IC_{{{_num(confianza, 6)}\%}}=[{_num(lower, 12)},\ {_num(upper, 12)}]",
            ]
            guardar_cuentas("Monte Carlo 2D", cuentas)

        except Exception as exc:
            st.error(f"Error en integral doble: {exc}")


def section_ajuste_curvas():
    st.subheader("Ajuste de curvas (minimos cuadrados)")

    with st.form("form_ajuste"):
        c1, c2, c3 = st.columns(3)
        with c1:
            x_text = st.text_input("x (separados por coma)", value="0, 1, 2, 3, 4")
            y_text = st.text_input("y (separados por coma)", value="1, 2.1, 2.9, 3.8, 5.2")
        with c2:
            tipo = st.selectbox("Tipo de regresion", ["Lineal", "Polinomial"])
            grado = st.number_input("Grado (si es polinomial)", value=2, min_value=1, step=1)
        with c3:
            mostrar_residuos = st.checkbox("Mostrar analisis de residuos", value=True)
            explorar_grados = st.checkbox("Explorar grados (polinomial)", value=True)
        run_btn = st.form_submit_button("Calcular ajuste")

    if run_btn:
        try:
            x_vals = to_float_array(parse_expr_list(x_text))
            y_vals = to_float_array(parse_expr_list(y_text))

            if tipo == "Lineal":
                result = regresion_lineal(x_vals, y_vals)
            else:
                result = regresion_polinomial(x_vals, y_vals, int(grado))

            y_fit = np.array(result["y_pred"], dtype=float)
            resid = y_vals - y_fit
            mae = float(np.mean(np.abs(resid)))
            rmse = float(np.sqrt(np.mean(resid ** 2)))

            st.write(f"Ecuacion de mejor ajuste: {result['ecuacion']}")
            c_r1, c_r2, c_r3 = st.columns(3)
            c_r1.metric("R^2", f"{result['r2']:.7f}")
            c_r2.metric("MAE", f"{mae:.6g}")
            c_r3.metric("RMSE", f"{rmse:.6g}")

            df = pd.DataFrame(
                {
                    "x": x_vals,
                    "y_real": y_vals,
                    "y_ajustada": y_fit,
                    "residuo": resid,
                }
            )
            st.dataframe(df, use_container_width=True)

            x_curve = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), 900)
            coef = np.array(result["coeficientes"], dtype=float)
            if result["tipo"] == "lineal":
                y_curve = coef[0] * x_curve + coef[1]
            else:
                y_curve = np.polyval(coef, x_curve)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.scatter(x_vals, y_vals, color="tab:red", s=50, zorder=3, label="Datos")
            ax.plot(x_curve, y_curve, color="tab:blue", linewidth=2, label="Curva de ajuste")
            ax.set_title("Regresion por minimos cuadrados")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.3)
            ax.legend()
            render_chart(fig)
            plt.close(fig)

            if mostrar_residuos:
                fig_r, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
                ax_r1.axhline(0, color="black", linewidth=0.8)
                ax_r1.scatter(x_vals, resid, color="tab:purple", s=45)
                ax_r1.plot(x_vals, resid, color="tab:purple", alpha=0.5)
                ax_r1.set_title("Residuos vs x")
                ax_r1.set_xlabel("x")
                ax_r1.set_ylabel("Residuo")
                ax_r1.grid(alpha=0.3)

                ax_r2.hist(resid, bins=min(10, max(4, len(resid))), color="tab:orange", edgecolor="black", alpha=0.8)
                ax_r2.set_title("Distribucion de residuos")
                ax_r2.set_xlabel("Residuo")
                ax_r2.set_ylabel("Frecuencia")
                ax_r2.grid(alpha=0.25, axis="y")
                render_chart(fig_r)
                plt.close(fig_r)

            if tipo == "Polinomial" and explorar_grados:
                max_deg = min(8, len(x_vals) - 1)
                if max_deg >= 1:
                    grados = np.arange(1, max_deg + 1)
                    rmse_vals = []
                    r2_vals = []
                    for g in grados:
                        res_g = regresion_polinomial(x_vals, y_vals, int(g))
                        y_g = np.array(res_g["y_pred"], dtype=float)
                        err_g = y_vals - y_g
                        rmse_vals.append(float(np.sqrt(np.mean(err_g ** 2))))
                        r2_vals.append(float(res_g["r2"]))

                    fig_g, ax_g1 = plt.subplots(figsize=(8.8, 4.2))
                    ax_g1.plot(grados, rmse_vals, "o-", color="tab:red", label="RMSE")
                    ax_g1.set_xlabel("Grado polinomial")
                    ax_g1.set_ylabel("RMSE", color="tab:red")
                    ax_g1.tick_params(axis="y", labelcolor="tab:red")
                    ax_g1.grid(alpha=0.3)

                    ax_g2 = ax_g1.twinx()
                    ax_g2.plot(grados, r2_vals, "s--", color="tab:blue", label="R^2")
                    ax_g2.set_ylabel("R^2", color="tab:blue")
                    ax_g2.tick_params(axis="y", labelcolor="tab:blue")

                    ax_g1.set_title("Sensibilidad del ajuste al grado")
                    render_chart(fig_g)
                    plt.close(fig_g)

            cuentas = [
                rf"\text{{Ecuacion ajustada}}: {result['ecuacion']}",
                rf"RMSE={_num(rmse)},\ MAE={_num(mae)},\ R^2={_num(result['r2'])}",
                rf"r_1=y_1-\hat y_1={_num(resid[0])}",
            ]
            guardar_cuentas("Ajuste de Curvas", cuentas)

        except Exception as exc:
            st.error(f"Error en ajuste de curvas: {exc}")


def section_sistemas_lineales():
    st.subheader("Resolucion de sistemas lineales")

    with st.form("form_sistemas"):
        n = st.number_input("Dimension n", value=3, min_value=1, step=1)
        mat_default = "4,1,2\n3,5,1\n1,1,3"
        vec_default = "4\n7\n3"

        c1, c2 = st.columns(2)
        with c1:
            A_text = st.text_area("Matriz A (una fila por linea, separada por comas)", value=mat_default, height=140)
            b_text = st.text_area("Vector b (un valor por linea)", value=vec_default, height=140)
        with c2:
            metodo = st.selectbox("Metodo", ["Gauss-Jordan", "Gauss-Seidel"])
            x0_text = st.text_input("x0 para Gauss-Seidel (opcional, csv)", value="")
            tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
            max_iter = st.number_input("Max iteraciones", value=100, min_value=1, step=1)
            mostrar_pasos = st.checkbox("Mostrar pasos/iteraciones", value=True)

        run_btn = st.form_submit_button("Resolver sistema")

    if run_btn:
        try:
            A = parse_matrix_text(A_text, int(n))
            b = parse_vector_text(b_text, int(n))

            # Diagnostico visual de la matriz A
            fig_hm, ax_hm = plt.subplots(figsize=(5.4, 4.4))
            im = ax_hm.imshow(np.abs(A), cmap="YlOrRd", aspect="auto")
            ax_hm.set_title("Heatmap |A|")
            ax_hm.set_xlabel("Columna")
            ax_hm.set_ylabel("Fila")
            fig_hm.colorbar(im, ax=ax_hm, shrink=0.85)
            render_chart(fig_hm)
            plt.close(fig_hm)

            diag = np.abs(np.diag(A))
            off_sum = np.sum(np.abs(A), axis=1) - diag
            dd_margin = diag - off_sum
            dd_ok = bool(np.all(dd_margin > 0))
            st.metric("Diagonal dominante estricta", "Si" if dd_ok else "No")

            fig_dd, ax_dd = plt.subplots(figsize=(7.8, 3.8))
            ax_dd.axhline(0, color="black", linewidth=0.8)
            ax_dd.bar(np.arange(1, int(n) + 1), dd_margin, color=["tab:green" if m > 0 else "tab:red" for m in dd_margin])
            ax_dd.set_title("Margen de dominancia diagonal por fila: |a_ii| - sum(|a_ij|)")
            ax_dd.set_xlabel("Fila")
            ax_dd.set_ylabel("Margen")
            ax_dd.grid(axis="y", alpha=0.3)
            render_chart(fig_dd)
            plt.close(fig_dd)

            if metodo == "Gauss-Jordan":
                sol, pasos = gauss_jordan(A, b)
                df_sol = pd.DataFrame([sol], columns=[f"x{i + 1}" for i in range(len(sol))])
                st.dataframe(df_sol, use_container_width=True)

                if mostrar_pasos:
                    with st.expander("Ver pasos de Gauss-Jordan", expanded=False):
                        for i, paso in enumerate(pasos, start=1):
                            st.write(f"Paso {i}: {paso['paso']}")
                            cols = [f"x{j + 1}" for j in range(int(n))] + ["b"]
                            st.dataframe(pd.DataFrame(paso["matriz"], columns=cols), use_container_width=True)

            else:
                x0 = parse_optional_vector_csv(x0_text, int(n))
                sol, iters, convergio = gauss_seidel(
                    A,
                    b,
                    x0=x0,
                    tol=float(tol),
                    max_iter=int(max_iter),
                )
                df_sol = pd.DataFrame([sol], columns=[f"x{i + 1}" for i in range(len(sol))])
                st.dataframe(df_sol, use_container_width=True)
                st.metric("Convergencia", "Si" if convergio else "No")
                st.metric("Iteraciones", len(iters))

                if mostrar_pasos and iters:
                    df_iters = pd.DataFrame(iters)
                    st.dataframe(df_iters, use_container_width=True)

                    fig_s1, ax_s1 = plt.subplots(figsize=(8.2, 4.0))
                    err_vals = df_iters["Error_inf"].to_numpy(dtype=float)
                    iter_vals = df_iters["Iteracion"].to_numpy(dtype=float)

                    # La escala logaritmica solo admite valores positivos.
                    if np.all(err_vals > 0):
                        ax_s1.semilogy(iter_vals, err_vals, "o-", linewidth=2)
                    else:
                        ax_s1.plot(iter_vals, err_vals, "o-", linewidth=2)
                    ax_s1.set_title("Convergencia Gauss-Seidel (Error infinito)")
                    ax_s1.set_xlabel("Iteracion")
                    ax_s1.set_ylabel("Error_inf")
                    ax_s1.grid(alpha=0.3, which="both")
                    render_chart(fig_s1)
                    plt.close(fig_s1)

                    fig_s2, ax_s2 = plt.subplots(figsize=(8.6, 4.1))
                    x_cols = [c for c in df_iters.columns if c.startswith("x")]
                    for col in x_cols:
                        ax_s2.plot(df_iters["Iteracion"], df_iters[col], "o-", linewidth=1.5, label=col)
                    ax_s2.set_title("Evolucion de variables por iteracion")
                    ax_s2.set_xlabel("Iteracion")
                    ax_s2.set_ylabel("Valor")
                    ax_s2.grid(alpha=0.3)
                    ax_s2.legend()
                    render_chart(fig_s2)
                    plt.close(fig_s2)

            cuentas = [rf"A\,x=b\ \text{{con }}n={int(n)}"]
            if metodo == "Gauss-Jordan":
                cuentas.append(r"[A|b]\rightarrow[I|x]")
                cuentas.append(rf"x_1={_num(sol[0])}")
            else:
                cuentas.append(rf"\text{{Iteraciones Seidel}}={len(iters)}")
                if iters:
                    cuentas.append(rf"\|e\|_\infty^{{(final)}}={_num(iters[-1]['Error_inf'])}")
            guardar_cuentas("Sistemas Lineales", cuentas)

        except Exception as exc:
            st.error(f"Error al resolver sistema lineal: {exc}")


def section_edo():
    st.subheader("EDO de valor inicial")

    with st.form("form_edo"):
        c1, c2, c3 = st.columns(3)
        with c1:
            fxy = st.text_input("y' = f(x, y)", value="x + y")
            x0 = st.number_input("x0", value=0.0)
            y0 = st.number_input("y0", value=1.0)
        with c2:
            h = st.number_input("Paso h", value=0.1)
            n = st.number_input("Cantidad de pasos n", value=10, min_value=1, step=1)
            metodo = st.selectbox("Metodo", ["Euler", "RK4"])
        with c3:
            comparar_euler_rk4 = st.checkbox("Comparar Euler vs RK4", value=True)
            mostrar_campo = st.checkbox("Mostrar campo de pendientes", value=True)
            y_exact_text = st.text_input("y(x) exacta opcional", value="")
        run_btn = st.form_submit_button("Resolver EDO")

    if run_btn:
        try:
            rows_euler = metodo_euler(fxy, float(x0), float(y0), float(h), int(n))
            rows_rk4 = metodo_rk4(fxy, float(x0), float(y0), float(h), int(n))

            if metodo == "Euler":
                rows = rows_euler
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
            else:
                rows = rows_rk4
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

            x_num_e = np.array([r["x"] for r in rows_euler], dtype=float)
            y_num_e = np.array([r["y"] for r in rows_euler], dtype=float)
            x_num_r = np.array([r["x"] for r in rows_rk4], dtype=float)
            y_num_r = np.array([r["y"] for r in rows_rk4], dtype=float)

            x_end = float(x_num_e[-1])
            st.metric("Resultado Euler", f"y({x_end:.7g}) = {float(y_num_e[-1]):.12g}")
            st.metric("Resultado RK4", f"y({x_end:.7g}) = {float(y_num_r[-1]):.12g}")

            exact_vals = None
            if y_exact_text.strip():
                try:
                    expr_exact = safe_eval_expr(y_exact_text, "x")
                    exact_fun = sp.lambdify(sp.Symbol("x"), expr_exact, "numpy")
                    exact_vals = np.array(exact_fun(x_num_r), dtype=float)
                except Exception as exc:
                    st.warning(f"No se pudo evaluar y(x) exacta: {exc}")

            fig, ax = plt.subplots(figsize=(9.5, 4.8))
            if comparar_euler_rk4:
                ax.plot(x_num_e, y_num_e, "o-", linewidth=1.8, label="Euler")
                ax.plot(x_num_r, y_num_r, "s-", linewidth=1.8, label="RK4")
            else:
                if metodo == "Euler":
                    ax.plot(x_num_e, y_num_e, "o-", linewidth=2, label="Euler")
                else:
                    ax.plot(x_num_r, y_num_r, "s-", linewidth=2, label="RK4")

            if exact_vals is not None:
                ax.plot(x_num_r, exact_vals, "--", linewidth=2, label="Exacta")

            ax.set_title("Trayectoria de solucion numérica")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.3)
            ax.legend()
            render_chart(fig)
            plt.close(fig)

            if exact_vals is not None:
                err_e = np.abs(y_num_e - exact_vals)
                err_r = np.abs(y_num_r - exact_vals)
                fig_er, ax_er = plt.subplots(figsize=(8.8, 4.2))
                ax_er.semilogy(x_num_e, np.clip(err_e, 1e-16, None), "o-", label="Error Euler")
                ax_er.semilogy(x_num_r, np.clip(err_r, 1e-16, None), "s-", label="Error RK4")
                ax_er.set_title("Error absoluto vs x")
                ax_er.set_xlabel("x")
                ax_er.set_ylabel("|error|")
                ax_er.grid(alpha=0.3, which="both")
                ax_er.legend()
                render_chart(fig_er)
                plt.close(fig_er)

            if mostrar_campo:
                _, f_eval = construir_funcion_edo(fxy)
                x_min = float(min(x_num_e.min(), x_num_r.min()))
                x_max = float(max(x_num_e.max(), x_num_r.max()))
                y_all = np.concatenate([y_num_e, y_num_r])
                y_min = float(np.min(y_all))
                y_max = float(np.max(y_all))
                pad_y = max(0.5, 0.2 * (y_max - y_min + 1e-12))

                xx = np.linspace(x_min, x_max, 20)
                yy = np.linspace(y_min - pad_y, y_max + pad_y, 20)
                Xg, Yg = np.meshgrid(xx, yy)
                S = np.array(f_eval(Xg, Yg), dtype=float)
                U = np.ones_like(S)
                V = S
                N = np.sqrt(U ** 2 + V ** 2)
                U = U / np.where(N == 0, 1, N)
                V = V / np.where(N == 0, 1, N)

                fig_fld, ax_fld = plt.subplots(figsize=(9.2, 4.8))
                ax_fld.quiver(Xg, Yg, U, V, N, cmap="viridis", alpha=0.75)
                ax_fld.plot(x_num_e, y_num_e, "o-", linewidth=1.4, label="Euler")
                ax_fld.plot(x_num_r, y_num_r, "s-", linewidth=1.4, label="RK4")
                ax_fld.set_title("Campo de pendientes y trayectorias")
                ax_fld.set_xlabel("x")
                ax_fld.set_ylabel("y")
                ax_fld.grid(alpha=0.25)
                ax_fld.legend()
                render_chart(fig_fld)
                plt.close(fig_fld)

            cuentas = [
                rf"x_0={_num(x0)},\ y_0={_num(y0)},\ h={_num(h)},\ n={int(n)}",
                rf"y_{{Euler}}(x_f)={_num(y_num_e[-1], 12)}",
                rf"y_{{RK4}}(x_f)={_num(y_num_r[-1], 12)}",
            ]
            if exact_vals is not None:
                cuentas.append(rf"|e_{{RK4}}(x_f)|={_num(abs(y_num_r[-1]-exact_vals[-1]))}")
            guardar_cuentas("EDO", cuentas)

        except Exception as exc:
            st.error(f"Error en EDO: {exc}")


def section_red_neuronal_descenso():
    st.subheader("Red neuronal base con descenso de gradiente")

    with st.form("form_red_descenso"):
        c1, c2 = st.columns(2)
        with c1:
            alpha = st.number_input("Tasa de aprendizaje (alpha)", value=0.03, min_value=1e-6, format="%.4f")
            epocas = st.number_input("Epocas", value=120, min_value=1, step=1)
        with c2:
            semilla = st.number_input("Semilla de pesos iniciales", value=7, step=1)
            interval_ms = st.number_input("Intervalo animacion (ms)", value=110, min_value=20, step=10)

        mostrar_anim = st.checkbox("Mostrar animacion en el dashboard", value=False)
        run_btn = st.form_submit_button("Entrenar modelo")

    if run_btn:
        try:
            x, y = dataset_prueba_pequeno()
            train = entrenar_descenso_gradiente_lineal(
                x,
                y,
                alpha=float(alpha),
                epocas=int(epocas),
                semilla=int(semilla),
            )

            st.metric("Costo final (MSE)", f"{float(train['hist_costo'][-1]):.8f}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("w inicial", f"{train['w0']:.6f}")
            c2.metric("b inicial", f"{train['b0']:.6f}")
            c3.metric("w final", f"{train['w_final']:.6f}")
            c4.metric("b final", f"{train['b_final']:.6f}")

            df_hist = pd.DataFrame(
                {
                    "epoca": np.arange(1, len(train["hist_costo"]) + 1),
                    "costo": train["hist_costo"],
                    "w": train["hist_w"][1:],
                    "b": train["hist_b"][1:],
                }
            )
            st.dataframe(df_hist, use_container_width=True)

            fig_static, _ = figura_tres_subplots_descenso(x, y, train, animar=False)
            render_chart(fig_static)
            plt.close(fig_static)

            if mostrar_anim:
                fig_anim, anim = figura_tres_subplots_descenso(
                    x,
                    y,
                    train,
                    animar=True,
                    interval_ms=int(interval_ms),
                )
                if anim is not None:
                    components.html(anim.to_jshtml(), height=540, scrolling=True)
                plt.close(fig_anim)

            cuentas = [
                rf"\hat y = wx+b",
                rf"w_0={_num(train['w0'])},\ b_0={_num(train['b0'])}",
                rf"w_f={_num(train['w_final'])},\ b_f={_num(train['b_final'])}",
                rf"J_f={_num(train['hist_costo'][-1])}",
            ]
            guardar_cuentas("Red Neuronal GD", cuentas)

        except Exception as exc:
            st.error(f"Error en la simulacion de red neuronal: {exc}")


def main():
    st.set_page_config(page_title="Dashboard Integrador de Metodos", layout="wide")
    st.sidebar.selectbox(
        "Preset de color",
        ["Viva", "Pastel oscuro"],
        index=0,
        key="palette_preset",
        help="Cambia colores de acentos y series manteniendo fondo negro.",
    )
    st.sidebar.toggle(
        "Ver paso a paso en todos los metodos",
        value=False,
        key="show_step_by_step_all",
        help="Muestra desglose simbolico y numerico en cada apartado.",
    )

    paleta = paleta_activa()
    aplicar_tema_visual_dashboard(paleta)

    # Fondo blanco y alto contraste para todos los graficos de Matplotlib.
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.labelcolor"] = "black"
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=paleta["series"])

    header_left, header_right = st.columns([8, 2], vertical_alignment="center")
    with header_left:
        st.title("Dashboard Integrador de Metodos Numericos")
        st.caption("Interfaz grafica con formularios, botones, tablas y graficos de funcion/error")
    with header_right:
        logo_ferro, logo_estudiantes, logo_nuevo = st.columns(3)
        with logo_ferro:
            mostrar_imagen_encabezado("ferro.webp", ancho=110, texto_alternativo="Ferro")
        with logo_estudiantes:
            mostrar_imagen_encabezado("estudiantes.webp", ancho=110, texto_alternativo="Estudiantes")
        with logo_nuevo:
            mostrar_imagen_encabezado("uni.webp", ancho=110, texto_alternativo="Nuevo escudo")

    st.sidebar.toggle(
        "Graficos interactivos",
        value=True,
        key="interactive_charts",
        help="Convierte graficos de Matplotlib a Plotly para hacer zoom, paneo y hover.",
    )

    tabs = st.tabs(
        [
            "Newton-Raphson",
            "Aitken",
            "Biseccion",
            "Punto Fijo",
            "Comparativa",
            "Lagrange + Derivacion",
            "Integracion Numerica",
            "Monte Carlo",
            "Monte Carlo 2D",
            "Ajuste de Curvas",
            "Sistemas Lineales",
            "EDO",
            "Red Neuronal GD",
        ]
    )

    with tabs[0]:
        mostrar_machete("Newton-Raphson")
        section_newton()
        render_panel_formulas(
            "Formulario de Newton-Raphson",
            FORMULAS_POR_APARTADO["Newton-Raphson"],
            SIMBOLOS_POR_APARTADO["Newton-Raphson"],
            CONDICIONES_POR_APARTADO["Newton-Raphson"],
            PASOS_POR_APARTADO["Newton-Raphson"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Newton-Raphson"],
            "Newton-Raphson",
        )
    with tabs[1]:
        mostrar_machete("Aitken")
        section_aitken()
        render_panel_formulas(
            "Formulario de Aitken",
            FORMULAS_POR_APARTADO["Aitken"],
            SIMBOLOS_POR_APARTADO["Aitken"],
            CONDICIONES_POR_APARTADO["Aitken"],
            PASOS_POR_APARTADO["Aitken"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Aitken"],
            "Aitken",
        )
    with tabs[2]:
        mostrar_machete("Biseccion")
        section_biseccion()
        render_panel_formulas(
            "Formulario de Biseccion",
            FORMULAS_POR_APARTADO["Biseccion"],
            SIMBOLOS_POR_APARTADO["Biseccion"],
            CONDICIONES_POR_APARTADO["Biseccion"],
            PASOS_POR_APARTADO["Biseccion"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Biseccion"],
            "Biseccion",
        )
    with tabs[3]:
        mostrar_machete("Punto Fijo")
        section_punto_fijo()
        render_panel_formulas(
            "Formulario de Punto Fijo",
            FORMULAS_POR_APARTADO["Punto Fijo"],
            SIMBOLOS_POR_APARTADO["Punto Fijo"],
            CONDICIONES_POR_APARTADO["Punto Fijo"],
            PASOS_POR_APARTADO["Punto Fijo"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Punto Fijo"],
            "Punto Fijo",
        )
    with tabs[4]:
        st.markdown("### Comparación de 4 métodos de búsqueda de raíces")
        section_comparativa()
        render_panel_formulas(
            "Formulario de Comparativa",
            FORMULAS_POR_APARTADO["Comparativa"],
            SIMBOLOS_POR_APARTADO["Comparativa"],
            CONDICIONES_POR_APARTADO["Comparativa"],
            PASOS_POR_APARTADO["Comparativa"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Comparativa"],
            "Comparativa",
        )
    with tabs[5]:
        mostrar_machete("Lagrange + Derivacion")
        section_lagrange()
        render_panel_formulas(
            "Formulario de Lagrange y Derivacion",
            FORMULAS_POR_APARTADO["Lagrange + Derivacion"],
            SIMBOLOS_POR_APARTADO["Lagrange + Derivacion"],
            CONDICIONES_POR_APARTADO["Lagrange + Derivacion"],
            PASOS_POR_APARTADO["Lagrange + Derivacion"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Lagrange + Derivacion"],
            "Lagrange + Derivacion",
        )
    with tabs[6]:
        mostrar_machete("Integracion Numerica")
        section_integracion_numerica()
        render_panel_formulas(
            "Formulario de Integracion Numerica",
            FORMULAS_POR_APARTADO["Integracion Numerica"],
            SIMBOLOS_POR_APARTADO["Integracion Numerica"],
            CONDICIONES_POR_APARTADO["Integracion Numerica"],
            PASOS_POR_APARTADO["Integracion Numerica"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Integracion Numerica"],
            "Integracion Numerica",
        )
    with tabs[7]:
        section_montecarlo()
        render_panel_formulas(
            "Formulario de Integracion por Monte Carlo",
            FORMULAS_POR_APARTADO["Monte Carlo"],
            SIMBOLOS_POR_APARTADO["Monte Carlo"],
            CONDICIONES_POR_APARTADO["Monte Carlo"],
            PASOS_POR_APARTADO["Monte Carlo"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Monte Carlo"],
            "Monte Carlo",
        )
    with tabs[8]:
        section_montecarlo_2d()
        render_panel_formulas(
            "Formulario de Integracion Doble por Monte Carlo",
            FORMULAS_POR_APARTADO["Monte Carlo 2D"],
            SIMBOLOS_POR_APARTADO["Monte Carlo 2D"],
            CONDICIONES_POR_APARTADO["Monte Carlo 2D"],
            PASOS_POR_APARTADO["Monte Carlo 2D"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Monte Carlo 2D"],
            "Monte Carlo 2D",
        )
    with tabs[9]:
        mostrar_machete("Ajuste de Curvas")
        section_ajuste_curvas()
        render_panel_formulas(
            "Formulario de Ajuste de Curvas",
            FORMULAS_POR_APARTADO["Ajuste de Curvas"],
            SIMBOLOS_POR_APARTADO["Ajuste de Curvas"],
            CONDICIONES_POR_APARTADO["Ajuste de Curvas"],
            PASOS_POR_APARTADO["Ajuste de Curvas"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Ajuste de Curvas"],
            "Ajuste de Curvas",
        )
    with tabs[10]:
        mostrar_machete("Sistemas Lineales")
        section_sistemas_lineales()
        render_panel_formulas(
            "Formulario de Sistemas Lineales",
            FORMULAS_POR_APARTADO["Sistemas Lineales"],
            SIMBOLOS_POR_APARTADO["Sistemas Lineales"],
            CONDICIONES_POR_APARTADO["Sistemas Lineales"],
            PASOS_POR_APARTADO["Sistemas Lineales"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Sistemas Lineales"],
            "Sistemas Lineales",
        )
    with tabs[11]:
        mostrar_machete("EDO")
        section_edo()
        render_panel_formulas(
            "Formulario de EDO",
            FORMULAS_POR_APARTADO["EDO"],
            SIMBOLOS_POR_APARTADO["EDO"],
            CONDICIONES_POR_APARTADO["EDO"],
            PASOS_POR_APARTADO["EDO"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["EDO"],
            "EDO",
        )
    with tabs[12]:
        mostrar_machete("Red Neuronal GD")
        section_red_neuronal_descenso()
        render_panel_formulas(
            "Formulario de Red Neuronal GD",
            FORMULAS_POR_APARTADO["Red Neuronal GD"],
            SIMBOLOS_POR_APARTADO["Red Neuronal GD"],
            CONDICIONES_POR_APARTADO["Red Neuronal GD"],
            PASOS_POR_APARTADO["Red Neuronal GD"],
            st.session_state.get("show_step_by_step_all", False),
            DESGLOSE_COMPLETO_POR_APARTADO["Red Neuronal GD"],
            "Red Neuronal GD",
        )


def _is_streamlit_context():
    """Detecta si el script se ejecuta con `streamlit run`."""
    try:
        return st.runtime.exists()
    except Exception:
        return False


if __name__ == "__main__":
    if _is_streamlit_context():
        main()
    else:
        print("Este dashboard debe ejecutarse con Streamlit para evitar errores de contexto.")
        print("Usa este comando desde la carpeta del proyecto:")
        print("streamlit run dashboard_integrador.py")
        sys.exit(0)
