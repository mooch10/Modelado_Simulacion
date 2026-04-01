import io
import contextlib
import sys

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
    regla_rectangulo,
    regla_trapecio,
    regla_simpson_13,
    regla_simpson_38,
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


def parse_numeric_expr(expr_text, field_name):
    try:
        expr = sp.sympify(expr_text, locals=ALLOWED_LOCALS)
    except Exception as exc:
        raise ValueError(f"{field_name} invalido: {exc}") from exc

    if expr.free_symbols:
        raise ValueError(f"{field_name} no debe contener variables")

    value = float(sp.N(expr))
    if not np.isfinite(value):
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

    return np.nan


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


def render_chart(fig):
    """Renderiza Matplotlib en modo interactivo (Plotly) cuando esta disponible."""
    interactive = st.session_state.get("interactive_charts", True)

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
                legend=dict(font=dict(color="black"), title=dict(font=dict(color="black"))),
            )
            fig_plotly.update_xaxes(
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
                showline=True,
                linecolor="black",
                zerolinecolor="black",
                gridcolor="rgba(0,0,0,0.15)",
            )
            fig_plotly.update_yaxes(
                color="black",
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
                showline=True,
                linecolor="black",
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
            marker=dict(color="red", size=10),
        )
    )

    fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.6)
    fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.6)
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="f(x)",
        margin=dict(l=18, r=18, t=60, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        template="plotly_white",
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
        run_btn = st.form_submit_button("Ejecutar Newton")

    if run_btn:
        try:
            root, iterations, converged = run_silent(
                metodo_newton_raphson, func, float(x0), float(tol), int(max_iter)
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

            render_newton_charts(func, root, x0, errors)

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

        except Exception as exc:
            st.error(f"Error en diferencias divididas: {exc}")


def section_integracion_numerica():
    st.subheader("Integracion numerica")

    with st.form("form_integracion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            func = st.text_input("f(x)", value="sin(x)")
            a_text = st.text_input("Limite inferior a", value="0")
            b_text = st.text_input("Limite superior b", value="pi")
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
            a_val = parse_numeric_expr(a_text, "a")
            b_val = parse_numeric_expr(b_text, "b")

            if b_val <= a_val:
                st.error("Se requiere b > a.")
                return

            x_sym = sp.Symbol("x")
            exact_val = None
            expr = None
            if referencia_exacta:
                try:
                    expr = safe_eval_expr(func, "x")
                    exact_expr = sp.integrate(expr, (x_sym, a_val, b_val))
                    exact_val = float(sp.N(exact_expr))
                    if not np.isfinite(exact_val):
                        exact_val = None
                except Exception:
                    exact_val = None

            if expr is None:
                try:
                    expr = safe_eval_expr(func, "x")
                except Exception:
                    expr = None

            max_f2 = None
            max_f4 = None
            if expr is not None:
                try:
                    max_f2 = estimate_max_abs_derivative(expr, x_sym, 2, a_val, b_val)
                except Exception:
                    max_f2 = None
                try:
                    max_f4 = estimate_max_abs_derivative(expr, x_sym, 4, a_val, b_val)
                except Exception:
                    max_f4 = None

            if metodo == "Rectangulo":
                valor, x_nodes, y_nodes = regla_rectangulo(func, a_val, b_val, int(n))
            elif metodo == "Trapecio":
                valor, x_nodes, y_nodes = regla_trapecio(func, a_val, b_val, int(n))
            elif metodo == "Simpson 1/3":
                valor, x_nodes, y_nodes = regla_simpson_13(func, a_val, b_val, int(n))
            else:
                valor, x_nodes, y_nodes = regla_simpson_38(func, a_val, b_val, int(n))

            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Resultado", f"{valor:.12g}")
            c_m2.metric("Metodo", metodo)
            c_m3.metric("Intervalos n", int(n))

            cota_sel = cota_truncamiento_integracion(metodo, a_val, b_val, int(n), max_f2=max_f2, max_f4=max_f4)
            if np.isfinite(cota_sel):
                st.metric("Cota teorica de truncamiento", f"{float(cota_sel):.7f}")
            else:
                st.info("No se pudo estimar la cota teorica de truncamiento para este metodo.")

            if exact_val is not None:
                err_abs = abs(float(valor) - float(exact_val))
                err_rel = err_abs / abs(exact_val) if abs(exact_val) > 1e-15 else np.nan
                c_e1, c_e2, c_e3 = st.columns(3)
                c_e1.metric("Integral exacta", f"{exact_val:.12g}")
                c_e2.metric("Error absoluto", f"{err_abs:.7f}")
                c_e3.metric("Error relativo", "no definido" if np.isnan(err_rel) else f"{err_rel:.7f}")

            df_nodes = pd.DataFrame({"x_i": x_nodes, "f(x_i)": y_nodes})
            st.dataframe(
                df_nodes,
                use_container_width=True,
                column_config={
                    "x_i": st.column_config.NumberColumn("x_i", format="%.7f"),
                    "f(x_i)": st.column_config.NumberColumn("f(x_i)", format="%.7f"),
                },
            )

            _, f_num = build_numeric_function(func)
            x_plot = np.linspace(a_val, b_val, 900)
            y_plot = np.array(f_num(x_plot), dtype=float)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(x_plot, y_plot, linewidth=2, label=f"f(x) = {func}")
            ax.fill_between(x_plot, y_plot, 0, alpha=0.25, color="tab:blue", label="Area bajo la curva")
            ax.scatter(x_nodes, y_nodes, color="tab:red", zorder=3, label="Nodos")
            ax.plot(x_nodes, y_nodes, color="tab:red", alpha=0.7, linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_title(f"Integracion por {metodo}")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(alpha=0.3)
            ax.legend()
            render_chart(fig)
            plt.close(fig)

            if comparar_metodos:
                comp_rows = []
                for nombre in ["Rectangulo", "Trapecio", "Simpson 1/3", "Simpson 3/8"]:
                    try:
                        if nombre == "Rectangulo":
                            v, _, _ = regla_rectangulo(func, a_val, b_val, int(n))
                        elif nombre == "Trapecio":
                            v, _, _ = regla_trapecio(func, a_val, b_val, int(n))
                        elif nombre == "Simpson 1/3":
                            v, _, _ = regla_simpson_13(func, a_val, b_val, int(n))
                        else:
                            v, _, _ = regla_simpson_38(func, a_val, b_val, int(n))

                        row = {"Metodo": nombre, "Integral": float(v)}
                        cota = cota_truncamiento_integracion(
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
                    ref_val, _, _ = regla_trapecio(func, a_val, b_val, n_ref)
                else:
                    ref_val = exact_val

                for ni in n_vals:
                    try:
                        v_r, _, _ = regla_rectangulo(func, a_val, b_val, int(ni))
                        curves["Rectangulo"].append(abs(float(v_r) - float(ref_val)))
                    except Exception:
                        curves["Rectangulo"].append(np.nan)

                    try:
                        v_t, _, _ = regla_trapecio(func, a_val, b_val, int(ni))
                        curves["Trapecio"].append(abs(float(v_t) - float(ref_val)))
                    except Exception:
                        curves["Trapecio"].append(np.nan)

                    if ni % 2 == 0:
                        try:
                            v_s13, _, _ = regla_simpson_13(func, a_val, b_val, int(ni))
                            curves["Simpson 1/3"].append(abs(float(v_s13) - float(ref_val)))
                        except Exception:
                            curves["Simpson 1/3"].append(np.nan)
                    else:
                        curves["Simpson 1/3"].append(np.nan)

                    if ni % 3 == 0:
                        try:
                            v_s38, _, _ = regla_simpson_38(func, a_val, b_val, int(ni))
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

        except Exception as exc:
            st.error(f"Error en integracion numerica: {exc}")


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

        except Exception as exc:
            st.error(f"Error en la simulacion de red neuronal: {exc}")


def main():
    st.set_page_config(page_title="Dashboard Integrador de Metodos", layout="wide")

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

    st.title("Dashboard Integrador de Metodos Numericos")
    st.caption("Interfaz grafica con formularios, botones, tablas y graficos de funcion/error")

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
            "Ajuste de Curvas",
            "Sistemas Lineales",
            "EDO",
            "Red Neuronal GD",
        ]
    )

    with tabs[0]:
        section_newton()
    with tabs[1]:
        section_aitken()
    with tabs[2]:
        section_biseccion()
    with tabs[3]:
        section_punto_fijo()
    with tabs[4]:
        section_comparativa()
    with tabs[5]:
        section_lagrange()
    with tabs[6]:
        section_integracion_numerica()
    with tabs[7]:
        section_ajuste_curvas()
    with tabs[8]:
        section_sistemas_lineales()
    with tabs[9]:
        section_edo()
    with tabs[10]:
        section_red_neuronal_descenso()


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
