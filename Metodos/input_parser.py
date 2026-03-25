import math
import sympy as sp


_LOCALS = {
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


def parse_real(texto: str, nombre: str = "valor") -> float:
    texto = texto.strip()
    if not texto:
        raise ValueError(f"{nombre} no puede estar vacío")

    expr = sp.sympify(texto, locals=_LOCALS)
    if expr.free_symbols:
        raise ValueError(f"{nombre} no debe contener variables")

    valor = float(sp.N(expr))
    if not math.isfinite(valor):
        raise ValueError(f"{nombre} no es finito")
    return valor


def parse_real_or_default(texto: str, default: float, nombre: str = "valor") -> float:
    texto = texto.strip()
    if not texto:
        return default
    return parse_real(texto, nombre)


def parse_int_or_default(texto: str, default: int, nombre: str = "valor") -> int:
    texto = texto.strip()
    if not texto:
        return default

    valor = parse_real(texto, nombre)
    entero = round(valor)
    if abs(valor - entero) > 1e-12:
        raise ValueError(f"{nombre} debe ser un entero")
    return int(entero)
