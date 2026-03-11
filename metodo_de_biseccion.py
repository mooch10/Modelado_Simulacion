import math

def evaluar_funcion(func_str, x):
    """
    Evalúa la función definida por el usuario en un valor dado de x.
    Soporta funciones matemáticas básicas como sin, cos, exp, etc.
    """
    try:
        # Define nombres permitidos por seguridad
        nombres_permitidos = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "pow": pow
        }
        # Evalúa la cadena de la función con x y nombres permitidos
        return eval(func_str, {"__builtins__": None}, {**nombres_permitidos, "x": x})
    except Exception as e:
        raise ValueError(f"Error al evaluar la función: {e}")

def metodo_biseccion(func_str, a, b, tol=1e-6, max_iter=100):
    """
    Encuentra la raíz de f(x) = 0 usando el método de Bisección.
    Requiere que f(a) * f(b) < 0.
    """
    fa = evaluar_funcion(func_str, a)
    fb = evaluar_funcion(func_str, b)

    if fa * fb >= 0:
        raise ValueError("La función debe tener signos opuestos en a y b (f(a) * f(b) < 0)")

    print(f"Iniciando método de Bisección con intervalo [{a}, {b}]")
    print(f"f({a}) = {fa}, f({b}) = {fb}")

    for iteracion in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_funcion(func_str, c)

        print(f"Iteración {iteracion + 1}: c = {c}, f(c) = {fc}")

        if abs(fc) < tol:
            print(f"Convergido a la raíz: {c} después de {iteracion + 1} iteraciones")
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    print(f"Máximo de iteraciones alcanzado. Raíz aproximada: {(a + b) / 2}")
    return (a + b) / 2

def main():
    print("Búsqueda de Raíces usando Método de Bisección")
    print("============================================")

    # Obtener entradas del usuario
    func_str = input("Ingresa la función en términos de x (ej. x**2 - 2): ")
    a = float(input("Ingresa el límite inferior a: "))
    b = float(input("Ingresa el límite superior b: "))
    tol_input = input("Ingresa la tolerancia (por defecto 1e-6): ")
    tol = float(tol_input) if tol_input else 1e-6
    max_iter_input = input("Ingresa el máximo de iteraciones (por defecto 100): ")
    max_iter = int(max_iter_input) if max_iter_input else 100

    try:
        raiz = metodo_biseccion(func_str, a, b, tol, max_iter)
        print(f"\nRaíz aproximada encontrada: {raiz}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()