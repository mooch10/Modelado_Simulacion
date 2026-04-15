CASOS_PRACTICOS = {
    "Newton-Raphson": {
        "nombre": "🏀 Altura máxima de balón",
        "descripcion": "Un jugador lanza una pelota. La altura h(t) = 20*t - 5*t² metros a los t segundos. Hallar el tiempo exacto donde la altura es máxima (donde h'(t) = 0).",
        "formula_para_copiar": "20 - 10*x",
        "funcion": "20 - 10*x",
        "x0": 1.5,
        "tol": 1e-8,
        "max_iter": 50,
        "aplicacion": "En deportes y física, encontrar el pico de trayectoria es común. Newton-Raphson resuelve instantáneamente el tiempo exacto del punto más alto sin necesidad de prueba y error."
    },
    "Aitken": {
        "nombre": "🎮 Nivel de jugador videojuego",
        "descripcion": "Videojuego donde el nivel aumenta por iteración: nivel = 0.7*nivel_anterior + 30. ¿Cuál será mi nivel final en equilibrio?",
        "formula_para_copiar": "0.7*x + 30",
        "g": "0.7*x + 30",
        "x0": 0.0,
        "tol": 1e-10,
        "max_iter": 50,
        "aplicacion": "En videojuegos, apps, y sistemas que convergen lentamente, Aitken predice el resultado final sin esperar. Juega 3 veces rápido y Aitken te dice cuándo llegarás a máximo nivel."
    },
    "Biseccion": {
        "nombre": "🎯 Nivel de dificultad en videojuego",
        "descripcion": "Videojuego tiene slider de dificultad 1-100. Bisección encuentra el nivel exacto donde tienes 50% de ganar.",
        "formula_para_copiar": "sin(x/50 - 1) - 0.3",
        "funcion": "sin(x/50 - 1) - 0.3",
        "a": 1.0,
        "b": 100.0,
        "tol": 1e-4,
        "max_iter": 50,
        "aplicacion": "En diseño de juegos, apps de fitness, y controles inteligentes, bisección encuentra el punto perfecto. Divide el rango a la mitad cada vez hasta encontrar el equilibrio exacto."
    },
    "Punto Fijo": {
        "nombre": "💬 Trending topic en redes",
        "descripcion": "Hashtag trending: tweets_{n+1} = 0.9*tweets_n + 5000. ¿Cuántos tweets en equilibrio?",
        "formula_para_copiar": "0.9*x + 5000",
        "g": "0.9*x + 5000",
        "x0": 10000.0,
        "tol": 1e-3,
        "max_iter": 100,
        "aplicacion": "En redes sociales y fenómenos virales, punto fijo predice cuándo un hashtag se estabiliza. Sin fórmulas complejas, solo iterar hasta que el número de tweets no cambie más."
    },
    "Lagrange + Derivacion": {
        "nombre": "🚗 Velocidad en una carretera",
        "descripcion": "GPS marca posición cada 10 segundos: (0, 0 km), (10, 2.5 km), (20, 5 km). Lagrange interpola: ¿dónde estabas a los 7 segundos? Derivada = velocidad en ese momento.",
        "formula_para_copiar": "datos_x=[0,10,20]; datos_y=[0,2.5,5]; punto=7.0",
        "datos_x": [0, 10, 20],
        "datos_y": [0, 2.5, 5],
        "punto": 7.0,
        "aplicacion": "En navegación GPS, apps de fitness y deportes wearables, tienes mediciones cada pocos segundos pero necesitas posición/velocidad precisa entre puntos. Lagrange lo hace suavemente."
    },
    "Integracion Numerica": {
        "nombre": "🏊 Volumen de una piscina irregular",
        "descripcion": "Piscina con profundidad variable. Integrar la profundidad para obtener volumen total.",
        "formula_para_copiar": "2 + sin(x)",
        "funcion": "2 + sin(x)",
        "a": 0.0,
        "b": 10.0,
        "n": 20,
        "aplicacion": "En construcción, parques acuáticos, y diseño, necesitas volumen total sin levantar cada sección. Integración numérica suma automáticamente usando pocos datos."
    },
    "Ajuste de Curvas": {
        "nombre": "🏋️ Progreso en el gimnasio",
        "descripcion": "Peso levantado cada semana: (1, 20kg), (2, 22kg), (3, 25kg), (4, 28kg), (5, 32kg). Ajuste polinomial predice: ¿cuánto levantarás en semana 8?",
        "formula_para_copiar": "X=[1,2,3,4,5]; Y=[20,22,25,28,32]",
        "datos_x": [1, 2, 3, 4, 5],
        "datos_y": [20, 22, 25, 28, 32],
        "tipo": "polinomial",
        "grado": 2,
        "aplicacion": "En fitness, salud y deportes, ajuste de curvas predice progreso. Ingresa datos de semanas pasadas, obtén predicción de futuro sin cálculos complejos."
    },
    "Monte Carlo": {
        "nombre": "🎲 Probabilidad de ganar la Loto",
        "descripcion": "Simular tiradas de Loto. Ganas si el número aleatorio > 0.99 (1% de probabilidad).",
        "formula_para_copiar": "1 if x > 0.99 else 0",
        "funcion": "1 if x > 0.99 else 0",
        "a": 0.0,
        "b": 1.0,
        "n": 100000,
        "aplicacion": "En juegos de azar, seguros, y riesgos, Monte Carlo simula millones de escenarios para responder cuál es la probabilidad real. Sin fórmulas complicadas, solo simulación."
    },
    "Sistemas Lineales": {
        "nombre": "🍜 Receta con 3 ingredientes",
        "descripcion": "Harina cuesta $2/kg, azúcar $3/kg, mantequilla $5/kg. Necesitas 10 kg total, costo $30, y mantequilla sea el doble de harina. ¿Cuánto de cada uno?",
        "formula_para_copiar": "A=[[1,1,1],[2,3,5],[1,0,-2]]; b=[10,30,0]",
        "A": [[1, 1, 1], [2, 3, 5], [1, 0, -2]],
        "b": [10, 30, 0],
        "metodo": "Gauss-Jordan",
        "aplicacion": "En cocina, finanzas personales y mezclas, sistemas lineales resuelven recetas y presupuestos. Especifica restricciones, obtén cantidades exactas de cada ingrediente."
    },
    "EDO": {
        "nombre": "💊 Medicamento en sangre",
        "descripcion": "Pastilla de 500mg. El cuerpo degrada 30% cada hora: dM/dt = -0.3*M.",
        "formula_para_copiar": "-0.3*y",
        "f": "-0.3*y",
        "x0": 0.0,
        "y0": 500.0,
        "h": 0.1,
        "n": 50,
        "aplicacion": "En farmacología y medicina, ecuaciones diferenciales modelan cómo el cuerpo metaboliza drogas. RK4 predice concentración en sangre sin experimentos, solo matemáticas."
    },
    "Red Neuronal GD": {
        "nombre": "📚 Calificación vs horas estudiadas",
        "descripcion": "Datos de varios estudiantes: (2 hrs, 60%), (4 hrs, 75%), (6 hrs, 85%), (8 hrs, 95%). Red neuronal aprende: calificación = w*horas + b. ¿Qué nota obtendrás con 5 horas?",
        "formula_para_copiar": "generar_datos=lineal; alpha=0.01; epocas=200",
        "generar_datos": "lineal",
        "alpha": 0.01,
        "epocas": 200,
        "semilla": 42,
        "aplicacion": "En educación, descenso de gradiente aprende la relación entre esfuerzo y resultados. Sin fórmula teórica, solo datos históricos suficientes para predecir notas futuras."
    },
    "Monte Carlo 2D": {
        "nombre": "🌳 Bosque en terreno",
        "descripcion": "Región 10x10 km con bosque irregular. Lanzar 50000 puntos aleatorios, contar cuántos caen en bosque. Monte Carlo calcula: ¿cuántos km² de bosque real?",
        "formula_para_copiar": "1 if ((x-5)**2 + (y-5)**2) < 16 else 0",
        "funcion": "1 if ((x-5)**2 + (y-5)**2) < 16 else 0",
        "a": 0.0,
        "b": 10.0,
        "c": 0.0,
        "d": 10.0,
        "n": 50000,
        "aplicacion": "En ecología, cartografía y urbanismo, Monte Carlo 2D mide áreas de bosques, ciudades, océanos sin levantar cada metro. Solo puntos aleatorios y conteo."
    }
}
