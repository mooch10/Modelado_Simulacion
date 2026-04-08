# 🎲 Simulador Monte Carlo de Penales en Fútbol

## 🚀 Comienza Aquí

### En 30 segundos:
```bash
pip install -r requirements.txt
streamlit run app_penales.py
```

### En 5 minutos:
1. Lee: **[COMIENZA_AQUI.txt](COMIENZA_AQUI.txt)**
2. Ejecuta: `python validar.py`
3. Corre: `streamlit run app_penales.py`

---

## 📚 Documentación por Nivel

### 🟢 Principiante (Tu primer contacto)
- **[COMIENZA_AQUI.txt](COMIENZA_AQUI.txt)** - Instalación y primeros pasos
- **[RESUMEN_FINAL.md](RESUMEN_FINAL.md)** - Qué se logró y por qué

### 🟡 Intermedio (Entender cómo funciona)
- **[README.md](README.md)** - Manual completo y ejemplos
- **[MAPA_ARCHIVOS.txt](MAPA_ARCHIVOS.txt)** - Guía de navegación

### 🔴 Avanzado (Aprender la ciencia)
- **[ANALISIS_DETALLADO.md](ANALISIS_DETALLADO.md)** - Matemática y teoría de Monte Carlo
- **[ESTRUCTURA_PROYECTO.md](ESTRUCTURA_PROYECTO.md)** - Arquitectura del código

---

## ⚡ Acciones Rápidas

| Quiero... | Comando |
|-----------|---------|
| Ejecutar la interfaz web | `streamlit run app_penales.py` |
| Ver ejemplos de código | `python ejemplos.py` |
| Validar que todo funciona | `python validar.py` |
| Verificar lista de jugadores | Abre `utilidades_penales.py` o corre ejemplos |

---

## 📂 Archivos Principales

```
📚 DOCUMENTACIÓN (empieza por aquí)
├─ COMIENZA_AQUI.txt          ⭐ Para principiantes
├─ README.md                  📖 Manual completo
├─ ANALISIS_DETALLADO.md      🔬 Para científicos
├─ ESTRUCTURA_PROYECTO.md     🗺️ Para desarrolladores
├─ RESUMEN_FINAL.md           ✨ Visión general
└─ MAPA_ARCHIVOS.txt          📋 Mapa de archivos

🎯 CÓDIGO EJECUTABLE
├─ app_penales.py             💻 Interfaz web
├─ penales_montecarlo.py      ⚙️ Motor de simulación
├─ utilidades_penales.py      📊 Base de datos
├─ ejemplos.py                🎮 Ejemplos runables
└─ validar.py                 ✅ Validación

⚙️ CONFIGURACIÓN
└─ requirements.txt           📦 Dependencias
```

---

## 🎮 Características

✅ **Simulación Monte Carlo** - 10,000+ iteraciones  
✅ **54 Jugadores Profesionales** - Reales y auténticos  
✅ **20 Porteros Profesionales** - Incluye los mejores  
✅ **Interfaz Futbolística** - Diseño moderno y atractivo  
✅ **Análisis Profundo** - Explicaciones estadísticas  
✅ **Agrupación por País** - Equipos realistas  
✅ **Documentación Completa** - 1,500+ líneas  
✅ **Ejemplos Ejecutables** - Aprende con código real  

---

## ✨ Ejemplo Rápido en Python

```python
from utilidades_penales import GeneradorDatos
from penales_montecarlo import SimuladorPenales

# Obtener equipo aleatorio (agrupado por país)
jugadores = GeneradorDatos.obtener_jugadores_aleatorios(5, por_equipo=True)
portero = GeneradorDatos.obtener_portero_aleatorio()

# Simular tanda de penales
sim = SimuladorPenales(semilla=2024)
resultado = sim.simular_tanda(jugadores, portero, n_simulaciones=10000)

# Resultados
print(f"Goles esperados: {resultado['goles_totales_esperado']:.2f}")
print(f"Probabilidad ganar: {resultado['probabilidad_ganar']:.1%}")
print(f"IC 95%: [{resultado['ic_lower']:.2f}, {resultado['ic_upper']:.2f}]")
```

---

## 🎓 Rutas de Aprendizaje

### Ruta Básica (2 horas)
1. Leer COMIENZA_AQUI.txt (20 min)
2. Ejecutar streamlit app (10 min)
3. Experimentar en interfaz (30 min)
4. Leer README.md (40 min)
5. Ejecutar ejemplos.py (20 min)

### Ruta Intermedia (4 horas)
1. Ruta básica (2 horas)
2. Leer ANALISIS_DETALLADO.md (60 min)
3. Modificar ejemplos.py (40 min)

### Ruta Profunda (8 horas)
1. Ruta intermedia (4 horas)
2. Leer código fuente (120 min)
3. Modificar penales_montecarlo.py (120 min)

---

## 🤔 Preguntas Frecuentes

**P: ¿Por qué se llama Monte Carlo?**  
R: Porque usa simulación aleatoria (muestreo) para calcular probabilidades. Como los casinos. Ver ANALISIS_DETALLADO.md

**P: ¿Puedo agregar mis propios jugadores?**  
R: Sí. Edita `utilidades_penales.py` o pasa `JugadorPenal` custom a `simular_tanda()`. Ver README.md

**P: ¿Qué tan preciso es?**  
R: Mejora con más simulaciones. 10,000 es buen balance. Ver ANALISIS_DETALLADO.md "Precisión"

**P: ¿Funciona en móvil?**  
R: La interfaz web (Streamlit) es responsive. Sí funciona.

**P: ¿Puedo modificar los pesos de la fórmula?**  
R: Sí, están en `SimuladorPenales.calcular_probabilidad_gol()` en penales_montecarlo.py

---

## 💡 Consejos

1. **Para principiantes:** Lee COMIENZA_AQUI.txt primero. No saltes a ANALISIS_DETALLADO.md

2. **Para aprender código:** Ejecuta `python ejemplos.py` y lee el código ejemplo a ejemplo

3. **Para validar:** Siempre corre `python validar.py` después de cambios

4. **Para experimentar:** Usa la interfaz web. Es más intuitiva que Python.

5. **Para profundidad:** Mapea las fórmulas. Son la piedra angular de todo (ANALISIS_DETALLADO.md)

---

## 🔧 Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Corre: `pip install -r requirements.txt` |
| La interfaz web no inicia | ¿Python 3.10+? Verifica con `python --version` |
| Errores en validar.py | Asegúrate de estar en carpeta PenalesSimulador/ |
| Resultados raros | Aumenta n_simulaciones. 100,000 es muy preciso. |

Para más troubleshooting, ver COMIENZA_AQUI.txt

---

## 📊 Estadísticas del Proyecto

```
📈 Codebase
├─ Líneas Python: 1,450+
├─ Líneas Documentación: 1,500+
├─ Total: 2,950+ líneas

🎮 Interfaz
├─ Tabs: 4
├─ Gráficos: 3
├─ Componentes custom: 10+

📚 Base Datos
├─ Jugadores: 54
├─ Porteros: 20
├─ Países: 15+

⚡ Performance
├─ Simulación (1k): 0.5 seg
├─ Simulación (10k): 5 seg
├─ Simulación (100k): 50 seg
```

---

## 📞 Soporte y Referencias

- **Documentación:** Comienza con COMIENZA_AQUI.txt
- **Código Ejemplo:** Ver ejemplos.py
- **Validación:** Ejecuta validar.py
- **Preguntas:** Revisa README.md § "FAQ"
- **Profundidad:** Lee ANALISIS_DETALLADO.md

---

## 🎊 ¡Disfruta Simulando!

```bash
# Esto es todo lo que necesitas:
pip install -r requirements.txt
streamlit run app_penales.py
```

Se abrirá en: **http://localhost:8501**

---

**Versión:** 2.0 | **Estado:** ✅ Listo para usar | **2024**

👉 **Próximo paso:** Lee [COMIENZA_AQUI.txt](COMIENZA_AQUI.txt) o ejecuta `streamlit run app_penales.py`
