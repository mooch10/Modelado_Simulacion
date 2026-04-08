"""Motor de simulación Monte Carlo para penales en fútbol."""

import numpy as np
from dataclasses import dataclass

@dataclass
class JugadorPenal:
    nombre: str
    peso: float
    altura: float
    precision: float
    velocidad_pies: float
    experiencia: float

@dataclass
class PorteroPenal:
    nombre: str
    peso: float
    altura: float
    reflejos: float
    experiencia: float

class SimuladorPenales:
    def __init__(self, semilla=None):
        if semilla is not None:
            np.random.seed(semilla)
        self.pesos = {
            'precision': 0.40,
            'velocidad': 0.25,
            'experiencia': 0.15,
            'portero_reflejos': -0.15,
            'portero_experiencia': -0.05
        }
    
    def calcular_probabilidad_gol(self, jugador: JugadorPenal, portero: PorteroPenal) -> float:
        """Calcula probabilidad de gol basada en atributos."""
        prob = (
            self.pesos['precision'] * (jugador.precision / 100) +
            self.pesos['velocidad'] * (jugador.velocidad_pies / 130) +
            self.pesos['experiencia'] * min(jugador.experiencia / 20, 1.0) +
            self.pesos['portero_reflejos'] * (portero.reflejos / 100) +
            self.pesos['portero_experiencia'] * min(portero.experiencia / 25, 1.0)
        )
        return max(0.0, min(prob, 0.95))
    
    def simular_penal(self, jugador: JugadorPenal, portero: PorteroPenal) -> bool:
        """Simula un penal. Retorna True si gol."""
        prob = self.calcular_probabilidad_gol(jugador, portero)
        return np.random.random() < prob
    
    def simular_tanda(self, jugadores: list, portero: PorteroPenal, n_simulaciones: int = 10000) -> dict:
        """Simula N tandas completas."""
        goles_por_simulacion = []
        probs_por_jugador = {}
        
        for jugador in jugadores:
            probs_por_jugador[jugador.nombre] = self.calcular_probabilidad_gol(jugador, portero)
        
        for _ in range(n_simulaciones):
            goles = sum(self.simular_penal(j, portero) for j in jugadores)
            goles_por_simulacion.append(goles)
        
        goles_array = np.array(goles_por_simulacion)
        media = np.mean(goles_array)
        std = np.std(goles_array)
        se = std / np.sqrt(n_simulaciones)
        ic_lower = media - 1.96 * se
        ic_upper = media + 1.96 * se
        
        prob_ganar = np.mean(goles_array > 3) * 100
        prob_empatar = np.mean(goles_array == 3) * 100
        prob_perder = np.mean(goles_array < 3) * 100
        
        return {
            'goles_totales_esperado': float(media),
            'goles_std': float(std),
            'error_estandar': float(se),
            'ic_lower': float(ic_lower),
            'ic_upper': float(ic_upper),
            'probabilidad_ganar': float(prob_ganar),
            'probabilidad_empatar': float(prob_empatar),
            'probabilidad_perder': float(prob_perder),
            'prob_por_jugador': probs_por_jugador,
            'distribucion_goles': goles_array.tolist()
        }
