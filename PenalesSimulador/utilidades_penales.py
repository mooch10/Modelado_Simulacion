"""Base de datos de jugadores y utilidades."""

import random
from penales_montecarlo import JugadorPenal, PorteroPenal

# ARGENTINA - FINAL 2022
JUGADORES_ARGENTINA = {
    'Lionel Messi': JugadorPenal('Lionel Messi', 72, 170, 94, 118, 18),
    'Angel Di Maria': JugadorPenal('Ángel Di María', 78, 180, 88, 125, 17),
    'Gonzalo Montiel': JugadorPenal('Gonzalo Montiel', 80, 182, 85, 120, 12),
    'Enzo Fernandez': JugadorPenal('Enzo Fernández', 75, 178, 84, 115, 3),
    'Julian Alvarez': JugadorPenal('Julián Álvarez', 75, 177, 88, 125, 4),
}

JUGADORES_FRANCIA = {
    'Kylian Mbappe': JugadorPenal('Kylian Mbappé', 73, 178, 91, 130, 7),
    'Olivier Giroud': JugadorPenal('Olivier Giroud', 89, 192, 85, 115, 18),
    'Antoine Griezmann': JugadorPenal('Antoine Griezmann', 74, 183, 88, 120, 13),
    'Aurélien Tchouaméni': JugadorPenal('Aurélien Tchouaméni', 79, 187, 83, 118, 3),
    'Benjamin Pavard': JugadorPenal('Benjamin Pavard', 83, 186, 84, 116, 10),
}

PORTEROS_ESPECIALES = {
    'Emiliano Martinez': PorteroPenal('Emiliano Martínez (ARG)', 87, 195, 88, 11),
    'Hugo Lloris': PorteroPenal('Hugo Lloris (FRA)', 86, 188, 89, 20),
}

class GeneradorDatos:
    @staticmethod
    def obtener_equipo_argentina():
        """Retorna los 5 lanzadores de Argentina."""
        return list(JUGADORES_ARGENTINA.values())
    
    @staticmethod
    def obtener_equipo_francia():
        """Retorna los 5 lanzadores de Francia."""
        return list(JUGADORES_FRANCIA.values())
    
    @staticmethod
    def obtener_portero_argentina():
        """Portero de Argentina."""
        return PORTEROS_ESPECIALES['Emiliano Martinez']
    
    @staticmethod
    def obtener_portero_francia():
        """Portero de Francia."""
        return PORTEROS_ESPECIALES['Hugo Lloris']
    
    @staticmethod
    def obtener_jugadores_aleatorios(cantidad=11, por_equipo=True):
        """Retorna N jugadores aleatorios."""
        todos = list(JUGADORES_ARGENTINA.values()) + list(JUGADORES_FRANCIA.values())
        return random.sample(todos, min(cantidad, len(todos)))
    
    @staticmethod
    def obtener_portero_aleatorio():
        """Retorna portero aleatorio."""
        return random.choice(list(PORTEROS_ESPECIALES.values()))

    @staticmethod
    def obtener_portero_argentina():
        """Portero de Argentina."""
        return PORTEROS_ESPECIALES['Emiliano Martinez']
    
    @staticmethod
    def obtener_portero_francia():
        """Portero de Francia."""
        return PORTEROS_ESPECIALES['Hugo Lloris']

class GeneradorDatos:
    @staticmethod
    def obtener_equipo_argentina():
        """Retorna los 11 lanzadores de Argentina."""
        return list(JUGADORES_ARGENTINA.values())[:11]
    
    @staticmethod
    def obtener_equipo_francia():
        """Retorna los 11 lanzadores de Francia."""
        return list(JUGADORES_FRANCIA.values())[:11]
    
    @staticmethod
    def obtener_portero_argentina():
        """Portero de Argentina."""
        return PORTEROS_ESPECIALES['Emiliano Martinez']
    
    @staticmethod
    def obtener_portero_francia():
        """Portero de Francia."""
        return PORTEROS_ESPECIALES['Hugo Lloris']
    
    @staticmethod
    def obtener_jugadores_aleatorios(cantidad=11, por_equipo=True):
        """Retorna N jugadores aleatorios."""
        todos = list(JUGADORES_ARGENTINA.values()) + list(JUGADORES_FRANCIA.values())
        return random.sample(todos, min(cantidad, len(todos)))
    
    @staticmethod
    def obtener_portero_aleatorio():
        """Retorna portero aleatorio."""
        return random.choice(list(PORTEROS_ESPECIALES.values()))
