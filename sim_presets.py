"""Biblioteca de presets para el simulador de sistemas dinámicos."""

PRESETS_1D = {
    "Bifurcación Silla-Nodo": {
        "equation": "r + x**2",
        "params": {"r": {"min": -5.0, "max": 5.0, "default": -1.0, "step": 0.1}},
        "x0": 0.5, "t_span": (0, 20), "x_range": (-5, 5),
        "bif_param": "r", "bif_range": (-5, 5),
        "theory": r"""
### Bifurcación Silla-Nodo (Saddle-Node)
$$\dot{x} = r + x^2$$
**Comportamiento cualitativo:**
- Para $r < 0$: Existen **dos** puntos de equilibrio: $x^* = \pm\sqrt{-r}$. El negativo es estable (atractor) y el positivo es inestable (repulsor).
- Para $r = 0$: Los dos puntos se **fusionan** en $x^* = 0$ (punto semi-estable).
- Para $r > 0$: **No existen** puntos de equilibrio. Todas las trayectorias divergen.

**Bifurcación:** En $r_c = 0$ se produce una *bifurcación silla-nodo*: dos puntos fijos (uno estable y uno inestable) colisionan y se aniquilan mutuamente. Es el mecanismo genérico por el cual se crean o destruyen puntos fijos.
"""
    },
    "Bifurcación Transcrítica": {
        "equation": "r*x - x**2",
        "params": {"r": {"min": -5.0, "max": 5.0, "default": 1.0, "step": 0.1}},
        "x0": 0.1, "t_span": (0, 20), "x_range": (-5, 5),
        "bif_param": "r", "bif_range": (-5, 5),
        "theory": r"""
### Bifurcación Transcrítica
$$\dot{x} = rx - x^2$$
**Comportamiento cualitativo:**
- Siempre existen dos puntos fijos: $x^* = 0$ y $x^* = r$.
- Para $r < 0$: $x^*=0$ es estable, $x^*=r$ es inestable.
- Para $r > 0$: **Intercambian estabilidad**. $x^*=0$ se vuelve inestable y $x^*=r$ se vuelve estable.

**Bifurcación:** En $r_c = 0$ los puntos fijos se cruzan e intercambian su estabilidad. A diferencia de la silla-nodo, los puntos fijos no se destruyen sino que *intercambian roles*.
"""
    },
    "Bifurcación Tridente Supercrítica": {
        "equation": "r*x - x**3",
        "params": {"r": {"min": -5.0, "max": 5.0, "default": 1.0, "step": 0.1}},
        "x0": 0.1, "t_span": (0, 20), "x_range": (-5, 5),
        "bif_param": "r", "bif_range": (-3, 3),
        "theory": r"""
### Bifurcación Tridente Supercrítica (Pitchfork)
$$\dot{x} = rx - x^3$$
**Comportamiento cualitativo:**
- Para $r \leq 0$: Solo $x^*=0$ (estable).
- Para $r > 0$: $x^*=0$ se vuelve inestable y nacen dos ramas estables: $x^* = \pm\sqrt{r}$.

**Bifurcación:** Supercrítica porque las nuevas ramas estables aparecen de forma *suave y continua*. Es la transición típica de ruptura espontánea de simetría (ej: pandeo de una columna, transiciones de fase de 2do orden).
"""
    },
    "Bifurcación Tridente Subcrítica": {
        "equation": "r*x + x**3",
        "params": {"r": {"min": -5.0, "max": 5.0, "default": -1.0, "step": 0.1}},
        "x0": 0.1, "t_span": (0, 10), "x_range": (-3, 3),
        "bif_param": "r", "bif_range": (-3, 3),
        "theory": r"""
### Bifurcación Tridente Subcrítica
$$\dot{x} = rx + x^3$$
**Comportamiento cualitativo:**
- Para $r < 0$: $x^*=0$ es estable, flanqueado por dos ramas inestables $x^*=\pm\sqrt{-r}$.
- Para $r \geq 0$: $x^*=0$ se desestabiliza. **No hay ramas estables cercanas**, por lo que el sistema salta discontinuamente.

**Bifurcación:** Subcrítica = peligrosa. La transición es *abrupta e irreversible*. Se asocia a fenómenos de histéresis y catástrofes.
"""
    },
    "Histéresis (Cusp)": {
        "equation": "r + h*x - x**3",
        "params": {
            "r": {"min": -5.0, "max": 5.0, "default": 0.0, "step": 0.1},
            "h": {"min": -3.0, "max": 3.0, "default": 0.0, "step": 0.1}
        },
        "x0": 0.1, "t_span": (0, 20), "x_range": (-4, 4),
        "bif_param": "r", "bif_range": (-5, 5),
        "theory": r"""
### Histéresis y Catástrofe Cúspide
$$\dot{x} = r + hx - x^3$$
**Comportamiento cualitativo:**
- Con $h=0$: bifurcación tridente estándar.
- Con $h \neq 0$: la simetría se rompe y aparece un **lazo de histéresis**.
- Al variar $r$ lentamente, el sistema puede saltar abruptamente entre estados y no retorna al estado original por el mismo camino.

**Aplicaciones:** Interruptores ópticos, transiciones de fase de primer orden, fenómenos de snap-through en estructuras mecánicas.
"""
    },
}

PRESETS_2D = {
    "Lotka-Volterra (Presa-Depredador)": {
        "fx": "a*x - b*x*y",
        "fy": "-c*y + d*x*y",
        "params": {
            "a": {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1},
            "b": {"min": 0.1, "max": 3.0, "default": 0.5, "step": 0.1},
            "c": {"min": 0.1, "max": 3.0, "default": 0.75, "step": 0.1},
            "d": {"min": 0.1, "max": 3.0, "default": 0.25, "step": 0.1},
        },
        "x0": 4.0, "y0": 2.0, "t_span": (0, 40),
        "x_range": (0, 10), "y_range": (0, 10),
        "labels": ("Presas x", "Depredadores y"),
        "theory": r"""
### Modelo Presa-Depredador de Lotka-Volterra
$$\dot{x} = ax - bxy \quad \text{(presas)}$$
$$\dot{y} = -cy + dxy \quad \text{(depredadores)}$$
**Parámetros:**
- $a$: Tasa de crecimiento de presas | $b$: Tasa de depredación
- $c$: Mortalidad de depredadores | $d$: Eficiencia de conversión

**Comportamiento:** El sistema posee un punto de equilibrio de coexistencia en $(x^*, y^*) = (c/d, a/b)$ que es un **centro** (órbitas cerradas). Las poblaciones oscilan periódicamente: cuando hay muchas presas, los depredadores crecen; al crecer los depredadores, las presas disminuyen, y así cíclicamente.

**Integral de movimiento:** $H(x,y) = dx - c\ln x + by - a\ln y = \text{const}$
"""
    },
    "Oscilador Van der Pol": {
        "fx": "y",
        "fy": "mu*(1 - x**2)*y - x",
        "params": {
            "mu": {"min": 0.0, "max": 5.0, "default": 1.0, "step": 0.1},
        },
        "x0": 2.0, "y0": 0.0, "t_span": (0, 40),
        "x_range": (-5, 5), "y_range": (-8, 8),
        "labels": ("x", "dx/dt"),
        "theory": r"""
### Oscilador de Van der Pol
$$\ddot{x} - \mu(1-x^2)\dot{x} + x = 0$$
Escrito como sistema: $\dot{x} = y, \quad \dot{y} = \mu(1-x^2)y - x$

**Comportamiento:**
- $\mu = 0$: Oscilador armónico (centro).
- $\mu > 0$ pequeño: **Bifurcación de Hopf supercrítica** en el origen. Nace un **ciclo límite estable**.
- $\mu \gg 1$: Oscilaciones de relajación (forma casi rectangular). El sistema alterna entre fases lentas y rápidas.

**Aplicaciones:** Circuitos eléctricos con retroalimentación no lineal, latidos cardíacos, oscilaciones neuronales.
"""
    },
    "Circuito RLC No Lineal": {
        "fx": "y",
        "fy": "-(1/(L*C))*x - (R/L)*y",
        "params": {
            "R": {"min": 0.0, "max": 5.0, "default": 0.5, "step": 0.1},
            "L": {"min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
            "C": {"min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1},
        },
        "x0": 3.0, "y0": 0.0, "t_span": (0, 30),
        "x_range": (-5, 5), "y_range": (-5, 5),
        "labels": ("Carga q", "Corriente i"),
        "theory": r"""
### Circuito RLC Serie
$$L\ddot{q} + R\dot{q} + \frac{q}{C} = 0$$
Sistema: $\dot{x} = y, \quad \dot{y} = -\frac{1}{LC}x - \frac{R}{L}y$

**Clasificación del punto de equilibrio (origen):**
- $R = 0$: **Centro** (oscilaciones perpetuas sin amortiguamiento).
- $R$ pequeño ($R^2 < 4L/C$): **Foco estable** (oscilaciones amortiguadas).
- $R^2 = 4L/C$: **Nodo estrella** (amortiguamiento crítico).
- $R^2 > 4L/C$: **Nodo estable** (sobreamortiguado).

**Frecuencia natural:** $\omega_0 = 1/\sqrt{LC}$, **Factor de amortiguamiento:** $\zeta = R/(2\sqrt{L/C})$
"""
    },
    "Sistema Lineal General": {
        "fx": "a11*x + a12*y",
        "fy": "a21*x + a22*y",
        "params": {
            "a11": {"min": -5.0, "max": 5.0, "default": -1.0, "step": 0.1},
            "a12": {"min": -5.0, "max": 5.0, "default": 1.0, "step": 0.1},
            "a21": {"min": -5.0, "max": 5.0, "default": -1.0, "step": 0.1},
            "a22": {"min": -5.0, "max": 5.0, "default": -1.0, "step": 0.1},
        },
        "x0": 2.0, "y0": 1.0, "t_span": (0, 20),
        "x_range": (-5, 5), "y_range": (-5, 5),
        "labels": ("x", "y"),
        "theory": r"""
### Sistema Lineal 2D General
$$\dot{\mathbf{x}} = A\mathbf{x}, \quad A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$$

**Clasificación por traza y determinante:**
- $\Delta = \det(A) > 0, \tau = \text{tr}(A) < 0$: **Nodo/Foco estable**
- $\Delta > 0, \tau > 0$: **Nodo/Foco inestable**
- $\Delta < 0$: **Punto silla**
- $\Delta > 0, \tau = 0$: **Centro**
- $\tau^2 - 4\Delta$: discrimina nodos ($>0$) de focos ($<0$)
"""
    },
}

PRESETS_3D = {
    "Atractor de Lorenz": {
        "fx": "sigma*(y - x)",
        "fy": "x*(rho - z) - y",
        "fz": "x*y - beta*z",
        "params": {
            "sigma": {"min": 0.0, "max": 30.0, "default": 10.0, "step": 0.5},
            "rho": {"min": 0.0, "max": 50.0, "default": 28.0, "step": 0.5},
            "beta": {"min": 0.0, "max": 10.0, "default": 2.667, "step": 0.1},
        },
        "x0": 1.0, "y0": 1.0, "z0": 1.0, "t_span": (0, 50),
        "theory": r"""
### Atractor de Lorenz
$$\dot{x} = \sigma(y-x), \quad \dot{y} = x(\rho-z)-y, \quad \dot{z} = xy - \beta z$$
Derivado por Edward Lorenz (1963) como modelo simplificado de convección atmosférica.

**Parámetros clásicos:** $\sigma=10, \rho=28, \beta=8/3$

**Transiciones según $\rho$:**
- $0 < \rho < 1$: El origen es el único atractor (estable).
- $1 < \rho < 24.74$: Aparecen dos puntos fijos estables $C^{\pm}$ (bifurcación tridente). El origen se vuelve inestable.
- $\rho \approx 24.74$: **Transición al caos**. Los puntos $C^{\pm}$ pierden estabilidad vía bifurcación de Hopf subcrítica. Aparece el atractor extraño (efecto mariposa).

**Efecto mariposa:** Dos condiciones iniciales arbitrariamente cercanas divergen exponencialmente (sensibilidad a condiciones iniciales). **Exponente de Lyapunov** máximo $\lambda_1 \approx 0.91 > 0$.
"""
    },
    "Problema de 3 Cuerpos (Simplificado)": {
        "fx": "-x/((x**2+y**2+0.1)**1.5) - (x-1)/((( x-1)**2+y**2+0.1)**1.5)",
        "fy": "-y/((x**2+y**2+0.1)**1.5) - y/(((x-1)**2+y**2+0.1)**1.5)",
        "fz": "0*z",
        "params": {},
        "x0": 0.5, "y0": 0.5, "z0": 0.0, "t_span": (0, 20),
        "theory": r"""
### Problema Restringido de 3 Cuerpos (Versión Simplificada)
Se modela el movimiento de una partícula de prueba bajo la influencia gravitatoria de dos cuerpos masivos fijos en $(0,0)$ y $(1,0)$.

$$\ddot{x} = -\frac{x}{(x^2+y^2)^{3/2}} - \frac{x-1}{((x-1)^2+y^2)^{3/2}}$$

**Nota:** Esta es una versión simplificada para visualización. El problema real de 3 cuerpos es **no integrable** (Poincaré, 1890) y exhibe caos determinista. Las trayectorias son extremadamente sensibles a las condiciones iniciales.

**Puntos de Lagrange:** Existen 5 puntos de equilibrio ($L_1$ a $L_5$), de los cuales $L_4$ y $L_5$ (triángulos equiláteros) son estables bajo ciertas condiciones de masa.
"""
    },
}
