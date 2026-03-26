# Guía Completa de Estudio: Modelado y Simulación (Clase 3)

## 1. El Problema de los Datos Discretos
En ingeniería y ciencia, los datos suelen presentarse como un conjunto de puntos aislados (experimentos, sensores, mediciones) y no como funciones continuas. 
- **Desafío**: ¿Cómo predecir el comportamiento entre los puntos?
- **Solución**: Construir un modelo funcional continuo a partir de datos discretos.

---

## 2. Interpolación de Lagrange
Es una técnica para construir un **polinomio único** que pasa exactamente por todos los puntos de datos. A diferencia de otros métodos, no es una aproximación general, sino un ajuste perfecto a los valores conocidos.

### La Estructura del Polinomio $P(x)$
El polinomio final es una **suma ponderada** donde cada punto $y_i$ aporta su "influencia" mediante un polinomio base $L_i(x)$:
$$P(x) = \sum_{i=0}^{n} y_i L_i(x)$$

### Polinomios Base $L_i(x)$
Están diseñados con una propiedad específica:
- Valen **1** en su propio punto $x_i$.
- Valen **0** en todos los demás puntos $x_j$ (don de $j \neq i$).



---

## 3. Límites del Modelo y Precisión
- **Interpolación**: Estimar valores dentro del rango de los datos conocidos (es confiable).
- **Extrapolación**: Estimar valores fuera del rango (es riesgoso y puede llevar a errores graves).
- **Error de Truncamiento**: En la aproximación de derivadas, el error proviene de ignorar los términos de orden superior en la serie de Taylor.

---

## 4. Diferencias Finitas y Divididas
Son reordenamientos de la serie de Taylor truncada que permiten estimar derivadas a partir de los puntos.

### Concepto de Diferencias Centrales
Para obtener una mejor aproximación de la derivada en un punto $x_i$, se utiliza la información de ambos lados ($x_{i-1}$ y $x_{i+1}$) y se promedia el cambio. Esto es más preciso que usar solo un lado.



### Fórmulas Clave (Paso constante $h$)
Si la distancia entre puntos $x$ es constante ($h$), usamos:

1. **Primera Derivada (Pendiente)**:
   $$f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}$$

2. **Segunda Derivada (Concavidad/Aceleración)**:
   $$f''(x_i) \approx \frac{f(x_{i+1}) - 2f(x_i) + f(x_{i-1})}{h^2}$$

---

## 5. Implementación en Código (Python)
Para verificar la efectividad de las diferencias centrales, se puede aplicar a una función conocida:

```python
# Ejemplo: f(x) = x^3 - 2x + 1
# Punto de interés: x = 2, Paso: h = 0.1

def f(x):
    return x**3 - 2*x + 1

def primera_derivada(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def segunda_derivada(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)