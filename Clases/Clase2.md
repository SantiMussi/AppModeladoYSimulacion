# Velocidad de Convergencia en Metodos Numericos

## 1. Convergencia Lineal
El error disminuye de manera proporcional en cada iteracion. Es un proceso constante y predecible, pero puede ser lento.

### Metodo de Biseccion
Algoritmo logico basado en el Teorema de Bolzano. Es un metodo "cerrado" que siempre converge si la funcion es continua.

**Pasos para resolver:**
1. **Definir intervalo:** Elegir a y b tales que f(a) * f(b) < 0.
2. **Calcular punto medio:** c = (a + b) / 2.
3. **Evaluar signo:**
    * Si f(a) * f(c) < 0: La raiz esta en [a, c]. Hacer b = c.
    * Si f(b) * f(c) < 0: La raiz esta en [c, b]. Hacer a = c.
4. **Criterio de parada:** Repetir desde el paso 2 hasta que |f(c)| < Tolerancia o (b - a) / 2 < Tolerancia.

---

### Punto Fijo (x = g(x))
Transforma la ecuacion f(x) = 0 en la forma x = g(x).

**Pasos para resolver:**
1. **Transformar:** Despejar f(x) = 0 para obtener x = g(x).
2. **Verificar convergencia:** Comprobar que |g'(x)| < 1 cerca de la raiz.
3. **Semilla:** Elegir un valor inicial x0.
4. **Iterar:** Calcular x_{k+1} = g(x_k).
5. **Criterio de parada:** Repetir hasta que |x_{k+1} - x_k| < Tolerancia.

---

## 2. Convergencia Cuadratica (o Superior)
La precision se duplica (aproximadamente) en cada iteracion. Ideal para alta precision con pocos pasos.

### Newton-Raphson
Es el metodo mas eficiente si se conoce la derivada y el punto inicial es cercano a la raiz.

**Pasos para resolver:**
1. **Preparar:** Obtener la derivada f'(x).
2. **Semilla:** Elegir x0 (evitar puntos donde f'(x) = 0).
3. **Iterar:** Aplicar la formula:
   $$x_{n+1} = x_k - \frac{f(x_n)}{f'(x_n)}$$
4. **Criterio de parada:** Repetir hasta que $$|x_{n+1} - x_n| < Tolerancia$$

---

### Aceleracion de Aitken (Proceso Delta^2)
Este metodo se usa para "acelerar" una sucesion que ya esta convergiendo linealmente (como la de Punto Fijo).

**Procedimiento de Tabla (Aitken-Steffensen):**
Para resolver en papel o Excel siguiendo el metodo del profesor:

1. **Columna 1 (xn):** Es tu punto de partida (Semilla).
2. **Columna 2 (xn+1):** Aplicas punto fijo una vez: g(xn).
3. **Columna 3 (xn+2):** Aplicas punto fijo al resultado anterior: g(xn+1).
4. **Columna 4 (x_hat):** Aplicas la formula de Aitken usando las tres columnas anteriores:
   $$\hat{x} = x_n - \frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2x_{n+1} + x_n}$$

**Importante:** El valor de la Columna 4 (x_hat) se convierte en el xn de la siguiente fila. Esto transforma un metodo lineal en uno de convergencia cuadratica.

---

## Comparativa Rapida

| Metodo | Tipo | Usa Derivadas? | Robustez |
| :--- | :--- | :--- | :--- |
| **Biseccion** | Lineal | No | **Alta** (No falla) |
| **Punto Fijo** | Lineal | No | **Media** (Depende de g'(x)) |
| **Newton** | Cuadratica | Si | **Baja** (Depende de x0) |
| **Aitken** | Acelerado | No | **Media** (Mejora al Punto Fijo) |

> **Nota de estudio:** Si un metodo de Punto Fijo diverge, Aitken no podra salvarlo. Aitken solo acelera lo que ya tiende a la raiz.