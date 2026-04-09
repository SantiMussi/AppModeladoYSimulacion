import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re
import sympy as sp

# Configuración de la página
st.set_page_config(page_title="Santiago Mussi | Numeric Solver", layout="wide")

st.title("Analizador de Métodos Numéricos")
st.markdown("---")

# --- FUNCIONES DE EVALUACIÓN Y DERIVADA ---

def evaluar_f(f_str, x_val):
    """Evalúa funciones para cálculos numéricos rápidos (gráficos y raíces)"""
    try:
        f_proc = f_str.replace("^", "**")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_val, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "exp": np.exp, "log": np.log, 
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "sp": sp
        }
        return eval(f_proc, {"__builtins__": None}, contexto)
    except:
        return None

def calcular_derivada_robusta(f_str, x_val, h=1e-5):
    f = lambda x: evaluar_f(f_str, x)
    val_plus2 = f(x_val + 2*h)
    val_plus1 = f(x_val + h)
    val_minus1 = f(x_val - h)
    val_minus2 = f(x_val - 2*h)
    if any(v is None for v in [val_plus2, val_plus1, val_minus1, val_minus2]): return 0
    num = -val_plus2 + 8*val_plus1 - 8*val_minus1 + val_minus2
    den = 12 * h
    return num / den

# --- LÓGICA DE LAGRANGE (SIMBÓLICO CON VARIABLES) ---

def calcular_lagrange_avanzado(x_strs, y_strs):
    x_sym = sp.symbols('x')
    n = len(x_strs)
    listado_L = []
    polinomio_total = 0
    
    # Convertimos strings a objetos simbólicos (acepta 'b', 'pi', etc.)
    x_exact = [sp.nsimplify(sp.sympify(x), constants=[sp.pi, sp.E]) for x in x_strs]
    y_exact = [sp.nsimplify(sp.sympify(y), constants=[sp.pi, sp.E]) for y in y_strs]
    
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x_sym - x_exact[j]) / (x_exact[i] - x_exact[j])
        
        li_limpio = sp.simplify(li)
        listado_L.append(li_limpio)
        polinomio_total += y_exact[i] * li
    
    poly_final = sp.simplify(sp.expand(polinomio_total))
    return poly_final, listado_L

# --- LÓGICA DE DIFERENCIAS CENTRALES ---

def metodo_diferencias_centrales(x_pts, y_pts):
    if len(x_pts) < 3: return None
    h = x_pts[1] - x_pts[0]
    derivadas = []
    for i in range(1, len(x_pts) - 1):
        d1 = (y_pts[i+1] - y_pts[i-1]) / (2 * h)
        d2 = (y_pts[i+1] - 2*y_pts[i] + y_pts[i-1]) / (h**2)
        derivadas.append({
            "Punto x": x_pts[i], "f'(x)": d1, "f''(x)": d2, "Error O(h^2)": h**2
        })
    return pd.DataFrame(derivadas)

# --- INTEGRACIÓN: SIMPSON 1/3 ---

def metodo_simpson_13(f_str, a, b, n):
    """Regla de Simpson 1/3 compuesta. n debe ser par."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts = np.array([evaluar_f(f_str, xi) for xi in x_pts])
    if any(v is None for v in y_pts):
        return None, None, None, None

    # Suma compuesta Simpson 1/3
    suma = y_pts[0] + y_pts[-1]
    for i in range(1, n):
        suma += (4 if i % 2 != 0 else 2) * y_pts[i]
    integral = (h / 3) * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^4/180 * max|f''''(xi)|
    h_num = max(h * 0.1, 1e-4)
    f4_vals = []
    for xi in x_pts[2:-2]:
        vals = [evaluar_f(f_str, xi + k * h_num) for k in [-2, -1, 0, 1, 2]]
        if all(v is not None for v in vals):
            f4 = (vals[0] - 4*vals[1] + 6*vals[2] - 4*vals[3] + vals[4]) / h_num**4
            f4_vals.append(abs(f4))
    f4_max = max(f4_vals) if f4_vals else 0.0
    error_trunc = abs((b - a) * h**4 * f4_max / 180)

    # Tabla por segmentos (cada par de subintervalos)
    tabla = []
    for i in range(0, n, 2):
        integral_local = (h / 3) * (y_pts[i] + 4 * y_pts[i+1] + y_pts[i+2])
        tabla.append({
            "Segmento": f"[{x_pts[i]:.4f}, {x_pts[i+2]:.4f}]",
            "f(xᵢ)": round(float(y_pts[i]), 6),
            "f(xᵢ₊₁)": round(float(y_pts[i+1]), 6),
            "f(xᵢ₊₂)": round(float(y_pts[i+2]), 6),
            "Área parcial": round(float(integral_local), 8),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla)

# --- MÉTODOS DE RAÍCES ---

def metodo_biseccion(f_str, a, b, tol, max_iter):
    history = []
    fa, fb = evaluar_f(f_str, a), evaluar_f(f_str, b)
    if fa is None or fb is None or fa * fb >= 0: return None, "error_signos", 0, 100
    x_ant = a
    for i in range(max_iter):
        c = (a + b) / 2
        fc = evaluar_f(f_str, c)
        err = abs(c - x_ant) / abs(c) * 100 if abs(c) > 1e-12 else 0
        history.append({"Iter": i+1, "a": a, "b": b, "x_n": c, "f(x_n)": fc, "Error (%)": err})
        if abs(fc) < 1e-12 or err < tol: return pd.DataFrame(history), "convergencia", c, err
        if fa * fc < 0: b = c
        else: a, fa = c, fc
        x_ant = c
    return pd.DataFrame(history), "limite", x_ant, err

def metodo_newton_raphson(f_str, x0, tol, max_iter):
    history = []
    x_n = x0
    for i in range(max_iter):
        fx = evaluar_f(f_str, x_n)
        dfx = calcular_derivada_robusta(f_str, x_n)
        if dfx == 0: break
        x_next = x_n - fx / dfx
        err = abs(x_next - x_n) / abs(x_next) * 100 if abs(x_next) > 1e-12 else 0
        history.append({"Iter": i+1, "x_n": x_n, "f(x_n)": fx, "f'(x_n)": dfx, "Error (%)": err})
        if err < tol: return pd.DataFrame(history), "convergencia", x_next, err
        x_n = x_next
    return pd.DataFrame(history), "limite", x_n, err

# --- INTERFAZ ---

st.sidebar.header("Configuración")
metodo_sel = st.sidebar.selectbox("Selecciona Método",
    ["Bisección", "Newton-Raphson", "Interpolación Lagrange", "Diferencias Centrales", "Simpson 1/3"])

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entrada")
    if "Interpolación" in metodo_sel or "Diferencias" in metodo_sel:
        st.info("Formato: x, y (un punto por línea)")
        default_pts = "1, 1\n4, 2"
        puntos_input = st.text_area("Puntos:", value=default_pts, height=150)
        
        if metodo_sel == "Interpolación Lagrange":
            func_teorica = st.text_input("Función real f(x):", value="sqrt(x)")
            x_eval = st.text_input("Evaluar en x (Epsilon):", value="2")
            
        try:
            lineas = [l.strip() for l in puntos_input.strip().split('\n') if l.strip()]
            x_in_strs = [l.split(',')[0].strip() for l in lineas]
            y_in_strs = [l.split(',')[1].strip() for l in lineas]
            x_in_num = np.array([float(sp.sympify(x).evalf()) for x in x_in_strs])
            y_in_num = np.array([float(sp.sympify(y).evalf()) for y in y_in_strs])
        except:
            st.error("Error al leer puntos.")
    elif metodo_sel == "Simpson 1/3":
        func_input = st.text_input("f(x):", value="x**3")
        a_simp_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_simp_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_simp = int(st.number_input("Nº subintervalos n (debe ser par)", value=4, min_value=2, step=2))
        if n_simp % 2 != 0:
            st.warning("n debe ser par — se ajustará a n+1 automáticamente.")
        try:
            a_simp = float(sp.sympify(a_simp_str).evalf())
            b_simp = float(sp.sympify(b_simp_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_simp, b_simp = 0.0, 1.0
    else:
        func_input = st.text_input("f(x):", value="x**2 - 2")
        if metodo_sel == "Bisección":
            a_in, b_in = st.number_input("a", value=0.0), st.number_input("b", value=2.0)
        else:
            x0_in = st.number_input("x0", value=1.0)
        tol_in = st.number_input("Tolerancia (%)", value=0.001, format="%.6f")
        iter_in = st.slider("Max Iter", 5, 100, 20)

    ejecutar = st.button("Calcular")

with col2:
    if ejecutar:
        if metodo_sel == "Interpolación Lagrange":
            poly, lista_L = calcular_lagrange_avanzado(x_in_strs, y_in_strs)
            st.subheader("Fórmulas")
            st.latex(
                r"P(x) = \sum_{i=0}^{n} y_i \, L_i(x)"
            )
            st.latex(
                r"L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}"
            )
            with st.expander("📖 Notación"):
                st.markdown("""
| Símbolo | Significado |
|---|---|
| $P(x)$ | Polinomio interpolante de Lagrange |
| $n$ | Grado del polinomio (número de puntos − 1) |
| $x_i,\\, y_i$ | Nodos e imágenes conocidos |
| $L_i(x)$ | $i$-ésimo polinomio base de Lagrange |
| $x$ | Punto en el que se evalúa $P(x)$ |
""")
            st.subheader("Resultado")
            st.latex(f"P(x) = {sp.latex(poly)}")

            # Evaluación y Error
            x_ev_sym = sp.sympify(x_eval)
            val_p = poly.subs(sp.symbols('x'), x_ev_sym)
            st.info(f"Evaluación: P({x_eval}) = {float(val_p.evalf()):.6f}")
            st.latex(f"P({sp.latex(x_ev_sym)}) = {sp.latex(sp.simplify(val_p))}")
            
            if func_teorica:
                x_eval_float = float(x_ev_sym.evalf())
                v_real = evaluar_f(func_teorica, x_eval_float)
                if v_real is not None:
                    err_local = abs(float(val_p.evalf()) - v_real)
                    st.metric("Error Local  |P(x) − f(x)|", f"{err_local:.8f}")

            # Gráfico (solo si no hay variables b, c, etc en y)
            if all(sp.sympify(y).is_number for y in y_in_strs):
                x_range = np.linspace(min(x_in_num)-1, max(x_in_num)+1, 150)
                
                # Evaluación segura iterativa para el gráfico
                y_poly_np = [float(poly.subs(sp.symbols('x'), xi).evalf()) for xi in x_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_range, y=y_poly_np, name="P(x)", line=dict(color='#00cfcc')))
                
                # Agregamos la Función Real al gráfico
                if func_teorica:
                    y_real_np = [evaluar_f(func_teorica, xi) for xi in x_range]
                    if None not in y_real_np:
                        fig.add_trace(go.Scatter(x=x_range, y=y_real_np, name="Función Real", line=dict(color='#ff4b4b', dash='dash')))
                
                fig.add_trace(go.Scatter(x=x_in_num, y=y_in_num, mode='markers', name="Nodos", marker=dict(size=10, color='white')))
                fig.update_layout(template="plotly_dark", title="Interpolación vs Función Real")
                st.plotly_chart(fig, use_container_width=True)

        elif metodo_sel == "Diferencias Centrales":
            st.subheader("Fórmulas")
            st.latex(
                r"f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}"
            )
            st.latex(
                r"f''(x_i) \approx \frac{f(x_{i+1}) - 2f(x_i) + f(x_{i-1})}{h^2}"
            )
            st.latex(
                r"\text{Error de truncamiento: } O(h^2)"
            )
            with st.expander("📖 Notación"):
                st.markdown("""
| Símbolo | Significado |
|---|---|
| $x_i$ | Punto interior donde se estima la derivada |
| $x_{i-1},\\, x_{i+1}$ | Puntos vecinos anterior y siguiente |
| $f(x_i)$ | Valor de la función en $x_i$ |
| $h$ | Espaciado uniforme entre puntos: $h = x_{i+1} - x_i$ |
| $f'(x_i)$ | Aproximación de la primera derivada en $x_i$ |
| $f''(x_i)$ | Aproximación de la segunda derivada en $x_i$ |
| $O(h^2)$ | El error es proporcional a $h^2$ (orden de precisión) |
""")
            st.subheader("Resultado")
            df = metodo_diferencias_centrales(x_in_num, y_in_num)
            if df is not None: st.table(df)
            else: st.error("Se necesitan al menos 3 puntos.")

        elif metodo_sel == "Simpson 1/3":
            integral, err_trunc, h_step, df_tabla = metodo_simpson_13(func_input, a_simp, b_simp, n_simp)
            if integral is not None:
                st.subheader("Fórmulas")
                st.latex(
                    r"h = \frac{b - a}{n}"
                )
                st.latex(
                    r"I \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2) + \cdots + 4f(x_{n-1}) + f(x_n)\right]"
                )
                st.latex(
                    r"|E_T| \leq \frac{(b-a)\,h^4}{180}\,\max_{\xi \in [a,b]}\left|f^{(4)}(\xi)\right|"
                )
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos (debe ser par) |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_0, x_1, \\ldots, x_n$ | Nodos equiespaciados: $x_k = a + k\\,h$ |
| $f(x_k)$ | Valor de la función en el nodo $x_k$ |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f^{(4)}(\\xi)$ | Cuarta derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:.8f}")
                col_r2.metric("Paso h", f"{h_step:.6f}")
                col_r3.metric("Error Trunc. |Eₜ|", f"{err_trunc:.4e}")
                st.dataframe(df_tabla, use_container_width=True)
                # Gráfico con área sombreada
                x_plot = np.linspace(a_simp, b_simp, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#00cfcc', width=2),
                        fill='tozeroy', fillcolor='rgba(0,207,204,0.15)'
                    ))
                    x_nodes = np.linspace(a_simp, b_simp, n_simp + 1)
                    y_nodes = [evaluar_f(func_input, xi) for xi in x_nodes]
                    fig.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes, mode='markers',
                        name="Nodos Simpson", marker=dict(size=8, color='#ff4b4b')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Simpson 1/3 — ∫f(x)dx ≈ {integral:.6f}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        else: # Raíces
            st.subheader("Fórmulas")
            if metodo_sel == "Bisección":
                st.latex(
                    r"x_n = \frac{a + b}{2}"
                )
                st.latex(
                    r"\text{Error} = \left|\frac{x_n - x_{n-1}}{x_n}\right| \times 100\%"
                )
                st.latex(
                    r"\text{Criterio de cambio de signo: } f(a) \cdot f(b) < 0"
                )
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Extremos del intervalo activo en cada iteración |
| $x_n$ | Punto medio del intervalo: aproximación a la raíz |
| $x_{n-1}$ | Aproximación de la iteración anterior |
| $f(a),\\, f(b)$ | Valores de $f$ en los extremos del intervalo |
| Error (%) | Error relativo porcentual entre iteraciones sucesivas |
""")
            else:  # Newton-Raphson
                st.latex(
                    r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}"
                )
                st.latex(
                    r"f'(x_n) \approx \frac{-f(x_n+2h) + 8f(x_n+h) - 8f(x_n-h) + f(x_n-2h)}{12h}"
                )
                st.latex(
                    r"\text{Error} = \left|\frac{x_{n+1} - x_n}{x_{n+1}}\right| \times 100\%"
                )
                with st.expander("📖 Notación"):
                    st.markdown("""
| Símbolo | Significado |
|---|---|
| $x_n$ | Aproximación actual a la raíz |
| $x_{n+1}$ | Nueva aproximación calculada en la iteración |
| $f(x_n)$ | Valor de la función en $x_n$ |
| $f'(x_n)$ | Derivada de $f$ en $x_n$ (calculada numéricamente) |
| $h$ | Paso para la diferenciación numérica ($h = 10^{-5}$) |
| Error (%) | Error relativo porcentual entre iteraciones sucesivas |
""")
            st.subheader("Resultado")
            df, estado, raiz, err = metodo_biseccion(func_input, a_in, b_in, tol_in, iter_in) if metodo_sel == "Bisección" else metodo_newton_raphson(func_input, x0_in, tol_in, iter_in)
            if df is not None:
                st.success(f"Raíz: {raiz:.8f}")
                st.dataframe(df)