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
    ["Bisección", "Newton-Raphson", "Interpolación Lagrange", "Diferencias Centrales"])

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
            st.subheader("Resultado")
            st.latex(f"P(x) = {sp.latex(poly)}")
            
            # Evaluación y Error
            x_ev_sym = sp.sympify(x_eval)
            val_p = poly.subs(sp.symbols('x'), x_ev_sym)
            st.info(f"Evaluación: P({x_eval}) = {float(val_p.evalf()):.6f}")
            st.latex(f"P({sp.latex(x_ev_sym)}) = {sp.latex(sp.simplify(val_p))}")
            
            if func_teorica:
                v_real = evaluar_f(func_teorica, float(x_ev_sym.evalf()))
                if v_real is not None:
                    err_abs = abs(float(val_p.evalf()) - v_real)
                    st.warning(f"Error Real (Epsilon): {err_abs:.8f}")

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
            df = metodo_diferencias_centrales(x_in_num, y_in_num)
            if df is not None: st.table(df)
            else: st.error("Se necesitan al menos 3 puntos.")

        else: # Raíces
            df, estado, raiz, err = metodo_biseccion(func_input, a_in, b_in, tol_in, iter_in) if metodo_sel == "Bisección" else metodo_newton_raphson(func_input, x0_in, tol_in, iter_in)
            if df is not None:
                st.success(f"Raíz: {raiz:.8f}")
                st.dataframe(df)