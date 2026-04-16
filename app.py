import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re
import sympy as sp
import scipy.stats as st_stats
import scipy.integrate as spi
import streamlit.components.v1 as components

# Configuración de la página
st.set_page_config(page_title="Santiago Mussi | Numeric Solver", layout="wide")

st.title("Analizador de Métodos Numéricos")
st.markdown("---")

# --- INYECCIÓN DE SHORTCUT OCULTO ---
components.html(
    """
    <script>
    const parent = window.parent.document;
    if (!parent.getElementById('shortcut-listener')) {
        parent.addEventListener('keydown', function(event) {
            if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'f') {
                event.preventDefault();
                let labels = parent.querySelectorAll('div[data-testid="stCheckbox"] label p');
                for (let p of labels) {
                    if (p.textContent.includes('Visor Fórmulas (Ctrl+Shift+F)')) {
                        p.closest('label').click();
                        break;
                    }
                }
            }
        });
        let div = parent.createElement('div');
        div.id = 'shortcut-listener';
        div.style.display = 'none';
        parent.body.appendChild(div);
    }
    
    setTimeout(() => {
        let labels = parent.querySelectorAll('div[data-testid="stCheckbox"] label p');
        for (let p of labels) {
            if (p.textContent.includes('Visor Fórmulas (Ctrl+Shift+F)')) {
                let container = p.closest('div[data-testid="stCheckbox"]');
                if (container) {
                    container.style.opacity = '0';
                    container.style.height = '0px';
                    container.style.overflow = 'hidden';
                    container.style.position = 'absolute';
                }
                break;
            }
        }
    }, 50);
    </script>
    """,
    height=0,
    width=0
)

# --- FUNCIONES DE EVALUACIÓN Y DERIVADA ---

def evaluar_f(f_str, x_val):
    """Evalúa funciones para cálculos numéricos rápidos (gráficos y raíces)"""
    try:
        f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_val, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "exp": np.exp, "log": np.log, 
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "sp": sp
        }
        res = eval(f_proc, {"__builtins__": None}, contexto)
        if np.isnan(float(res)) or np.isinf(float(res)):
            raise ValueError("Valor indefinido")
        return float(res)
    except Exception:
        # Intento de aproximación por límite si hay singularidad matemática (ej. x=0 en sin(x)/x)
        try:
            contexto["x"] = x_val + 1e-12
            res_lim = eval(f_proc, {"__builtins__": None}, contexto)
            if np.isnan(float(res_lim)) or np.isinf(float(res_lim)):
                return None
            return float(res_lim)
        except:
            return None

def evaluar_f_array(f_str, x_arr, y_arr=None):
    """Evalúa funciones con arrays de numpy de forma segura"""
    try:
        f_proc = f_str.replace("^", "**").replace("sen", "sin").replace("ln", "log")
        f_proc = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', f_proc)
        contexto = {
            "np": np, "x": x_arr, "y": y_arr, "sin": np.sin, "cos": np.cos, 
            "tan": np.tan, "exp": np.exp, "log": np.log, 
            "sqrt": np.sqrt, "pi": np.pi, "e": np.e, "sp": sp
        }
        res = eval(f_proc, {"__builtins__": None}, contexto)
        if isinstance(res, (int, float)):
            res = np.full(np.shape(x_arr), res)
        return np.asarray(res, dtype=float)
    except Exception:
        # Fallback iterativo
        res_list = []
        if y_arr is None:
            for xv in np.atleast_1d(x_arr):
                val = evaluar_f(f_str, xv)
                res_list.append(val if val is not None else np.nan)
        else:
            for xv, yv in zip(np.atleast_1d(x_arr), np.atleast_1d(y_arr)):
                contexto_scalar = {
                    "np": np, "x": xv, "y": yv, "sin": np.sin, "cos": np.cos, 
                    "tan": np.tan, "exp": np.exp, "log": np.log, 
                    "sqrt": np.sqrt, "pi": np.pi, "e": np.e
                }
                try:
                    val = eval(f_proc, {"__builtins__": None}, contexto_scalar)
                    res_list.append(float(val) if not (np.isnan(float(val)) or np.isinf(float(val))) else np.nan)
                except:
                    res_list.append(np.nan)
        return np.array(res_list)

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

    # Tabla por nodo: N, Xn, F(Xn), coeficiente Simpson, término ponderado
    tabla = []
    for k in range(n + 1):
        if k == 0 or k == n:
            coef = 1
        elif k % 2 != 0:
            coef = 4
        else:
            coef = 2
        fk = float(y_pts[k])
        tabla.append({
            "N": k,
            "Xₙ": round(float(x_pts[k]), 8),
            "F(Xₙ)": round(fk, 8),
            "Coef.": coef,
            "Coef. × F(Xₙ)": round(coef * fk, 8),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla)


# --- INTEGRACIÓN: SIMPSON 3/8 ---

def metodo_simpson_38(f_str, a, b, n, a_str="", b_str=""):
    """Regla de Simpson 3/8 compuesta. n debe ser múltiplo de 3."""
    if n % 3 != 0:
        n += (3 - (n % 3))
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts = np.array([evaluar_f(f_str, xi) for xi in x_pts])
    if any(v is None for v in y_pts):
        return None, None, None, None

    # Suma compuesta Simpson 3/8
    suma = y_pts[0] + y_pts[-1]
    for i in range(1, n):
        if i % 3 == 0:
            suma += 2 * y_pts[i]
        else:
            suma += 3 * y_pts[i]
    integral = (3 * h / 8) * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^4/80 * max|f''''(xi)|
    h_num = max(h * 0.1, 1e-4)
    f4_vals = []
    for xi in x_pts[2:-2]:
        vals = [evaluar_f(f_str, xi + k * h_num) for k in [-2, -1, 0, 1, 2]]
        if all(v is not None for v in vals):
            f4 = (vals[0] - 4*vals[1] + 6*vals[2] - 4*vals[3] + vals[4]) / h_num**4
            f4_vals.append(abs(f4))
    f4_max = max(f4_vals) if f4_vals else 0.0
    error_trunc = abs((b - a) * h**4 * f4_max / 80)

    def format_xn(val_float, k):
        try:
            a_s = str(a_str).replace('π', 'pi').lower()
            b_s = str(b_str).replace('π', 'pi').lower()
            if 'pi' in a_s or 'pi' in b_s:
                a_sym = sp.sympify(a_s)
                b_sym = sp.sympify(b_s)
                a_c = sp.simplify(a_sym / sp.pi)
                b_c = sp.simplify(b_sym / sp.pi)
                if a_c.is_Rational and b_c.is_Rational:
                    h_c = (b_c - a_c) / n
                    D = int(sp.lcm(a_c.q, h_c.q))
                    num = int((a_c + k * h_c) * D)
                    if num == 0:
                        return "0"
                    
                    if D == 1:
                        if num == 1: return "π"
                        elif num == -1: return "-π"
                        return f"{num}π"
                    
                    if num == 1: return f"π/{D}"
                    elif num == -1: return f"-π/{D}"
                    return f"{num}π/{D}"
        except:
            pass
        return str(round(float(val_float), 8))

    # Tabla por nodo
    tabla = []
    for k in range(n + 1):
        if k == 0 or k == n:
            coef = 1
        elif k % 3 == 0:
            coef = 2
        else:
            coef = 3
        fk = float(y_pts[k])
        tabla.append({
            "N": k,
            "Xₙ": format_xn(x_pts[k], k),
            "F(Xₙ)": round(fk, 8),
            "Coef.": coef,
            "Coef. × F(Xₙ)": round(coef * fk, 8),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla)


# --- INTEGRACIÓN: TRAPECIOS ---

def metodo_trapecios(f_str, a, b, n):
    """Regla de Trapecios compuesta con n subintervalos."""
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    y_pts = np.array([evaluar_f(f_str, xi) for xi in x_pts])
    if any(v is None for v in y_pts):
        return None, None, None, None

    # Suma compuesta: extremos x1, interiores x2
    suma = y_pts[0] + y_pts[-1] + 2 * np.sum(y_pts[1:-1])
    integral = (h / 2) * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^2/12 * max|f''(xi)|
    h_num = max(h * 0.1, 1e-4)
    f2_vals = []
    for xi in x_pts[1:-1]:
        vals = [evaluar_f(f_str, xi + k * h_num) for k in [-1, 0, 1]]
        if all(v is not None for v in vals):
            f2 = (vals[0] - 2*vals[1] + vals[2]) / h_num**2
            f2_vals.append(abs(f2))
    f2_max = max(f2_vals) if f2_vals else 0.0
    error_trunc = abs((b - a) * h**2 * f2_max / 12)

    # Tabla por nodo
    tabla = []
    for k in range(n + 1):
        coef = 1 if (k == 0 or k == n) else 2
        fk = float(y_pts[k])
        tabla.append({
            "N": k,
            "Xₙ": round(float(x_pts[k]), 8),
            "F(Xₙ)": round(fk, 8),
            "Coef.": coef,
            "Coef. × F(Xₙ)": round(coef * fk, 8),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla)


# --- INTEGRACIÓN: RECTÁNGULO MEDIO ---

def metodo_rectangulo_medio(f_str, a, b, n):
    """Regla de Rectángulo Medio (Punto Medio) compuesta con n subintervalos."""
    h = (b - a) / n
    x_pts = np.linspace(a, b, n + 1)
    x_mid = (x_pts[:-1] + x_pts[1:]) / 2
    y_mid = np.array([evaluar_f(f_str, xi) for xi in x_mid])
    if any(v is None for v in y_mid):
        return None, None, None, None

    # Suma compuesta
    suma = np.sum(y_mid)
    integral = h * suma

    # Error de truncamiento: |E_T| <= (b-a)*h^2/24 * max|f''(xi)|
    h_num = max(h * 0.1, 1e-4)
    f2_vals = []
    for xi in x_mid:
        vals = [evaluar_f(f_str, xi + k * h_num) for k in [-1, 0, 1]]
        if all(v is not None for v in vals):
            f2 = (vals[0] - 2*vals[1] + vals[2]) / h_num**2
            f2_vals.append(abs(f2))
    f2_max = max(f2_vals) if f2_vals else 0.0
    error_trunc = abs((b - a) * h**2 * f2_max / 24)

    # Tabla por nodo (puntos medios e intervalos)
    tabla = []
    for k in range(n):
        fk = float(y_mid[k])
        tabla.append({
            "N": k+1,
            "Xₙ_inf": round(float(x_pts[k]), 8),
            "Xₙ_sup": round(float(x_pts[k+1]), 8),
            "Xₘ (Pto. Medio)": round(float(x_mid[k]), 8),
            "F(Xₘ)": round(fk, 8),
            "Coef.": 1,
            "Coef. × F(Xₘ)": round(1 * fk, 8),
        })
    return integral, error_trunc, h, pd.DataFrame(tabla)


# --- INTEGRACIÓN: MONTECARLO ---

def metodo_montecarlo(f_str, a, b, n, conf_level=95.0, seed=None, antithetic=False):
    if seed is not None:
        np.random.seed(seed)
        
    if antithetic:
        half_n = n // 2
        x_half = np.random.uniform(a, b, half_n)
        x_rand = np.concatenate([x_half, a + b - x_half])
        if n % 2 != 0:
            x_rand = np.append(x_rand, np.random.uniform(a, b, 1))
    else:
        x_rand = np.random.uniform(a, b, n)
        
    y_eval = evaluar_f_array(f_str, x_rand)
    
    vol = (b - a)
    valid_mask = ~np.isnan(y_eval)
    valid_n = np.sum(valid_mask)
    if valid_n == 0:
        return 0.0, 0.0, vol, 0.0, 0.0, pd.DataFrame(), x_rand, y_eval, n
        
    y_valid = y_eval[valid_mask]
    integral = vol * np.mean(y_valid)
    
    var = np.var(y_valid, ddof=1) if valid_n > 1 else 0.0
    std_dev = np.sqrt(var)
    std_error = vol * np.sqrt(var / valid_n)
    
    alpha = 1 - (conf_level / 100.0)
    z = st_stats.norm.ppf(1 - alpha/2)
    margin = z * std_error
    ic_lower = integral - margin
    ic_upper = integral + margin
    
    tabla = []
    for i in range(min(10, n)):
        tabla.append({
            "Punto i": i+1,
            "x_i": round(x_rand[i], 6),
            "f(x_i)": round(y_eval[i], 6) if not np.isnan(y_eval[i]) else "Indefinido",
        })
    df_tabla = pd.DataFrame(tabla)
    
    invalid_count = n - valid_n
    return integral, std_error, std_dev, vol, ic_lower, ic_upper, df_tabla, x_rand, y_eval, invalid_count

def metodo_montecarlo_doble(f_str, a_x, b_x, a_y, b_y, n, conf_level=95.0, seed=None, antithetic=False):
    if seed is not None:
        np.random.seed(seed)
        
    if antithetic:
        half_n = n // 2
        x_half = np.random.uniform(a_x, b_x, half_n)
        y_half = np.random.uniform(a_y, b_y, half_n)
        x_rand = np.concatenate([x_half, a_x + b_x - x_half])
        y_rand = np.concatenate([y_half, a_y + b_y - y_half])
        if n % 2 != 0:
            x_rand = np.append(x_rand, np.random.uniform(a_x, b_x, 1))
            y_rand = np.append(y_rand, np.random.uniform(a_y, b_y, 1))
    else:
        x_rand = np.random.uniform(a_x, b_x, n)
        y_rand = np.random.uniform(a_y, b_y, n)
        
    z_eval = evaluar_f_array(f_str, x_rand, y_rand)
    
    area = (b_x - a_x) * (b_y - a_y)
    valid_mask = ~np.isnan(z_eval)
    valid_n = np.sum(valid_mask)
    if valid_n == 0:
        return 0.0, 0.0, area, 0.0, 0.0, pd.DataFrame(), x_rand, y_rand, z_eval, n
        
    z_valid = z_eval[valid_mask]
    integral = area * np.mean(z_valid)
    
    var = np.var(z_valid, ddof=1) if valid_n > 1 else 0.0
    std_dev = np.sqrt(var)
    std_error = area * np.sqrt(var / valid_n)
    
    alpha = 1 - (conf_level / 100.0)
    z = st_stats.norm.ppf(1 - alpha/2)
    margin = z * std_error
    ic_lower = integral - margin
    ic_upper = integral + margin
    
    tabla = []
    for i in range(min(10, n)):
        tabla.append({
            "Punto i": i+1,
            "x_i": round(x_rand[i], 6),
            "y_i": round(y_rand[i], 6),
            "f(x_i, y_i)": round(z_eval[i], 6) if not np.isnan(z_eval[i]) else "Indefinido",
        })
    df_tabla = pd.DataFrame(tabla)
    
    invalid_count = n - valid_n
    return integral, std_error, std_dev, area, ic_lower, ic_upper, df_tabla, x_rand, y_rand, z_eval, invalid_count


# --- MÉTODOS DE RAÍCES ---

def formatear_error(valor):
    if valor == 0:
        return "0.0000"
    if abs(valor) < 0.0001:
        s = f"{valor:.4e}"
        base, exp = s.split('e')
        exp_int = int(exp)  # Convierte '-05' a -5
        superscripts = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        exp_sup = ''.join(superscripts.get(c, c) for c in str(exp_int))
        return f"{base} × 10{exp_sup}"
    return f"{valor:.6f}"


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
mostrar_formulas = st.sidebar.checkbox("Visor Fórmulas (Ctrl+Shift+F)", value=False)
metodo_sel = st.sidebar.selectbox("Selecciona Método",
    ["Bisección", "Newton-Raphson", "Interpolación Lagrange", "Diferencias Centrales", "Rectángulo Medio", "Trapecios", "Simpson 1/3", "Simpson 3/8", "Montecarlo", "Montecarlo Doble"])

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
    elif metodo_sel == "Simpson 3/8":
        func_input = st.text_input("f(x):", value="x**3")
        a_simp38_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_simp38_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_simp38 = int(st.number_input("Nº subintervalos n (múltiplo de 3)", value=3, min_value=3, step=3))
        if n_simp38 % 3 != 0:
            st.warning("n debe ser múltiplo de 3 — se ajustará automáticamente.")
        try:
            a_simp38 = float(sp.sympify(a_simp38_str).evalf())
            b_simp38 = float(sp.sympify(b_simp38_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_simp38, b_simp38 = 0.0, 1.0
    elif metodo_sel == "Trapecios":
        func_input = st.text_input("f(x):", value="x**2")
        a_trap_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_trap_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_trap = int(st.number_input("Nº subintervalos n", value=4, min_value=1, step=1))
        try:
            a_trap = float(sp.sympify(a_trap_str).evalf())
            b_trap = float(sp.sympify(b_trap_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_trap, b_trap = 0.0, 1.0
    elif metodo_sel == "Rectángulo Medio":
        func_input = st.text_input("f(x):", value="x**2")
        a_rect_str = st.text_input("Límite inferior a", value="0", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        b_rect_str = st.text_input("Límite superior b", value="1", help="Podés escribir expresiones como pi/2, sqrt(2), 2*pi, etc.")
        n_rect = int(st.number_input("Nº subintervalos n", value=4, min_value=1, step=1))
        try:
            a_rect = float(sp.sympify(a_rect_str).evalf())
            b_rect = float(sp.sympify(b_rect_str).evalf())
        except:
            st.error("Límites inválidos. Usá expresiones como: pi/2, sqrt(2), 1.5, 2*pi")
            a_rect, b_rect = 0.0, 1.0
    elif metodo_sel == "Montecarlo":
        func_input = st.text_input("f(x):", value="sin(x)")
        a_mc_str = st.text_input("Límite inferior a", value="0", help="Usar expresiones: pi/2, sqrt(2)")
        b_mc_str = st.text_input("Límite superior b", value="pi", help="Usar expressions: pi/2, sqrt(2)")
        n_mc = int(st.number_input("Cantidad de Puntos N", value=10000, step=1000))
        conf_mc = st.number_input("Nivel de Confianza (%)", value=95.0, min_value=1.0, max_value=99.9)
        
        col_m1, col_m2 = st.columns(2)
        use_seed_mc = col_m1.checkbox("Fijar Semilla", value=True, key="smc1")
        seed_mc = col_m2.number_input("Semilla", value=42, step=1, key="smc2") if use_seed_mc else None
        antithetic_mc = st.checkbox("Variables Antitéticas", value=False, help="Técnica de reducción de varianza.", key="amc1")
        
        try:
            a_mc = float(sp.sympify(a_mc_str).evalf())
            b_mc = float(sp.sympify(b_mc_str).evalf())
        except:
            st.error("Límites inválidos.")
            a_mc, b_mc = 0.0, 3.14159
    elif metodo_sel == "Montecarlo Doble":
        func_input = st.text_input("f(x, y):", value="x**2 + y**2")
        col_ax, col_bx = st.columns(2)
        a_x_mc_str = col_ax.text_input("Lím. inf. X (a)", value="0")
        b_x_mc_str = col_bx.text_input("Lím. sup. X (b)", value="1")
        col_ay, col_by = st.columns(2)
        a_y_mc_str = col_ay.text_input("Lím. inf. Y (c)", value="0")
        b_y_mc_str = col_by.text_input("Lím. sup. Y (d)", value="2")
        n_mc2 = int(st.number_input("Cantidad de Puntos N", value=10000, step=1000))
        conf_mc2 = st.number_input("Nivel de Confianza (%)", value=95.0, min_value=1.0, max_value=99.9)
        
        col_md1, col_md2 = st.columns(2)
        use_seed_mc2 = col_md1.checkbox("Fijar Semilla", value=True, key="smcd1")
        seed_mc2 = col_md2.number_input("Semilla", value=42, step=1, key="smcd2") if use_seed_mc2 else None
        antithetic_mc2 = st.checkbox("Variables Antitéticas", value=False, help="Técnica de reducción de varianza.", key="amcd1")
        
        try:
            a_x_mc = float(sp.sympify(a_x_mc_str).evalf())
            b_x_mc = float(sp.sympify(b_x_mc_str).evalf())
            a_y_mc = float(sp.sympify(a_y_mc_str).evalf())
            b_y_mc = float(sp.sympify(b_y_mc_str).evalf())
        except:
            st.error("Límites inválidos.")
            a_x_mc, b_x_mc, a_y_mc, b_y_mc = 0.0, 1.0, 0.0, 2.0
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
            if mostrar_formulas:
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
            if mostrar_formulas:
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
                if mostrar_formulas:
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
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
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

        elif metodo_sel == "Simpson 3/8":
            integral, err_trunc, h_step, df_tabla = metodo_simpson_38(func_input, a_simp38, b_simp38, n_simp38, a_simp38_str, b_simp38_str)
            if integral is not None:
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"I \approx \frac{3h}{8}\left[f(x_0) + 3f(x_1) + 3f(x_2) + 2f(x_3) + \cdots + f(x_n)\right]"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^4}{80}\,\max_{\xi \in [a,b]}\left|f^{(4)}(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos (debe ser múltiplo de 3) |
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
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(df_tabla, use_container_width=True)
                # Gráfico con área sombreada
                x_plot = np.linspace(a_simp38, b_simp38, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#e03ce6', width=2),
                        fill='tozeroy', fillcolor='rgba(224,60,230,0.15)'
                    ))
                    x_nodes = np.linspace(a_simp38, b_simp38, n_simp38 + 1)
                    y_nodes = [evaluar_f(func_input, xi) for xi in x_nodes]
                    fig.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes, mode='markers',
                        name="Nodos Simpson 3/8", marker=dict(size=8, color='#00cfcc')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Simpson 3/8 — ∫f(x)dx ≈ {integral:.6f}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Trapecios":
            integral, err_trunc, h_step, df_tabla = metodo_trapecios(func_input, a_trap, b_trap, n_trap)
            if integral is not None:
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"I \approx \frac{h}{2}\left[f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)\right]"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^2}{12}\,\max_{\xi \in [a,b]}\left|f''(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_0, x_1, \\ldots, x_n$ | Nodos equiespaciados: $x_k = a + k\\,h$ |
| $f(x_k)$ | Valor de la función en el nodo $x_k$ |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f''(\\xi)$ | Segunda derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:.8f}")
                col_r2.metric("Paso h", f"{h_step:.6f}")
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(df_tabla, use_container_width=True)
                # Gráfico con área sombreada y trapecios
                x_plot = np.linspace(a_trap, b_trap, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#f0a500', width=2),
                        fill='tozeroy', fillcolor='rgba(240,165,0,0.12)'
                    ))
                    x_nodes = np.linspace(a_trap, b_trap, n_trap + 1)
                    y_nodes = [evaluar_f(func_input, xi) for xi in x_nodes]
                    # Dibuja los trapecios como líneas verticales desde el eje
                    for xi, yi in zip(x_nodes, y_nodes):
                        fig.add_shape(type="line",
                            x0=xi, y0=0, x1=xi, y1=yi,
                            line=dict(color='rgba(240,165,0,0.5)', width=1, dash='dot')
                        )
                    fig.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes, mode='markers+lines',
                        name="Nodos Trapecios",
                        line=dict(color='#f0a500', width=1, dash='dash'),
                        marker=dict(size=8, color='white')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Trapecios — ∫f(x)dx ≈ {integral:.6f}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Rectángulo Medio":
            integral, err_trunc, h_step, df_tabla = metodo_rectangulo_medio(func_input, a_rect, b_rect, n_rect)
            if integral is not None:
                if mostrar_formulas:
                    st.subheader("Fórmulas")
                    st.latex(
                        r"h = \frac{b - a}{n}"
                    )
                    st.latex(
                        r"x_{m_i} = \frac{x_i + x_{i+1}}{2}"
                    )
                    st.latex(
                        r"I \approx h \sum_{i=0}^{n-1} f(x_{m_i})"
                    )
                    st.latex(
                        r"|E_T| \leq \frac{(b-a)\,h^2}{24}\,\max_{\xi \in [a,b]}\left|f''(\xi)\right|"
                    )
                    with st.expander("📖 Notación"):
                        st.markdown("""
| Símbolo | Significado |
|---|---|
| $a,\\, b$ | Límites inferior y superior del intervalo de integración |
| $n$ | Número de subintervalos |
| $h$ | Ancho de cada subintervalo: $h = (b-a)/n$ |
| $x_{m_i}$ | Punto medio del subintervalo $i$ |
| $f(x_{m_i})$ | Valor de la función en el punto medio |
| $I$ | Valor aproximado de la integral $\\int_a^b f(x)\\,dx$ |
| $E_T$ | Error de truncamiento de la fórmula compuesta |
| $f''(\\xi)$ | Segunda derivada de $f$ en algún punto $\\xi \\in [a,b]$ |
""")
                st.subheader("Resultado")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Integral ≈", f"{integral:.8f}")
                col_r2.metric("Paso h", f"{h_step:.6f}")
                col_r3.metric("Error Trunc. |Eₜ|", formatear_error(err_trunc))
                st.dataframe(df_tabla, use_container_width=True)
                # Gráfico con área sombreada y rectángulos
                x_plot = np.linspace(a_rect, b_rect, 300)
                y_plot = [evaluar_f(func_input, xi) for xi in x_plot]
                if None not in y_plot:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_plot, y=y_plot, name="f(x)",
                        line=dict(color='#86e012', width=2),
                        fill='tozeroy', fillcolor='rgba(134,224,18,0.12)'
                    ))
                    x_nodes = np.linspace(a_rect, b_rect, n_rect + 1)
                    x_mid = (x_nodes[:-1] + x_nodes[1:]) / 2
                    y_mid = [evaluar_f(func_input, xi) for xi in x_mid]
                    # Dibujar los rectángulos
                    for x0, x1, ym in zip(x_nodes[:-1], x_nodes[1:], y_mid):
                        fig.add_shape(type="rect",
                            x0=x0, y0=0, x1=x1, y1=ym,
                            line=dict(color='rgba(134,224,18,0.8)', width=1, dash='solid'),
                            fillcolor='rgba(134,224,18,0.3)'
                        )
                    fig.add_trace(go.Scatter(
                        x=x_mid, y=y_mid, mode='markers',
                        name="Puntos Medios", marker=dict(size=8, color='white')
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        title=f"Rectángulo Medio — ∫f(x)dx ≈ {integral:.6f}",
                        xaxis_title="x", yaxis_title="f(x)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo evaluar f(x) en el intervalo. Verificá la función.")

        elif metodo_sel == "Montecarlo":
            integral, err_est, s_dev, vol, ic_low, ic_up, df_tabla, x_r, y_r, invalid_count = metodo_montecarlo(
                func_input, a_mc, b_mc, n_mc, conf_mc, seed=seed_mc, antithetic=antithetic_mc)
            
            if invalid_count > 0:
                st.warning(f"⚠️ Se omitieron {invalid_count} muestras que resultaron en valores indefinidos en la función.")
            
            # --- CÁLCULO DE ERROR VERDADERO CON SCIPY ---
            def func_scipy(x):
                val = evaluar_f(func_input, x)
                return val if val is not None else 0.0
            
            try:
                exact_val, exact_err = spi.quad(func_scipy, a_mc, b_mc)
                true_error_perc = abs(integral - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
            except:
                exact_val, true_error_perc = None, None

            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(r"I \approx V \cdot \frac{1}{N}\sum_{i=1}^{N} f(x_i)")
                st.latex(r"\text{Var} = \frac{1}{N-1}\sum_{i=1}^{N} \left(f(x_i) - \bar{f}\right)^2")
                st.latex(r"EE = V \cdot \sqrt{\frac{\text{Var}}{N}}")
                st.latex(r"IC = I \pm z_{\alpha/2} EE")
            
            st.subheader("Resultado")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Integral ≈", f"{integral:.6f}")
            if exact_val is not None:
                c2.metric("Valor Exacto (Scipy)", f"{exact_val:.6f}", f"Err: {true_error_perc:.4f}%", delta_color="inverse")
            else:
                c2.metric("Valor Exacto", "N/A")
            c3.metric("Desv. Estándar (S)", f"{s_dev:.6f}")
            c4.metric("Error Estándar (EE)", formatear_error(err_est))
            c5.metric(f"IC {conf_mc}%", f"[{ic_low:.4f}, {ic_up:.4f}]")
            st.write(f"**Área/Volumen Región:** {vol:.4f}")
            st.dataframe(df_tabla, use_container_width=True)
            
            # --- TABS PARA GRÁFICOS AVANZADOS ---
            tab1, tab2, tab3 = st.tabs(["Muestreo y Función", "Convergencia y Confianza", "Distribución de f(x)"])
            
            with tab1:
                fig = go.Figure()
                x_plot = np.linspace(a_mc, b_mc, 300)
                y_plot = evaluar_f_array(func_input, x_plot)
                fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)", line=dict(color='#86e012')))
                
                n_plot = min(n_mc, 1000)
                fig.add_trace(go.Scatter(x=x_r[:n_plot], y=y_r[:n_plot], mode='markers', name="Muestras", marker=dict(size=4, color='rgba(255,255,255,0.4)')))
                fig.update_layout(template="plotly_dark", title=f"Montecarlo — Función Integrada")
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                # Calculo de Convergencia Vectorizado ignorando NaNs
                valid_mask = ~np.isnan(y_r)
                y_valid = y_r[valid_mask]
                n_valid = len(y_valid)
                
                if n_valid > 0:
                    cum_mean = np.cumsum(y_valid) / np.arange(1, n_valid + 1)
                    cum_integral = vol * cum_mean
                    
                    if n_valid > 1:
                        cum_var = pd.Series(y_valid).expanding().var(ddof=1).fillna(0).values
                        cum_std_error = vol * np.sqrt(cum_var / np.arange(1, n_valid + 1))
                    else:
                        cum_std_error = np.zeros(n_valid)
                else:
                    cum_integral, cum_std_error = np.zeros(0), np.zeros(0)
                
                alpha = 1 - (conf_mc / 100.0)
                z_val = st_stats.norm.ppf(1 - alpha/2)
                margin = z_val * cum_std_error
                
                fig_conv = go.Figure()
                
                if n_valid > 5000:
                    idx = np.unique(np.geomspace(1, n_valid, 1000).astype(int)) - 1
                else:
                    idx = np.arange(n_valid)
                
                n_idx = idx + 1
                
                if exact_val is not None:
                    fig_conv.add_trace(go.Scatter(x=n_idx, y=np.full_like(n_idx, exact_val, dtype=float), mode='lines', name='Valor Exacto', line=dict(color='white', dash='dash')))

                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] + margin[idx],
                    mode='lines', line=dict(color='rgba(255, 75, 75, 0)'),
                    name='Límite Superior', showlegend=False
                ))
                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] - margin[idx],
                    mode='lines', fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)',
                    line=dict(color='rgba(255, 75, 75, 0)'),
                    name=f'Intervalo Confianza {conf_mc}%'
                ))
                fig_conv.add_trace(go.Scatter(x=n_idx, y=cum_integral[idx], mode='lines', name='Valor Estimado', line=dict(color='#00cfcc', width=2)))
                
                fig_conv.update_layout(template="plotly_dark", title="Convergencia de la Aproximación (N progresivo)", xaxis_title="Número de Muestras (N)", yaxis_title="Integral Estimada")
                st.plotly_chart(fig_conv, use_container_width=True)

            with tab3:
                fig_hist = go.Figure(data=[go.Histogram(x=y_r, nbinsx=50, marker_color='#e03ce6', opacity=0.75)])
                fig_hist.update_layout(template="plotly_dark", title="Distribución de Evaluaciones Aleatorias f(x)", xaxis_title="f(x)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_hist, use_container_width=True)

        elif metodo_sel == "Montecarlo Doble":
            integral, err_est, s_dev, area_xy, ic_low, ic_up, df_tabla, x_r, y_r, z_r, invalid_count = metodo_montecarlo_doble(
                func_input, a_x_mc, b_x_mc, a_y_mc, b_y_mc, n_mc2, conf_mc2, seed=seed_mc2, antithetic=antithetic_mc2)
            
            if invalid_count > 0:
                st.warning(f"⚠️ Se omitieron {invalid_count} muestras que resultaron en valores indefinidos en la función.")
            
            # --- CÁLCULO DE ERROR VERDADERO DOBLE CON SCIPY ---
            def func_scipy_dbl(y, x):
                return float(evaluar_f_array(func_input, np.array([x]), np.array([y]))[0])
            
            try:
                exact_val, exact_err = spi.dblquad(func_scipy_dbl, a_x_mc, b_x_mc, lambda x: a_y_mc, lambda x: b_y_mc)
                true_error_perc = abs(integral - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
            except:
                exact_val, true_error_perc = None, None
                
            if mostrar_formulas:
                st.subheader("Fórmulas")
                st.latex(r"I \approx S \cdot \frac{1}{N}\sum_{i=1}^{N} f(x_i, y_i)")
                st.latex(r"\text{Var} = \frac{1}{N-1}\sum_{i=1}^{N} \left(f(x_i, y_i) - \bar{f}\right)^2")
                st.latex(r"EE = S \cdot \sqrt{\frac{\text{Var}}{N}}")
                st.latex(r"IC = I \pm z_{\alpha/2} EE")
            
            st.subheader("Resultado")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Integral Acumulada ≈", f"{integral:.6f}")
            if exact_val is not None:
                c2.metric("Valor Exacto (Scipy)", f"{exact_val:.6f}", f"Err: {true_error_perc:.4f}%", delta_color="inverse")
            else:
                c2.metric("Valor Exacto", "N/A")
            c3.metric("Desv. Estándar (S)", f"{s_dev:.6f}")
            c4.metric("Error Estándar (EE)", formatear_error(err_est))
            c5.metric(f"IC {conf_mc2}%", f"[{ic_low:.4f}, {ic_up:.4f}]")
            st.write(f"**Área Integración:** {area_xy:.4f}")
            st.dataframe(df_tabla, use_container_width=True)
            
            tab1, tab2, tab3 = st.tabs(["Muestreo 3D", "Convergencia y Confianza", "Distribución de f(x,y)"])
            
            with tab1:
                fig = go.Figure()
                x_m = np.linspace(a_x_mc, b_x_mc, 40)
                y_m = np.linspace(a_y_mc, b_y_mc, 40)
                X, Y = np.meshgrid(x_m, y_m)
                Z = evaluar_f_array(func_input, X.ravel(), Y.ravel()).reshape(X.shape)
                fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8, name="f(x,y)"))
                
                n_plot = min(n_mc2, 500)
                fig.add_trace(go.Scatter3d(x=x_r[:n_plot], y=y_r[:n_plot], z=z_r[:n_plot], mode='markers', name="Muestras", marker=dict(size=3, color='red')))
                fig.update_layout(template="plotly_dark", title=f"Montecarlo Doble — Visualización 3D")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                valid_mask = ~np.isnan(z_r)
                z_valid = z_r[valid_mask]
                n_valid = len(z_valid)
                
                if n_valid > 0:
                    cum_mean = np.cumsum(z_valid) / np.arange(1, n_valid + 1)
                    cum_integral = area_xy * cum_mean
                    
                    if n_valid > 1:
                        cum_var = pd.Series(z_valid).expanding().var(ddof=1).fillna(0).values
                        cum_std_error = area_xy * np.sqrt(cum_var / np.arange(1, n_valid + 1))
                    else:
                        cum_std_error = np.zeros(n_valid)
                else:
                    cum_integral, cum_std_error = np.zeros(0), np.zeros(0)
                
                alpha = 1 - (conf_mc2 / 100.0)
                z_val = st_stats.norm.ppf(1 - alpha/2)
                margin = z_val * cum_std_error
                
                fig_conv = go.Figure()
                if n_valid > 5000:
                    idx = np.unique(np.geomspace(1, n_valid, 1000).astype(int)) - 1
                else:
                    idx = np.arange(n_valid)
                n_idx = idx + 1
                
                if exact_val is not None:
                    fig_conv.add_trace(go.Scatter(x=n_idx, y=np.full_like(n_idx, exact_val, dtype=float), mode='lines', name='Valor Exacto', line=dict(color='white', dash='dash')))

                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] + margin[idx],
                    mode='lines', line=dict(color='rgba(255, 75, 75, 0)'),
                    name='Límite Superior', showlegend=False
                ))
                fig_conv.add_trace(go.Scatter(
                    x=n_idx, y=cum_integral[idx] - margin[idx],
                    mode='lines', fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)',
                    line=dict(color='rgba(255, 75, 75, 0)'),
                    name=f'Intervalo Confianza {conf_mc2}%'
                ))
                fig_conv.add_trace(go.Scatter(x=n_idx, y=cum_integral[idx], mode='lines', name='Valor Estimado', line=dict(color='#00cfcc', width=2)))
                
                fig_conv.update_layout(template="plotly_dark", title="Convergencia Iterativa de Montecarlo Doble", xaxis_title="Número de Muestras (N)", yaxis_title="Integral Estimada")
                st.plotly_chart(fig_conv, use_container_width=True)

            with tab3:
                fig_hist = go.Figure(data=[go.Histogram(x=z_r, nbinsx=50, marker_color='#00cfcc', opacity=0.75)])
                fig_hist.update_layout(template="plotly_dark", title="Distribución de Evaluaciones Aleatorias f(x,y)", xaxis_title="f(x,y)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_hist, use_container_width=True)

        else: # Raíces
            if mostrar_formulas:
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