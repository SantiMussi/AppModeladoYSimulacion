"""Motor de cálculo para el simulador de sistemas dinámicos."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import sympy as sp

SAFE_DICT = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "pi": np.pi, "e": np.e,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
}

def parse_expr(expr_str, variables, params):
    safe = {**SAFE_DICT, **params}
    def func(*args):
        local = dict(zip(variables, args))
        local.update(safe)
        try:
            return eval(expr_str, {"__builtins__": {}}, local)
        except Exception:
            return np.zeros_like(args[0]) if hasattr(args[0], "__len__") else 0.0
    return func

def parse_expr_sympy(expr_str, variables, params):
    syms = {v: sp.Symbol(v) for v in variables}
    param_syms = {k: sp.Symbol(k) for k in params}
    all_syms = {**syms, **param_syms}
    try:
        expr = sp.sympify(expr_str, locals=all_syms)
        for k, v in params.items():
            expr = expr.subs(sp.Symbol(k), v)
        return expr, [syms[v] for v in variables]
    except Exception:
        return None, None

def find_equilibria_1d(expr_str, params, x_range=(-10, 10), n_seeds=200):
    f = parse_expr(expr_str, ["x"], params)
    seeds = np.linspace(x_range[0], x_range[1], n_seeds)
    equilibria = []
    for s in seeds:
        try:
            root, info, ier, _ = fsolve(lambda x: f(x), s, full_output=True)
            if ier == 1 and abs(info["fvec"][0]) < 1e-8:
                rv = float(root[0])
                if x_range[0] <= rv <= x_range[1]:
                    if all(abs(rv - eq) > 1e-4 for eq in equilibria):
                        equilibria.append(rv)
        except Exception:
            continue
    equilibria.sort()
    return equilibria

def classify_equilibrium_1d(expr_str, params, eq_point):
    expr, syms = parse_expr_sympy(expr_str, ["x"], params)
    if expr is None:
        return "Indeterminado", 0
    x_sym = syms[0]
    df = sp.diff(expr, x_sym)
    try:
        df_val = float(df.subs(x_sym, eq_point))
    except Exception:
        return "Indeterminado", 0
    if abs(df_val) < 1e-8:
        return "Semi-estable", df_val
    elif df_val < 0:
        return "Estable (Atractor)", df_val
    else:
        return "Inestable (Repulsor)", df_val

def find_equilibria_2d(fx_str, fy_str, params, xy_range=(-10, 10), n_seeds=15):
    fx = parse_expr(fx_str, ["x", "y"], params)
    fy = parse_expr(fy_str, ["x", "y"], params)
    def system(xy):
        return [fx(xy[0], xy[1]), fy(xy[0], xy[1])]
    sx = np.linspace(xy_range[0], xy_range[1], n_seeds)
    sy = np.linspace(xy_range[0], xy_range[1], n_seeds)
    equilibria = []
    for a in sx:
        for b in sy:
            try:
                root, info, ier, _ = fsolve(system, [a, b], full_output=True)
                if ier == 1 and np.linalg.norm(info["fvec"]) < 1e-8:
                    rx, ry = float(root[0]), float(root[1])
                    if xy_range[0] <= rx <= xy_range[1] and xy_range[0] <= ry <= xy_range[1]:
                        if all(np.sqrt((rx-ex)**2+(ry-ey)**2)>1e-3 for ex,ey in equilibria):
                            equilibria.append((rx, ry))
            except Exception:
                continue
    return equilibria

def classify_equilibrium_2d(fx_str, fy_str, params, eq_point):
    h = 1e-7
    fx = parse_expr(fx_str, ["x", "y"], params)
    fy = parse_expr(fy_str, ["x", "y"], params)
    x0, y0 = eq_point
    try:
        J = np.array([
            [(fx(x0+h,y0)-fx(x0-h,y0))/(2*h),(fx(x0,y0+h)-fx(x0,y0-h))/(2*h)],
            [(fy(x0+h,y0)-fy(x0-h,y0))/(2*h),(fy(x0,y0+h)-fy(x0,y0-h))/(2*h)]
        ])
        eigs = np.linalg.eigvals(J)
        re_p = np.real(eigs)
        im_p = np.imag(eigs)
        det = np.linalg.det(J)
        tr = np.trace(J)
        if det < 0:
            tipo = "Punto Silla"
        elif abs(det) < 1e-10:
            tipo = "No aislado"
        elif all(abs(i)<1e-8 for i in im_p):
            if all(r<0 for r in re_p): tipo = "Nodo Estable (Pozo)"
            elif all(r>0 for r in re_p): tipo = "Nodo Inestable (Fuente)"
            else: tipo = "Punto Silla"
        else:
            if all(r<-1e-8 for r in re_p): tipo = "Foco Estable"
            elif all(r>1e-8 for r in re_p): tipo = "Foco Inestable"
            else: tipo = "Centro"
        return tipo, eigs, J, tr, det
    except Exception:
        return "Indeterminado", np.array([0,0]), np.zeros((2,2)), 0, 0

def solve_1d(expr_str, params, x0, t_span, n_points=1000):
    f = parse_expr(expr_str, ["x"], params)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    try:
        sol = solve_ivp(lambda t,y: [f(y[0])], t_span, [x0], method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.01)
        return sol.t, sol.y[0]
    except Exception:
        return t_eval, np.full_like(t_eval, np.nan)

def solve_2d(fx_str, fy_str, params, x0, y0, t_span, n_points=2000):
    fx = parse_expr(fx_str, ["x", "y"], params)
    fy = parse_expr(fy_str, ["x", "y"], params)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    try:
        sol = solve_ivp(lambda t,y: [fx(y[0],y[1]), fy(y[0],y[1])],
                        t_span, [x0, y0], method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.01)
        return sol.t, sol.y[0], sol.y[1]
    except Exception:
        return t_eval, np.full_like(t_eval, np.nan), np.full_like(t_eval, np.nan)

def solve_3d(fx_str, fy_str, fz_str, params, x0, y0, z0, t_span, n_points=10000):
    fx = parse_expr(fx_str, ["x", "y", "z"], params)
    fy = parse_expr(fy_str, ["x", "y", "z"], params)
    fz = parse_expr(fz_str, ["x", "y", "z"], params)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    try:
        sol = solve_ivp(lambda t,y: [fx(y[0],y[1],y[2]),fy(y[0],y[1],y[2]),fz(y[0],y[1],y[2])],
                        t_span, [x0, y0, z0], method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, max_step=0.005)
        return sol.t, sol.y[0], sol.y[1], sol.y[2]
    except Exception:
        return t_eval, np.full_like(t_eval,np.nan), np.full_like(t_eval,np.nan), np.full_like(t_eval,np.nan)

def compute_vector_field(fx_str, fy_str, params, x_range, y_range, density=20):
    fx = parse_expr(fx_str, ["x", "y"], params)
    fy = parse_expr(fy_str, ["x", "y"], params)
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(density):
        for j in range(density):
            try:
                U[i,j] = fx(X[i,j], Y[i,j])
                V[i,j] = fy(X[i,j], Y[i,j])
            except Exception:
                pass
    return X, Y, U, V

def compute_bifurcation_diagram(expr_str, param_name, param_range, other_params, x_range=(-5,5)):
    param_vals = np.linspace(param_range[0], param_range[1], 300)
    s_x, s_p, u_x, u_p = [], [], [], []
    for pval in param_vals:
        cp = {**other_params, param_name: pval}
        eqs = find_equilibria_1d(expr_str, cp, x_range, 100)
        for eq in eqs:
            cls, _ = classify_equilibrium_1d(expr_str, cp, eq)
            if "Estable" in cls or "Atractor" in cls:
                s_x.append(eq); s_p.append(pval)
            else:
                u_x.append(eq); u_p.append(pval)
    return s_p, s_x, u_p, u_x
