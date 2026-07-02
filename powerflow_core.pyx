# powerflow_core.pyx - Windows-compatible version
import numpy as np

def newton_raphson_cartesian_cython(bus_data, line_data, tol=1e-4, max_iter=200):
    n = len(bus_data)
    bus_types = [b['type'] for b in bus_data]
    slack_idx = bus_types.index('SLB')
    pq_indices = [i for i, t in enumerate(bus_types) if t == 'PQ']
    npq = len(pq_indices)
    
    e = np.ones(n)
    f = np.zeros(n)
    e[slack_idx] = 1.0
    f[slack_idx] = 0.0
    
    P_spec = np.zeros(n)
    Q_spec = np.zeros(n)
    for i, b in enumerate(bus_data):
        if b['type'] == 'PQ':
            P_spec[i] = -b['P_load']
            Q_spec[i] = -b['Q_load']
    
    Ybus = np.zeros((n, n), dtype=complex)
    for line in line_data:
        i = line['from'] - 1
        j = line['to'] - 1
        Z = line['Z']
        Y_sh = line['Y_sh']
        if Z == 0:
            raise ValueError(f"Line impedance Z cannot be zero (Line from {line['from']} to {line['to']})")
        Y_series = 1 / Z
        Y_shunt = Y_sh / 2
        Ybus[i, i] += Y_series + Y_shunt
        Ybus[j, j] += Y_series + Y_shunt
        Ybus[i, j] -= Y_series
        Ybus[j, i] -= Y_series
    
    G = Ybus.real
    B = Ybus.imag
    
    errors = []
    actual_iter = 0
    
    for it in range(max_iter):
        actual_iter = it + 1
        P_calc = np.zeros(n)
        Q_calc = np.zeros(n)
        for i in range(n):
            for k in range(n):
                P_calc[i] += (e[i]*e[k] + f[i]*f[k]) * G[i,k] + (f[i]*e[k] - e[i]*f[k]) * B[i,k]
                Q_calc[i] += (f[i]*e[k] - e[i]*f[k]) * G[i,k] - (e[i]*e[k] + f[i]*f[k]) * B[i,k]
        
        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc
        mismatch = np.concatenate([dP[pq_indices], dQ[pq_indices]])
        error = np.max(np.abs(mismatch))
        errors.append(error)
        
        if error < tol:
            break
        
        J = np.zeros((2*npq, 2*npq))
        for ii, i in enumerate(pq_indices):
            for jj, j in enumerate(pq_indices):
                if i == j:
                    J[ii, jj] = 2*e[i]*G[i,i] + 2*f[i]*B[i,i]
                    J[ii, jj + npq] = 2*f[i]*G[i,i] - 2*e[i]*B[i,i]
                    J[ii + npq, jj] = 2*f[i]*G[i,i] - 2*e[i]*B[i,i]
                    J[ii + npq, jj + npq] = -2*e[i]*G[i,i] - 2*f[i]*B[i,i]
                else:
                    J[ii, jj] = e[j]*G[i,j] + f[j]*B[i,j]
                    J[ii, jj + npq] = f[j]*G[i,j] - e[j]*B[i,j]
                    J[ii + npq, jj] = f[j]*G[i,j] - e[j]*B[i,j]
                    J[ii + npq, jj + npq] = -e[j]*G[i,j] - f[j]*B[i,j]
        
        try:
            delta_x = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            raise RuntimeError("Jacobian is singular!")
        
        de = delta_x[:npq]
        df = delta_x[npq:]
        for idx, bus in enumerate(pq_indices):
            e[bus] += de[idx]
            f[bus] += df[idx]
    else:
        raise RuntimeError(f"Did not converge within {max_iter} iterations!")
    
    V = e + 1j * f
    I = Ybus @ V
    S = V * np.conj(I)
    
    return {
        'bus_num': np.array([b['num'] for b in bus_data]),
        'type': np.array(bus_types),
        'e': e,
        'f': f,
        'V_mag': np.abs(V),
        'V_angle_deg': np.angle(V, deg=True),
        'P_injected': S.real,
        'Q_injected': S.imag,
        'P_load': np.array([b['P_load'] for b in bus_data]),
        'Q_load': np.array([b['Q_load'] for b in bus_data]),
        'errors': errors,
        'converged': True,
        'actual_iter': actual_iter,
        'tol': tol
    }