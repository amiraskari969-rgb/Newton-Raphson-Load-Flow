# Newton-Raphson-Load-Flow
A high-performance Cartesian Newton-Raphson load flow analysis core for power system transmission networks, accelerated using Cython for near-native C simulation speeds.
Load Flow Analysis using Cartesian Newton-Raphson Method (Cython Accelerated)
This project is an advanced high performance computational tool for solving non-linear power flow equations in power transmission networks. It uses Cartesian coordinate simulation framework and iterative Newton-Raphson (NR) numerical method.

The main feature of the project is the acceleration based on Cython. The pure Python code has been compiled into optimized C extensions (powerflow_core) to allow the heavy matrix calculations (building the Jacobian matrix, solving large-scale linear equations) to run at near-native C speeds. 📐 Mathematical Formulation & Cartesian Formulation
This implementation differs from the conventional Polar method, which uses voltage magnitude and angle ( ∣V ∣ , δ ) . Instead, the power flow problem is solved in the Cartesian coordinate system, by using the real part ( e ) and imaginary part ( f ) of the bus voltages ( V = e + jf ) .

The equations of active and reactive power injected in the iteration loop are:

i​= ∑ k=1 n​[(e i​e k​+f i​f k​)G ik​+(f i​e k​−e i​f k​)B ik​]Q i​= ∑ k=1 n​[(f i​e k​−e i​f k​)G ik​−(e i​e k​+f i​f k​)B ik​]
Where G and B are conductance and susceptance matrices obtained by decoupling the network Admittance Matrix (Ybus = G+jB).

Cartesian matrix Jacobian
In each iteration , voltage updates are performed by evaluating the mismatch vectors and solving the Cartesian Jacobian matrix (J):

[ ΔP ΔQ​]=[J][ Δe Δf​]
🛠️ powerflow_core Inputs & Data Structure
The core algorithm takes network data as an array of Python dictionaries:

Bus Data (bus_data) Contains bus number (num), bus type (type, support SLB: Swing/Slack bus, PQ: Load bus), active/reactive load demands (P_load, Q_load).

Line Data (line_data) Parameters of transmission line including starting bus (from), ending bus (to), series line impedance (Z), line shunt admittance (Y_sh) for πline representation.

📦 Complete Outputs
Once the Cython core has converged to the desired threshold (tol), it returns a structured database with:

Voltage Profile: Real parts (e), imaginary parts (f), voltage magnitudes calculated (V_mag) and phase angles in degrees (V_angle_deg).

Power Profile: Net active and reactive power injection at each bus (P_injected, Q_injected).

Convergence Behavior: A list of mismatch errors per iteration (errors) which is ideal for plotting convergence graphs, the success flag (converged) and the total number of iterations executed (actual_iter).

⚠️ Built-in validation and error handling
The module has strong engineering and mathematical sanity checks:

Zero Impedance Check: If an impedance of a line (Z) is set to zero, the core throws a ValueError with the exact line topology mapping.

Singular Matrix Guard: Low-level LinAlgError flags for diverge conditions that produce a singular or non-invertible Jacobian matrix are neatly trapped and re-raised as a crisp RuntimeError with the message "Jacobian is singular!".
