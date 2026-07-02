import customtkinter as ctk
from tkinter import messagebox, filedialog, StringVar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import json
import os
import pickle
from datetime import datetime

# Optional: networkx for graph visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️  networkx not installed. Install with: pip install networkx")

# ----------------------------
# Try to import Cython core (with fallback)
# ----------------------------
try:
    from powerflow_core import newton_raphson_cartesian_cython as newton_raphson_cartesian
    print("✅ Using Cython-accelerated power flow solver")
except ImportError:
    print("⚠️  Cython module not found. Using pure Python fallback.")
    def build_Ybus(bus_data, line_data):
        n = len(bus_data)
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
        return Ybus

    def newton_raphson_cartesian(bus_data, line_data, tol=1e-4, max_iter=200):
        bus_df = pd.DataFrame(bus_data)
        n = len(bus_df)
        slack_idx = bus_df[bus_df['type'] == 'SLB'].index[0]
        pq_indices = bus_df[bus_df['type'] == 'PQ'].index.tolist()
        npq = len(pq_indices)
        
        e = np.ones(n)
        f = np.zeros(n)
        e[slack_idx] = 1.0
        f[slack_idx] = 0.0
        
        P_spec = np.zeros(n)
        Q_spec = np.zeros(n)
        for i, row in bus_df.iterrows():
            if row['type'] == 'PQ':
                P_spec[i] = -row['P_load']
                Q_spec[i] = -row['Q_load']
        
        Ybus = build_Ybus(bus_data, line_data)
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
            'bus_num': bus_df['num'].values,
            'type': bus_df['type'].values,
            'e': e,
            'f': f,
            'V_mag': np.abs(V),
            'V_angle_deg': np.angle(V, deg=True),
            'P_injected': S.real,
            'Q_injected': S.imag,
            'P_load': bus_df['P_load'].values,
            'Q_load': bus_df['Q_load'].values,
            'errors': errors,
            'converged': True,
            'actual_iter': actual_iter,
            'tol': tol
        }

class SmoothShadowFrame(ctk.CTkFrame):
    def __init__(self, parent, width=400, height=300, corner_radius=14, **kwargs):
        super().__init__(parent, fg_color="transparent")
        mode = ctk.get_appearance_mode()
        shadow_color = "#1A1A1A" if mode == "Dark" else "#D0D0D0"
        card_color = kwargs.get("fg_color", "#2C2C2E" if mode == "Dark" else "#FFFFFF")
        
        # Shadow
        self.shadow = ctk.CTkFrame(
            self, width=width, height=height,
            corner_radius=corner_radius, fg_color=shadow_color, border_width=0
        )
        self.shadow.place(x=4, y=4)
        
        # Content — this is what you add children TO
        self.content = ctk.CTkFrame(
            self, width=width, height=height,
            corner_radius=corner_radius, fg_color=card_color, border_width=0
        )
        self.content.place(x=0, y=0)

MACOS_COLORS = {
    "light": {"bg": "#F0F0F4", "card": "#FFFFFF", "text": "#1D1D1F", "accent": "#0A84FF", "success": "#30D158", "danger": "#FF453A", "border": "#D1D1D6", "input_bg": "#F9F9FA"},
    "dark": {"bg": "#1C1C1E", "card": "#2C2C2E", "text": "#F5F5F7", "accent": "#0A84FF", "success": "#30D158", "danger": "#FF453A", "border": "#424246", "input_bg": "#3A3A3C"},
}

class HoverButton(ctk.CTkButton):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.default_color = kwargs.get("fg_color", "transparent")
        self.hover_color = kwargs.get("hover_color", "#5A5A5A")
    def on_enter(self, event): self.configure(fg_color=self.hover_color)
    def on_leave(self, event): self.configure(fg_color=self.default_color)

class PowerFlowApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("⚡ PowerFlow Pro")
        self.geometry("1100x800")
        self.minsize(900, 700)
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        self.colors = MACOS_COLORS["light"] if ctk.get_appearance_mode() == "Light" else MACOS_COLORS["dark"]
        self.configure(fg_color=self.colors["bg"])
        
        # Recent cases storage
        self.recent_cases = []
        self.load_recent_cases()
        
        self.bus_widgets = []
        self.line_widgets = []
        self.create_widgets()

    def create_widgets(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # === CONTROL CARD ===
        ctrl_card = SmoothShadowFrame(main, fg_color=self.colors["card"], height=60)
        ctrl_frame = ctrl_card.content
        ctrl_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(ctrl_frame, text="Tolerance:", font=("Segoe UI", 13)).pack(side="left", padx=(10,5))
        self.tol = StringVar(value="0.0001")
        ctk.CTkEntry(ctrl_frame, textvariable=self.tol, width=90).pack(side="left", padx=5)
        ctk.CTkLabel(ctrl_frame, text="Max Iter:", font=("Segoe UI", 13)).pack(side="left", padx=(15,5))
        self.max_iter = StringVar(value="100")
        ctk.CTkEntry(ctrl_frame, textvariable=self.max_iter, width=70).pack(side="left", padx=5)
        
        ctrl_card.pack(fill="x", pady=(0,15))

        # === DATA SECTION ===
        data_frame = ctk.CTkFrame(main, fg_color="transparent")
        data_frame.pack(fill="both", expand=True, pady=(0,15))
        data_frame.grid_columnconfigure(0, weight=1)
        data_frame.grid_columnconfigure(1, weight=1)
        data_frame.grid_rowconfigure(0, weight=1)

        # Bus Section
        bus_card = SmoothShadowFrame(data_frame, fg_color=self.colors["card"])
        bus_content = bus_card.content
        bus_content.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(bus_content, text="🚌 Bus Data", font=("Segoe UI", 16, "bold"), anchor="w").pack(anchor="w", padx=5, pady=(0,10))
        self.setup_scroll_container(bus_content, "bus")
        bus_card.grid(row=0, column=0, sticky="nsew", padx=(0,10))

        # Line Section
        line_card = SmoothShadowFrame(data_frame, fg_color=self.colors["card"])
        line_content = line_card.content
        line_content.pack(fill="both", expand=True, padx=5, pady=5)
        ctk.CTkLabel(line_content, text="🔌 Line Data", font=("Segoe UI", 16, "bold"), anchor="w").pack(anchor="w", padx=5, pady=(0,5))
        self.setup_scroll_container(line_content, "line")
        line_card.grid(row=0, column=1, sticky="nsew", padx=(5,0))

        # === BUTTONS ===
        button_frame = ctk.CTkFrame(main, fg_color="transparent")
        button_frame.pack(fill="x", pady=10)
        
        self.run_btn = HoverButton(
            button_frame, text="▶️ Run Power Flow", 
            command=self.run_power_flow,
            fg_color=self.colors["accent"],
            hover_color="#0865C2",
            text_color="white",
            height=45,
            font=("Segoe UI", 14, "bold")
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        HoverButton(
            button_frame, text="💾 Save Case", 
            command=self.save_case,
            fg_color="#30D158",
            hover_color="#25A844",
            text_color="white",
            height=45,
            font=("Segoe UI", 14, "bold")
        ).pack(side="left", fill="x", expand=True, padx=(5, 5))
        
        HoverButton(
            button_frame, text="📥 Import CSV", 
            command=self.import_from_csv,
            fg_color="#FF9500",
            hover_color="#E08500",
            text_color="white",
            height=45,
            font=("Segoe UI", 14, "bold")
        ).pack(side="left", fill="x", expand=True, padx=(5, 5))
        
        HoverButton(
            button_frame, text="📥 Import as Load JSON File", 
            command=self.load_case,
            fg_color="#FF2D55",
            hover_color="#D62246",
            text_color="white",
            height=45,
            font=("Segoe UI", 14, "bold")
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))

    
    def setup_scroll_container(self, parent, typ):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="both", expand=True)
        
        canvas = ctk.CTkCanvas(container, bg=self.colors["card"], highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(container, command=canvas.yview)
        scrollable = ctk.CTkFrame(canvas, fg_color="transparent")
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # >>>> FIXED: Bind mouse wheel only when hovering over this canvas <<<<
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-1*(ev.delta/120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        setattr(self, f"{typ}_canvas", canvas)
        setattr(self, f"{typ}_scrollable", scrollable)
        
        add_btn = HoverButton(
            container,
            text=f"➕ Add {typ.capitalize()}",
            command=lambda: self.add_bus_row() if typ == "bus" else self.add_line_row(),
            fg_color=self.colors["accent"],
            hover_color="#0865C2",
            text_color="white",
            height=30,
            font=("Segoe UI", 12)
        )
        add_btn.pack(pady=(5, 0))

    def update_scroll_region(self, typ):
        canvas = getattr(self, f"{typ}_canvas")
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def add_bus_row(self):
        i = len(self.bus_widgets)+1
        frame = ctk.CTkFrame(self.bus_scrollable, fg_color=self.colors["input_bg"], corner_radius=10, border_width=1, border_color=self.colors["border"])
        frame.pack(pady=(6, 8), padx=5, fill="x")
        label = ctk.CTkLabel(frame, text=f"Bus {i}", font=("Segoe UI", 13, "bold"))
        typ = ctk.CTkComboBox(frame, values=["SLB","PQ"], width=80, font=("Segoe UI",12))
        typ.set("PQ")
        p = ctk.CTkEntry(frame, placeholder_text="P_load", font=("Segoe UI",12), height=30)
        q = ctk.CTkEntry(frame, placeholder_text="Q_load", font=("Segoe UI",12), height=30)
        btn = HoverButton(frame, text="✕", width=30, height=30, command=lambda: self.del_row(frame,"bus"), fg_color=self.colors["danger"], hover_color="#C83A22", text_color="white")
        label.grid(row=0,column=0,padx=(10,5),pady=6,sticky="w")
        typ.grid(row=0,column=1,padx=2,pady=6,sticky="ew")
        p.grid(row=0,column=2,padx=2,pady=6,sticky="ew")
        q.grid(row=0,column=3,padx=2,pady=6,sticky="ew")
        btn.grid(row=0,column=4,padx=(2,10),pady=6,sticky="e")
        frame.grid_columnconfigure((1,2,3), weight=1)
        self.bus_widgets.append((frame,typ,p,q))
        self.update_delete_btns(self.bus_widgets, "bus")
        self.update_scroll_region("bus")

    def add_line_row(self):
        i = len(self.line_widgets)+1
        frame = ctk.CTkFrame(self.line_scrollable, fg_color=self.colors["input_bg"], corner_radius=10, border_width=1, border_color=self.colors["border"])
        frame.pack(pady=(4, 6), padx=5, fill="x")
        fr = ctk.CTkEntry(frame, placeholder_text="From", font=("Segoe UI",8), height=30,width=60)
        to = ctk.CTkEntry(frame, placeholder_text="To", font=("Segoe UI",8), height=30,width=60)
        r = ctk.CTkEntry(frame, placeholder_text="R", font=("Segoe UI",8), height=30,width=60)
        x = ctk.CTkEntry(frame, placeholder_text="X", font=("Segoe UI",8), height=30,width=60)
        ysh = ctk.CTkEntry(frame, placeholder_text="Y_sh", font=("Segoe UI",8), height=30,width=60)
        btn = HoverButton(frame, text="✕", width=15, height=30, command=lambda: self.del_row(frame,"line"), fg_color=self.colors["danger"], hover_color="#C83A22", text_color="white")
        fr.grid(row=0,column=0,padx=2,pady=6,sticky="ew")
        to.grid(row=0,column=1,padx=2,pady=6,sticky="ew")
        r.grid(row=0,column=2,padx=2,pady=6,sticky="ew")
        x.grid(row=0,column=3,padx=2,pady=6,sticky="ew")
        ysh.grid(row=0,column=4,padx=2,pady=6,sticky="ew")
        btn.grid(row=0,column=5,padx=(2,10),pady=6,sticky="e")
        frame.grid_columnconfigure((0,1,2,3,4), weight=1)
        self.line_widgets.append((frame,fr,to,r,x,ysh))
        self.update_delete_btns(self.line_widgets, "line")
        self.update_scroll_region("line")

    def del_row(self, frame, typ):
        widgets = self.bus_widgets if typ=="bus" else self.line_widgets
        if len(widgets) <= 1:
            messagebox.showwarning("Warning", f"At least one {typ} is required!")
            return
        widgets[:] = [w for w in widgets if w[0] != frame]
        frame.destroy()
        self.update_numbers(typ)
        self.update_delete_btns(widgets, typ)
        self.update_scroll_region(typ)

    def update_numbers(self, typ):
        widgets = self.bus_widgets if typ=="bus" else self.line_widgets
        for i, (frame, *_) in enumerate(widgets):
            label = f"{typ.capitalize()} {i+1}"
            for w in frame.winfo_children():
                if isinstance(w, ctk.CTkLabel) and w.cget("text").startswith(typ.capitalize()):
                    w.configure(text=label)
                    break

    def update_delete_btns(self, widgets, typ):
        for i, (frame, *_) in enumerate(widgets):
            for w in frame.winfo_children():
                if isinstance(w, HoverButton) and w.cget("text")=="✕":
                    if i == len(widgets)-1:
                        w.configure(state="normal", fg_color=self.colors["danger"])
                    else:
                        w.configure(state="disabled", fg_color="#666666")
                    break

    def run_power_flow(self):
        try:
            tol = float(self.tol.get())
            max_iter = int(self.max_iter.get())
            if tol <= 0 or max_iter <= 0:
                raise ValueError("Tol and Max Iter must be positive!")

            bus_data = []
            for i, (_, typ, p, q) in enumerate(self.bus_widgets):
                t = typ.get()
                p_val = float(p.get() or "0")
                q_val = float(q.get() or "0")
                bus_data.append({'num': i+1, 'type': t, 'P_load': p_val, 'Q_load': q_val})

            if sum(1 for b in bus_data if b['type']=='SLB') != 1:
                messagebox.showerror("Error", "Exactly one SLB bus required!")
                return

            line_data = []
            nb = len(bus_data)
            for (_, fr, to, r, x, ysh) in self.line_widgets:
                fr_val = int(fr.get())
                to_val = int(to.get())
                if not (1 <= fr_val <= nb and 1 <= to_val <= nb):
                    raise ValueError(f"Bus numbers must be 1-{nb}")
                R = float(r.get() or "0")
                X = float(x.get() or "0")
                Z = complex(R, X)
                if Z == 0:
                    raise ValueError("Z cannot be zero")
                Y_sh = complex(0, float(ysh.get() or "0"))
                line_data.append({'from': fr_val, 'to': to_val, 'Z': Z, 'Y_sh': Y_sh})

            results = newton_raphson_cartesian(bus_data, line_data, tol, max_iter)
            
            # Save to recent cases
            case_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'tol': tol,
                'max_iter': max_iter,
                'bus_data': bus_data,
                'line_data': line_data,
                'results': results
            }
            self.recent_cases.insert(0, case_data)
            self.recent_cases = self.recent_cases[:5]  # Keep only last 5
            self.save_recent_cases()
            
            self.show_results(results)

        except Exception as e:
            messagebox.showerror("Error", f"Failed:\n{str(e)}")

    def save_case(self):
        try:
            bus_data = []
            for i, (_, typ, p, q) in enumerate(self.bus_widgets):
                bus_data.append({
                    'num': i+1,
                    'type': typ.get(),
                    'P_load': float(p.get() or "0"),
                    'Q_load': float(q.get() or "0")
                })
            line_data = []
            for (_, fr, to, r, x, ysh) in self.line_widgets:
                line_data.append({
                    'from': int(fr.get()),
                    'to': int(to.get()),
                    'R': float(r.get() or "0"),
                    'X': float(x.get() or "0"),
                    'Y_sh': float(ysh.get() or "0")
                })
            case = {
                'tol': self.tol.get(),
                'max_iter': self.max_iter.get(),
                'bus_data': bus_data,
                'line_data': line_data
            }
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Save Power Flow Case"
            )
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(case, f, indent=2)
                messagebox.showinfo("Saved", f"Case saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_case(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
                title="Load Power Flow Case"
            )
            if not file_path or not os.path.exists(file_path):
                return
            with open(file_path, 'r') as f:
                case = json.load(f)
            
            # Clear current data
            while len(self.bus_widgets) > 1:
                self.del_row(self.bus_widgets[0][0], "bus")
            while len(self.line_widgets) > 1:
                self.del_row(self.line_widgets[0][0], "line")
            
            # Load settings
            self.tol.set(str(case.get('tol', '0.0001')))
            self.max_iter.set(str(case.get('max_iter', '100')))
            
            # Load buses
            for b in case['bus_data']:
                if len(self.bus_widgets) == 0:
                    self.add_bus_row()
                frame, typ, p, q = self.bus_widgets[-1]
                typ.set(b['type'])
                p.delete(0, 'end')
                p.insert(0, str(b['P_load']))
                q.delete(0, 'end')
                q.insert(0, str(b['Q_load']))
                if len(self.bus_widgets) < len(case['bus_data']):
                    self.add_bus_row()
            
            # Load lines
            for l in case['line_data']:
                if len(self.line_widgets) == 0:
                    self.add_line_row()
                frame, fr, to, r, x, ysh = self.line_widgets[-1]
                fr.delete(0, 'end'); fr.insert(0, str(l['from']))
                to.delete(0, 'end'); to.insert(0, str(l['to']))
                r.delete(0, 'end'); r.insert(0, str(l['R']))
                x.delete(0, 'end'); x.insert(0, str(l['X']))
                ysh.delete(0, 'end'); ysh.insert(0, str(l['Y_sh']))
                if len(self.line_widgets) < len(case['line_data']):
                    self.add_line_row()
                    
            messagebox.showinfo("Loaded", f"Case loaded from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))



    def import_from_csv(self):
        """Import bus and line data from a CSV file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv")],
                title="Import Power Flow Data from CSV"
            )
            if not file_path or not os.path.exists(file_path):
                return

            df = pd.read_csv(file_path)
            
            # Clear current data
            while len(self.bus_widgets) > 1:
                self.del_row(self.bus_widgets[0][0], "bus")
            while len(self.line_widgets) > 1:
                self.del_row(self.line_widgets[0][0], "line")
            
            # Detect data type by columns
            if 'Bus' in df.columns:
                # Bus data format
                for _, row in df.iterrows():
                    if len(self.bus_widgets) == 0:
                        self.add_bus_row()
                    frame, typ, p, q = self.bus_widgets[-1]
                    typ.set(row.get('Type', 'PQ'))
                    p.delete(0, 'end')
                    p.insert(0, str(row.get('P_load', 0)))
                    q.delete(0, 'end')
                    q.insert(0, str(row.get('Q_load', 0)))
                    if len(self.bus_widgets) < len(df):
                        self.add_bus_row()
                        
            elif 'From' in df.columns:
                # Line data format
                for _, row in df.iterrows():
                    if len(self.line_widgets) == 0:
                        self.add_line_row()
                    frame, fr, to, r, x, ysh = self.line_widgets[-1]
                    fr.delete(0, 'end'); fr.insert(0, str(int(row.get('From', 1))))
                    to.delete(0, 'end'); to.insert(0, str(int(row.get('To', 2))))
                    r.delete(0, 'end'); r.insert(0, str(row.get('R', 0)))
                    x.delete(0, 'end'); x.insert(0, str(row.get('X', 0)))
                    ysh.delete(0, 'end'); ysh.insert(0, str(row.get('Y_sh', 0)))
                    if len(self.line_widgets) < len(df):
                        self.add_line_row()
            else:
                messagebox.showerror("Import Error", "CSV must contain either 'Bus' (for bus data) or 'From' (for line data) column!")
                return
                
            messagebox.showinfo("Imported", f"Data imported from:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import CSV:\n{str(e)}")

    def show_results(self, results):
        win = ctk.CTkToplevel(self)
        win.title("📊 Results")
        win.geometry("1000x750")
        win.configure(fg_color=self.colors["bg"])
        
        ctk.CTkLabel(win, text="✅ Power Flow Complete", font=("Segoe UI", 20, "bold")).pack(pady=10)
        ctk.CTkLabel(win, text=f"Converged in {results['actual_iter']} iterations (tol={results['tol']:.1e})", text_color=self.colors["success"]).pack()

        # Export buttons
        export_frame = ctk.CTkFrame(win, fg_color="transparent")
        export_frame.pack(fill="x", padx=20, pady=(0,10))
        
        HoverButton(
            export_frame, text="📄 Export PDF Report", 
            command=lambda: self.export_to_pdf(results),
            fg_color="#FF9500",
            hover_color="#E08500",
            text_color="white",
            height=35,
            font=("Segoe UI", 12)
        ).pack(side="left", padx=5)
        
        HoverButton(
            export_frame, text="📊 Export CSV", 
            command=lambda: self.export_to_csv(results),
            fg_color="#30D158",
            hover_color="#25A844",
            text_color="white",
            height=35,
            font=("Segoe UI", 12)
        ).pack(side="left", padx=5)

        tab_frame = ctk.CTkFrame(win, fg_color="transparent")
        tab_frame.pack(fill="x", padx=20, pady=(0,10))
        self.result_tabs = {}
        tabs = ["Table", "Voltage Plot", "Convergence", "Network"]
        if not HAS_NETWORKX:
            tabs.remove("Network")
        for name in tabs:
            btn = ctk.CTkButton(
                tab_frame, text=name, width=120,
                command=lambda n=name: self.switch_result_tab(n, content_frame, results),
                fg_color="transparent", border_width=1, border_color=self.colors["border"]
            )
            btn.pack(side="left", padx=5)
            self.result_tabs[name] = btn

        content_frame = ctk.CTkFrame(win, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0,20))
        
        self.switch_result_tab(tabs[0], content_frame, results)

        ctk.CTkButton(win, text="CloseOperation", command=win.destroy, fg_color=self.colors["accent"], text_color="white").pack(pady=10)

    def switch_result_tab(self, tab, parent, results):
        for widget in parent.winfo_children():
            widget.destroy()

        if tab == "Table":
            self.show_table(parent, results)
        elif tab == "Voltage Plot":
            self.show_voltage_plot(parent, results)
        elif tab == "Convergence":
            self.show_convergence_plot(parent, results)
        elif tab == "Network":
            self.show_network_graph(parent, results)

        for name, btn in self.result_tabs.items():
            if name == tab:
                btn.configure(fg_color=self.colors["accent"], text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=self.colors["text"])

    def show_table(self, parent, results):
        table_frame = ctk.CTkFrame(parent, fg_color=self.colors["card"], corner_radius=12)
        table_frame.pack(fill="both", expand=True)
        headers = ["Bus","Type","E","F","|V| (p.u.)","∠V (°)"]
        for i, h in enumerate(headers):
            ctk.CTkLabel(table_frame, text=h, font=("Segoe UI",12,"bold"), fg_color=self.colors["accent"], text_color="white", height=30, width=90 if i<2 else 110).grid(row=0,column=i,padx=1,pady=1)
        for row_idx in range(len(results['bus_num'])):
            bg = "transparent" if row_idx % 2 == 0 else self.colors["input_bg"]
            data = [
                str(results['bus_num'][row_idx]),
                results['type'][row_idx],
                f"{results['e'][row_idx]:.5f}",
                f"{results['f'][row_idx]:.5f}",
                f"{results['V_mag'][row_idx]:.5f}",
                f"{results['V_angle_deg'][row_idx]:.2f}"
            ]
            for col, val in enumerate(data):
                ctk.CTkLabel(table_frame, text=val, fg_color=bg, height=28, width=90 if col<2 else 110).grid(row=row_idx+1,column=col,padx=1,pady=1)

    def show_voltage_plot(self, parent, results):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        buses = results['bus_num']
        v_mag = results['V_mag']
        ax.plot(buses, v_mag, 'o-', color=self.colors["accent"], linewidth=2, markersize=6)
        ax.set_xlabel("Bus Number")
        ax.set_ylabel("Voltage Magnitude (p.u.)")
        ax.set_title("Voltage Profile")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0.8, 1.2)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_convergence_plot(self, parent, results):
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        iterations = list(range(1, len(results['errors']) + 1))
        ax.semilogy(iterations, results['errors'], 'o-', color=self.colors["success"], linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Max Mismatch (log scale)")
        ax.set_title("Convergence Behavior")
        ax.grid(True, linestyle='--', alpha=0.6)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_network_graph(self, parent, results):
        if not HAS_NETWORKX:
            ctk.CTkLabel(parent, text="⚠️ networkx not installed.\nRun: pip install networkx", text_color="red").pack()
            return

        # Build graph
        G = nx.DiGraph()  # Use directed graph for flow direction
        bus_types = dict(zip(results['bus_num'], results['type']))
        voltages = dict(zip(results['bus_num'], results['V_mag']))
        
        # Add nodes
        for bus in results['bus_num']:
            G.add_node(bus, type=bus_types[bus], voltage=voltages[bus])
        
        # Add edges with weights and direction
        edge_weights = []
        for (_, fr, to, r, x, ysh) in self.line_widgets:
            try:
                f = int(fr.get())
                t = int(to.get())
                # Calculate line weight (inverse of impedance magnitude)
                Z = complex(float(r.get() or "0"), float(x.get() or "0"))
                if Z != 0:
                    weight = 1 / abs(Z)
                else:
                    weight = 1.0
                edge_weights.append(weight)
                G.add_edge(f, t, weight=weight)
            except:
                pass

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Position nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Color nodes
        node_colors = []
        for node in G.nodes():
            if bus_types[node] == 'SLB':
                node_colors.append('#FF453A')  # Red for slack
            else:
                node_colors.append('#0A84FF')  # Blue for PQ
        
        # Draw edges with arrows and varying widths
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            widths = [2 + 5 * (w - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 3 for w in edge_weights]
        else:
            widths = [2] * len(G.edges())
            
        nx.draw_networkx_edges(
            G, pos, ax=ax, 
            arrows=True, 
            arrowsize=20,
            width=widths,
            alpha=0.7,
            edge_color='gray'
        )
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color="white")
        
        ax.set_title("Network Topology (with Flow Direction & Line Weights)")
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_to_pdf(self, results):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                title="Save Report as PDF"
            )
            if not file_path:
                return

            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph("Power Flow Analysis Report", title_style))
            story.append(Paragraph(f"Converged in {results['actual_iter']} iterations (tolerance: {results['tol']:.1e})", styles["Normal"]))
            story.append(Spacer(1, 20))

            data = [["Bus", "Type", "E", "F", "|V| (p.u.)", "∠V (°)"]]
            for i in range(len(results['bus_num'])):
                data.append([
                    str(results['bus_num'][i]),
                    results['type'][i],
                    f"{results['e'][i]:.5f}",
                    f"{results['f'][i]:.5f}",
                    f"{results['V_mag'][i]:.5f}",
                    f"{results['V_angle_deg'][i]:.2f}"
                ])

            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor(self.colors["accent"])),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor(self.colors["card"])),
            ]))
            story.append(table)

            # Voltage plot
            fig = Figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            ax.plot(results['bus_num'], results['V_mag'], 'o-', color='blue')
            ax.set_xlabel("Bus Number")
            ax.set_ylabel("Voltage Magnitude (p.u.)")
            ax.set_title("Voltage Profile")
            ax.grid(True)

            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            img = RLImage(img_buffer, width=400, height=200)
            story.append(Spacer(1, 20))
            story.append(img)

            doc.build(story)
            messagebox.showinfo("Success", f"Report saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PDF:\n{str(e)}")

    def export_to_csv(self, results):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Results as CSV"
            )
            if not file_path:
                return

            df = pd.DataFrame({
                'Bus': results['bus_num'],
                'Type': results['type'],
                'E': results['e'],
                'F': results['f'],
                'V_mag_pu': results['V_mag'],
                'V_angle_deg': results['V_angle_deg'],
                'P_injected': results['P_injected'],
                'Q_injected': results['Q_injected'],
                'P_load': results['P_load'],
                'Q_load': results['Q_load']
            })
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Data saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}")

    # Recent cases management
    def save_recent_cases(self):
        """Save recent cases to disk"""
        try:
            with open('recent_cases.pkl', 'wb') as f:
                pickle.dump(self.recent_cases, f)
        except Exception as e:
            print(f"Failed to save recent cases: {e}")

    def load_recent_cases(self):
        """Load recent cases from disk"""
        try:
            if os.path.exists('recent_cases.pkl'):
                with open('recent_cases.pkl', 'rb') as f:
                    self.recent_cases = pickle.load(f)
                    # Keep only last 5
                    self.recent_cases = self.recent_cases[:5]
        except Exception as e:
            print(f"Failed to load recent cases: {e}")
            self.recent_cases = []

    def export_recent_cases_to_csv(self):
        """Export all recent cases to a single CSV file"""
        if not self.recent_cases:
            messagebox.showinfo("Info", "No recent cases to export!")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Recent Cases as CSV"
            )
            if not file_path:
                return

            all_data = []
            for case in self.recent_cases:
                timestamp = case['timestamp']
                results = case['results']
                for i in range(len(results['bus_num'])):
                    all_data.append({
                        'Timestamp': timestamp,
                        'Bus': results['bus_num'][i],
                        'Type': results['type'][i],
                        'E': results['e'][i],
                        'F': results['f'][i],
                        'V_mag_pu': results['V_mag'][i],
                        'V_angle_deg': results['V_angle_deg'][i],
                        'P_injected': results['P_injected'][i],
                        'Q_injected': results['Q_injected'][i],
                        'P_load': results['P_load'][i],
                        'Q_load': results['Q_load'][i]
                    })
            
            df = pd.DataFrame(all_data)
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Recent cases saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export recent cases:\n{str(e)}")

if __name__ == "__main__":
    app = PowerFlowApp()
    app.mainloop()