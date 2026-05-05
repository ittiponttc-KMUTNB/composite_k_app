import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.optimize import brentq


"""
composite_k_nomograph.py
========================
AASHTO 1993 Figure 3.3 - Composite Modulus of Subgrade Reaction
แทน Nomograph ด้วย Digitized Data + 2D Interpolation

Input:
    E_eq  (psi)  - Equivalent Elastic Modulus of Subbase
    D_total (in) - Total Subbase Thickness
    M_R   (psi)  - Roadbed Soil Resilient Modulus

Output:
    k_inf (pci)  - Composite Modulus of Subgrade Reaction

Author: Composite K Module for Rigid Pavement Design
Reference: AASHTO Guide for Design of Pavement Structures 1993, Figure 3.3
"""

import numpy as np
from scipy.interpolate import RBFInterpolator, interp1d
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DIGITIZED DATA จาก AASHTO Figure 3.3
# =============================================================================
# Turning Line: M_R (psi) <-> D_SB (inches)
# อ่านจาก nomograph: เส้น Turning Line แนวทแยง
TURNING_LINE_DATA = {
    # M_R (psi) : D_SB (inches)  -- อ่านจากเส้น Turning Line
    "MR_psi":  [1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000,
                8000, 9000, 10000, 12000, 15000, 20000],
    "DSB_in":  [19.5, 17.2, 15.8, 13.5, 12.0, 10.8, 9.9,  9.2,
                8.6,  8.1,  7.7,  7.1,  6.4,  5.6]
}

# Calibration Data 3D จาก Nomograph จริง (อาจารย์อ่านค่าโดยตรง)
# 34 จุด ครอบคลุม E_SB, D_SB, M_R ทุกช่วง
CALIB_POINTS = [
    # (E_SB psi, D_SB in, M_R psi, k∞ pci)
    # D=6, M_R=4500
    (15000,   6,  4500,  220), (50000,   6,  4500,  300), (100000,  6,  4500,  350),
    # D=10, M_R=4500
    (15000,  10,  4500,  250), (50000,  10,  4500,  350), (100000, 10,  4500,  400),
    # D=14, M_R=4500
    (15000,  14,  4500,  300), (30000,  14,  4500,  350), (50000,  14,  4500,  420),
    (75000,  14,  4500,  450),
    # D=18, M_R=4500
    (15000,  18,  4500,  320), (100000, 18,  4500,  550), (200000, 18,  4500,  650),
    # D=20, M_R=4500
    (100000, 20,  4500,  250), (400000, 20,  4500,  990), (1000000,20,  4500, 1300),
    # D=14, M_R ต่างๆ
    (15000,  14,  3000,  250), (50000,  14,  3000,  350),
    (15000,  14,  7500,  460), (50000,  14,  7500,  600),
    (15000,  14, 15000,  760), (50000,  14, 15000, 1000),
    # D=6, M_R ต่างๆ
    (15000,   6,  3000,   80), (50000,   6,  3000,  160),
    (15000,   6,  7500,  390), (50000,   6,  7500,  450),
    (15000,   6, 15000,  600), (50000,   6, 15000,  800),
    # D=20, M_R ต่างๆ
    (15000,  20,  3000,  290), (100000, 20,  3000,  500),
    (15000,  20,  7500,  500), (100000, 20,  7500,  990),
    (15000,  20, 15000,  800), (100000, 20, 15000, 1400),
]

# =============================================================================
# วัสดุ Default (ชื่อภาษาไทย + E MPa)
# =============================================================================
DEFAULT_MATERIALS = {
    "รองผิวทางคอนกรีตด้วย PMA(AC)":               3700,
    "รองผิวทางคอนกรีตด้วย AC":                     2500,
    "หินคลุกปรับปรุงคุณภาพด้วยปูนซีเมนต์ (CTB)":  1200,
    "หินคลุกผสมซีเมนต์ UCS 24.5 ksc":              850,
    "ดินซีเมนต์ UCS 17.5 ksc":                      350,
    "รองพื้นทางวัสดุมวลรวม CBR 25%":               150,
    "วัสดุคัดเลือก ก":                              100,
}

# หน่วย Conversion
MPa_TO_PSI  = 145.038
CM_TO_INCH  = 0.393701

# =============================================================================
# CLASS: Turning Line  (M_R <-> D_SB)
# =============================================================================
class TurningLine:
    """แปลง M_R (psi) -> D_SB (inches) หรือกลับกัน ผ่าน Turning Line"""

    def __init__(self):
        mr  = np.array(TURNING_LINE_DATA["MR_psi"],  dtype=float)
        dsb = np.array(TURNING_LINE_DATA["DSB_in"],  dtype=float)

        log_mr  = np.log10(mr)
        log_dsb = np.log10(dsb)

        self._mr_to_dsb = interp1d(log_mr,  log_dsb, kind='linear',
                                    fill_value='extrapolate')
        self._dsb_to_mr = interp1d(log_dsb, log_mr,  kind='linear',
                                    fill_value='extrapolate')

    def mr_to_dsb(self, MR_psi: float) -> float:
        """M_R (psi) -> D_SB (inches)"""
        return float(10 ** self._mr_to_dsb(np.log10(MR_psi)))

    def dsb_to_mr(self, DSB_in: float) -> float:
        """D_SB (inches) -> M_R (psi)"""
        return float(10 ** self._dsb_to_mr(np.log10(DSB_in)))


# =============================================================================
# CLASS: Upper Chart  (E_SB, D_SB) -> k∞
# =============================================================================
class UpperChart:
    """
    2D Interpolation บน k∞ curve family
    ใช้ RBFInterpolator ใน log-log space
    """

    def __init__(self):
        # 3D Calibration: (log E, log D, log MR) -> log k∞
        pts  = np.array([[np.log10(e), np.log10(d), np.log10(mr)]
                          for e,d,mr,k in CALIB_POINTS])
        vals = np.array([np.log10(k) for e,d,mr,k in CALIB_POINTS])

        self._points = pts
        self._values = vals
        self._rbf = RBFInterpolator(pts, vals,
                                     kernel='thin_plate_spline', smoothing=0.0)

        # bounds
        self.E_min, self.E_max = 15000,  1000000
        self.D_min, self.D_max = 6.0,    20.0
        self.k_min, self.k_max = 50,     2000
        self.MR_min,self.MR_max= 3000,   15000

    def get_k(self, E_psi: float, D_in: float, MR_psi: float = 4500) -> float:
        """(E_SB psi, D_SB inches, M_R psi) -> k∞ (pci)"""
        MR_psi = np.clip(MR_psi, self.MR_min, self.MR_max)
        pt = np.array([[np.log10(E_psi), np.log10(D_in), np.log10(MR_psi)]])
        log_k = float(self._rbf(pt)[0])
        k = 10 ** log_k
        return float(np.clip(k, self.k_min, self.k_max))

    def get_D(self, E_psi: float, k_target: float,
              D_bounds=(6.0, 20.0), MR_psi: float = 4500) -> float:
        """
        กำหนด E_SB + k∞ target -> หา D_SB (inches)
        ใช้ bisection search
        """
        from scipy.optimize import brentq
        def obj(d):
            return self.get_k(E_psi, d, MR_psi) - k_target

        d_lo, d_hi = D_bounds
        try:
            k_lo = obj(d_lo)
            k_hi = obj(d_hi)
            if k_lo * k_hi > 0:
                # ไม่ตัดแกน: คืนค่าขอบที่ใกล้ที่สุด
                if abs(k_lo) < abs(k_hi):
                    return d_lo
                else:
                    return d_hi
            return float(brentq(obj, d_lo, d_hi, xtol=1e-4))
        except Exception:
            return float('nan')

    def get_E(self, D_in: float, k_target: float,
              E_bounds=(15000, 1000000), MR_psi: float = 4500) -> float:
        """
        กำหนด D_SB + k∞ target -> หา E_SB (psi)
        ใช้ bisection search
        """
        from scipy.optimize import brentq
        def obj(e):
            return self.get_k(e, D_in, MR_psi) - k_target

        e_lo, e_hi = E_bounds
        try:
            k_lo = obj(e_lo)
            k_hi = obj(e_hi)
            if k_lo * k_hi > 0:
                if abs(k_lo) < abs(k_hi):
                    return e_lo
                else:
                    return e_hi
            return float(brentq(obj, e_lo, e_hi, xtol=1.0))
        except Exception:
            return float('nan')


# =============================================================================
# CLASS: CompositeK  (Main Interface)
# =============================================================================
class CompositeK:
    """
    Main interface แทน AASHTO Figure 3.3

    Modes:
      A: {M_R, k∞_target, E_SB} -> D_SB
      B: {M_R, k∞_target, D_SB} -> E_SB
      C: {M_R, E_SB, D_SB}      -> k∞
      D: {E_SB, D_SB, k∞}       -> M_R
    """

    def __init__(self):
        self.turning = TurningLine()
        self.upper   = UpperChart()

    # ------------------------------------------------------------------
    # Mode A: M_R + k∞ + E_SB -> D_SB
    # ------------------------------------------------------------------
    def mode_A(self, MR_psi: float, k_target_pci: float,
               E_psi: float) -> dict:
        D_required_in = self.upper.get_D(E_psi, k_target_pci)
        D_required_cm = D_required_in / CM_TO_INCH
        D_from_turning = self.turning.mr_to_dsb(MR_psi)
        return {
            "mode": "A",
            "MR_psi":          MR_psi,
            "k_target_pci":    k_target_pci,
            "E_SB_psi":        E_psi,
            "D_required_in":   round(D_required_in, 3),
            "D_required_cm":   round(D_required_cm, 1),
            "D_turning_in":    round(D_from_turning, 3),
        }

    # ------------------------------------------------------------------
    # Mode B: M_R + k∞ + D_SB -> E_SB
    # ------------------------------------------------------------------
    def mode_B(self, MR_psi: float, k_target_pci: float,
               D_in: float) -> dict:
        E_required_psi = self.upper.get_E(D_in, k_target_pci)
        E_required_MPa = E_required_psi / MPa_TO_PSI
        return {
            "mode": "B",
            "MR_psi":          MR_psi,
            "k_target_pci":    k_target_pci,
            "D_SB_in":         D_in,
            "E_required_psi":  round(E_required_psi, 0),
            "E_required_MPa":  round(E_required_MPa, 1),
        }

    # ------------------------------------------------------------------
    # Mode C: M_R + E_SB + D_SB -> k∞
    # ------------------------------------------------------------------
    def mode_C(self, MR_psi: float, E_psi: float,
               D_in: float) -> dict:
        k_actual = self.upper.get_k(E_psi, D_in, MR_psi)
        return {
            "mode": "C",
            "MR_psi":      MR_psi,
            "E_SB_psi":    E_psi,
            "D_SB_in":     D_in,
            "k_actual_pci": round(k_actual, 1),
        }

    # ------------------------------------------------------------------
    # Mode D: E_SB + D_SB + k∞ -> M_R
    # ------------------------------------------------------------------
    def mode_D(self, E_psi: float, D_in: float,
               k_pci: float) -> dict:
        MR = self.turning.dsb_to_mr(D_in)
        k_check = self.upper.get_k(E_psi, D_in)
        return {
            "mode": "D",
            "E_SB_psi":    E_psi,
            "D_SB_in":     D_in,
            "k_pci":       k_pci,
            "MR_psi":      round(MR, 0),
            "k_check_pci": round(k_check, 1),
        }

    # ------------------------------------------------------------------
    # Multi-layer: คำนวณ E_eq + D_total แล้วเข้า Mode C
    # ------------------------------------------------------------------
    def multilayer_k(self, layers: list, MR_psi: float) -> dict:
        """
        layers = [{"name": str, "E_MPa": float, "D_cm": float}, ...]
        คำนวณ E_eq (Odemark weighted) + D_total -> k∞
        """
        E_list = [L["E_MPa"] * MPa_TO_PSI for L in layers]
        D_list = [L["D_cm"]  * CM_TO_INCH  for L in layers]

        D_total_in = sum(D_list)
        D_total_cm = sum(L["D_cm"] for L in layers)

        # E_eq = [Σ(Di × Ei^(1/3)) / Σ(Di)]³
        num = sum(d * (e ** (1/3)) for d, e in zip(D_list, E_list))
        den = D_total_in
        E_eq_psi = (num / den) ** 3
        E_eq_MPa = E_eq_psi / MPa_TO_PSI

        k_actual = self.upper.get_k(E_eq_psi, D_total_in, MR_psi)

        return {
            "mode":         "multilayer",
            "MR_psi":       MR_psi,
            "layers":       layers,
            "D_total_in":   round(D_total_in, 3),
            "D_total_cm":   round(D_total_cm, 1),
            "E_eq_psi":     round(E_eq_psi, 0),
            "E_eq_MPa":     round(E_eq_MPa, 1),
            "k_actual_pci": round(k_actual, 1),
        }

    # ------------------------------------------------------------------
    # หา D_SB required สำหรับแต่ละวัสดุ (single layer)
    # ------------------------------------------------------------------
    def required_thickness_per_material(self, MR_psi: float,
                                         k_target_pci: float) -> list:
        """
        สำหรับวัสดุแต่ละชนิด (E ทราบ) หา D_SB ที่ต้องการ
        """
        results = []
        for name, E_MPa in DEFAULT_MATERIALS.items():
            E_psi = E_MPa * MPa_TO_PSI
            D_in  = self.upper.get_D(E_psi, k_target_pci, MR_psi=4500)
            D_cm  = D_in / CM_TO_INCH if not np.isnan(D_in) else float('nan')
            results.append({
                "material":   name,
                "E_MPa":      E_MPa,
                "E_psi":      round(E_psi, 0),
                "D_req_in":   round(D_in, 3) if not np.isnan(D_in) else None,
                "D_req_cm":   round(D_cm, 1) if not np.isnan(D_cm) else None,
            })
        return results


# =============================================================================
# HELPER: คำนวณ E_eq จากหลายชั้น
# =============================================================================
def calc_E_eq(layers: list) -> tuple:
    """
    layers = [{"E_MPa": float, "D_cm": float}, ...]
    return (E_eq_MPa, E_eq_psi, D_total_cm, D_total_in)
    """
    E_list = [L["E_MPa"] * MPa_TO_PSI for L in layers]
    D_list = [L["D_cm"]  * CM_TO_INCH  for L in layers]

    D_total_in = sum(D_list)
    D_total_cm = sum(L["D_cm"] for L in layers)

    if D_total_in == 0:
        return 0, 0, 0, 0

    num      = sum(d * (e ** (1/3)) for d, e in zip(D_list, E_list))
    E_eq_psi = (num / D_total_in) ** 3
    E_eq_MPa = E_eq_psi / MPa_TO_PSI

    return (round(E_eq_MPa, 1), round(E_eq_psi, 0),
            round(D_total_cm, 1), round(D_total_in, 3))


# =============================================================================
# SELF TEST
# =============================================================================


st.set_page_config(
    page_title="Composite k — AASHTO 1993",
    page_icon="🛣️",
    layout="wide"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {padding: 8px 20px; border-radius: 8px 8px 0 0;}
.metric-card {
    background: #f8f9fa; border: 1px solid #dee2e6;
    border-radius: 10px; padding: 14px 18px; text-align: center;
}
.metric-label {font-size: 13px; color: #6c757d; margin-bottom: 4px;}
.metric-value {font-size: 22px; font-weight: 600; color: #212529;}
.metric-unit  {font-size: 12px; color: #adb5bd; margin-top: 2px;}
.pass-badge {background:#d4edda; color:#155724; padding:3px 12px;
             border-radius:20px; font-size:13px; font-weight:500;}
.fail-badge {background:#f8d7da; color:#721c24; padding:3px 12px;
             border-radius:20px; font-size:13px; font-weight:500;}
.section-header {
    font-size: 15px; font-weight: 600; color: #495057;
    border-bottom: 2px solid #dee2e6; padding-bottom: 6px;
    margin-bottom: 14px; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialize
# =============================================================================
@st.cache_resource
def get_model():
    return CompositeK()

ck = get_model()

# =============================================================================
# Header
# =============================================================================
st.title("🛣️ Composite Modulus of Subgrade Reaction")
st.caption("AASHTO 1993 — Figure 3.3 | แทน Nomograph ด้วยสมการ Interpolation")
st.divider()

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "📐 ออกแบบชั้นทาง (หลายชั้น)",
    "📊 เปรียบเทียบวัสดุ (Single Layer)",
    "🔧 คำนวณ k∞ โดยตรง"
])

# =============================================================================
# TAB 1: Multi-layer Design
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">ข้อมูลดินเดิม (Subgrade)</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        cbr = st.number_input("CBR ดินคันทาง (%)", min_value=1.0,
                               max_value=100.0, value=3.0, step=0.5,
                               format="%.1f", key="t1_cbr")
    MR_psi = cbr * 1500
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">M<sub>R</sub> (คำนวณ)</div>
          <div class="metric-value">{MR_psi:,.0f}</div>
          <div class="metric-unit">psi = CBR × 1,500</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        k_target = st.number_input("k∞ เป้าหมาย (pci)", min_value=50,
                                    max_value=2000, value=200, step=50,
                                    key="t1_k")

    st.markdown('<div class="section-header">ชั้นวัสดุชั้นทาง</div>',
                unsafe_allow_html=True)

    # Dynamic layer input
    if "n_layers" not in st.session_state:
        st.session_state.n_layers = 2

    col_add, col_del, _ = st.columns([1, 1, 6])
    with col_add:
        if st.button("➕ เพิ่มชั้น", use_container_width=True):
            if st.session_state.n_layers < 5:
                st.session_state.n_layers += 1
    with col_del:
        if st.button("➖ ลบชั้น", use_container_width=True):
            if st.session_state.n_layers > 1:
                st.session_state.n_layers -= 1

    mat_names = list(DEFAULT_MATERIALS.keys())
    layers = []

    for i in range(st.session_state.n_layers):
        with st.expander(f"ชั้นที่ {i+1}", expanded=True):
            lc1, lc2, lc3 = st.columns([3, 2, 2])
            with lc1:
                mat = st.selectbox(
                    "ชนิดวัสดุ",
                    options=["(กำหนดเอง)"] + mat_names,
                    key=f"mat_{i}",
                    index=i % len(mat_names) + 1
                )
            with lc2:
                default_E = DEFAULT_MATERIALS.get(mat, 300)
                E_val = st.number_input(
                    "E (MPa)", min_value=10, max_value=5000,
                    value=default_E, step=50, key=f"E_{i}"
                )
            with lc3:
                D_val = st.number_input(
                    "ความหนา (cm)", min_value=1.0, max_value=100.0,
                    value=20.0 if i == 0 else 15.0,
                    step=1.0, format="%.1f", key=f"D_{i}"
                )

            # แสดง conversion
            st.caption(
                f"E = {E_val} MPa = {E_val * MPa_TO_PSI:,.0f} psi  |  "
                f"D = {D_val} cm = {D_val * CM_TO_INCH:.3f} นิ้ว"
            )
            layers.append({
                "name":  mat if mat != "(กำหนดเอง)" else f"ชั้นที่ {i+1}",
                "E_MPa": E_val,
                "D_cm":  D_val
            })

    st.divider()

    # คำนวณ
    if layers:
        E_eq_MPa, E_eq_psi, D_total_cm, D_total_in = calc_E_eq(layers)
        result = ck.multilayer_k(layers, MR_psi)
        k_actual = result["k_actual_pci"]
        passed   = k_actual >= k_target

        st.markdown('<div class="section-header">ผลการคำนวณ</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        cards = [
            (m1, "E<sub>eq</sub>", f"{E_eq_MPa:,.1f}", "MPa"),
            (m2, "E<sub>eq</sub>", f"{E_eq_psi:,.0f}", "psi"),
            (m3, "D<sub>total</sub>", f"{D_total_cm:.1f}", "cm"),
            (m4, "D<sub>total</sub>", f"{D_total_in:.3f}", "นิ้ว"),
            (m5, "k∞ ที่ได้จริง", f"{k_actual:.0f}", "pci"),
        ]
        for col, label, val, unit in cards:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{val}</div>
                  <div class="metric-unit">{unit}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pass/Fail
        badge = "pass-badge" if passed else "fail-badge"
        symbol = "✅" if passed else "❌"
        st.markdown(
            f'<div style="text-align:center; font-size:18px;">'
            f'<span class="{badge}">{symbol} k∞ actual = {k_actual:.0f} pci '
            f'{"≥" if passed else "<"} k∞ target = {k_target} pci — '
            f'{"ผ่าน" if passed else "ไม่ผ่าน"}</span></div>',
            unsafe_allow_html=True
        )

        # สรุปตาราง
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**สรุปชั้นวัสดุ**")
        import pandas as pd
        rows = []
        for idx, L in enumerate(layers):
            rows.append({
                "ชั้น": f"ชั้นที่ {idx+1}",
                "วัสดุ": L["name"],
                "E (MPa)": L["E_MPa"],
                "E (psi)": f"{L['E_MPa']*MPa_TO_PSI:,.0f}",
                "ความหนา (cm)": L["D_cm"],
                "ความหนา (นิ้ว)": f"{L['D_cm']*CM_TO_INCH:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        # กราฟ Layer
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#4C9BE8", "#5CB85C", "#F0AD4E", "#D9534F", "#9B59B6"]
        y = 0
        for idx, L in enumerate(layers):
            ax.barh(y=0, width=L["D_cm"], left=y,
                    color=colors[idx % len(colors)], edgecolor="white",
                    height=0.6, label=f"ชั้นที่ {idx+1}: {L['name'][:20]}")
            ax.text(y + L["D_cm"]/2, 0,
                    f"{L['D_cm']:.0f} cm\n{L['E_MPa']} MPa",
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')
            y += L["D_cm"]
        ax.set_xlabel("ความหนาสะสม (cm)", fontsize=11)
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title("โครงสร้างชั้นทาง", fontsize=13, fontweight='bold')
        ax.set_xlim(0, y * 1.1)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


# =============================================================================
# TAB 2: Single Layer Material Comparison
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">ข้อมูลดินเดิม (Subgrade)</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        cbr2 = st.number_input("CBR ดินคันทาง (%)", min_value=1.0,
                                max_value=100.0, value=3.0, step=0.5,
                                format="%.1f", key="t2_cbr")
    MR2 = cbr2 * 1500
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">M<sub>R</sub> (คำนวณ)</div>
          <div class="metric-value">{MR2:,.0f}</div>
          <div class="metric-unit">psi</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        k2 = st.number_input("k∞ เป้าหมาย (pci)", min_value=50,
                              max_value=2000, value=200, step=50, key="t2_k")

    st.markdown('<div class="section-header">ความหนาที่ต้องการ (Single Layer)</div>',
                unsafe_allow_html=True)

    results = ck.required_thickness_per_material(MR2, k2)

    # กราฟ E_SB vs D_SB
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    E_range = np.logspace(np.log10(15000), np.log10(1000000), 200)
    D_curve = [ck.upper.get_D(e, k2) for e in E_range]
    valid   = [(e, d) for e, d in zip(E_range, D_curve)
               if not np.isnan(d) and 0.5 <= d <= 20]
    if valid:
        ex, dx = zip(*valid)
        ax2.plot(ex, dx, 'b-', linewidth=2.5,
                 label=f'k∞ = {k2} pci (required curve)')

    colors2 = ["#E74C3C","#E67E22","#27AE60","#2980B9","#8E44AD","#16A085","#2C3E50"]
    for idx, r in enumerate(results):
        if r["D_req_in"] and r["E_psi"]:
            ax2.scatter(r["E_psi"], r["D_req_in"],
                        color=colors2[idx % len(colors2)],
                        s=120, zorder=5)
            ax2.annotate(
                f"{r['material'][:18]}\n{r['D_req_cm']:.1f} cm",
                xy=(r["E_psi"], r["D_req_in"]),
                xytext=(10, 8), textcoords='offset points',
                fontsize=8.5, color=colors2[idx % len(colors2)]
            )

    ax2.set_xscale('log')
    ax2.set_xlabel("E_SB (psi)", fontsize=12)
    ax2.set_ylabel("D_SB required (inches)", fontsize=12)
    ax2.set_title(f"E_SB vs D_SB Required  |  CBR={cbr2}%  M_R={MR2:,} psi  k∞={k2} pci",
                  fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=10)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ตาราง
    import pandas as pd
    df_rows = []
    for r in results:
        df_rows.append({
            "วัสดุ":           r["material"],
            "E (MPa)":         r["E_MPa"],
            "E (psi)":         f"{r['E_psi']:,}",
            "D required (นิ้ว)": f"{r['D_req_in']:.3f}" if r['D_req_in'] else "—",
            "D required (cm)": f"{r['D_req_cm']:.1f}" if r['D_req_cm'] else "—",
        })
    st.dataframe(pd.DataFrame(df_rows), use_container_width=True,
                 hide_index=True)


# =============================================================================
# TAB 3: Direct k∞ Calculation (4 Modes)
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">เลือก Mode การคำนวณ</div>',
                unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        options=[
            "A — กำหนด M_R + k∞ + E_SB → หา D_SB",
            "B — กำหนด M_R + k∞ + D_SB → หา E_SB",
            "C — กำหนด M_R + E_SB + D_SB → หา k∞",
            "D — กำหนด E_SB + D_SB + k∞ → หา M_R",
        ],
        horizontal=False,
        key="mode_sel"
    )
    mode_key = mode[0]

    st.markdown('<div class="section-header">Input</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)

    if mode_key == "A":
        with mc1:
            cbr3 = st.number_input("CBR (%)", 1.0, 100.0, 3.0, 0.5, key="m_cbr")
            MR3  = cbr3 * 1500
            st.caption(f"M_R = {MR3:,.0f} psi")
        with mc2:
            k3 = st.number_input("k∞ target (pci)", 50, 2000, 200, 50, key="m_k")
        with mc3:
            E3_MPa = st.number_input("E_SB (MPa)", 10, 5000, 850, 50, key="m_e")
            E3_psi = E3_MPa * MPa_TO_PSI
            st.caption(f"= {E3_psi:,.0f} psi")

        if st.button("คำนวณ", type="primary", key="btn_A"):
            r = ck.mode_A(MR3, k3, E3_psi)
            st.success(
                f"**D_SB required = {r['D_required_in']:.3f} นิ้ว "
                f"= {r['D_required_cm']:.1f} cm**"
            )

    elif mode_key == "B":
        with mc1:
            cbr3 = st.number_input("CBR (%)", 1.0, 100.0, 3.0, 0.5, key="m_cbr")
            MR3  = cbr3 * 1500
            st.caption(f"M_R = {MR3:,.0f} psi")
        with mc2:
            k3 = st.number_input("k∞ target (pci)", 50, 2000, 200, 50, key="m_k")
        with mc3:
            D3_cm = st.number_input("D_SB (cm)", 1.0, 100.0, 20.0, 1.0, key="m_d")
            D3_in = D3_cm * CM_TO_INCH
            st.caption(f"= {D3_in:.3f} นิ้ว")

        if st.button("คำนวณ", type="primary", key="btn_B"):
            r = ck.mode_B(MR3, k3, D3_in)
            st.success(
                f"**E_SB required = {r['E_required_psi']:,.0f} psi "
                f"= {r['E_required_MPa']:.1f} MPa**"
            )

    elif mode_key == "C":
        with mc1:
            cbr3 = st.number_input("CBR (%)", 1.0, 100.0, 3.0, 0.5, key="m_cbr")
            MR3  = cbr3 * 1500
            st.caption(f"M_R = {MR3:,.0f} psi")
        with mc2:
            E3_MPa = st.number_input("E_SB (MPa)", 10, 5000, 850, 50, key="m_e")
            E3_psi = E3_MPa * MPa_TO_PSI
            st.caption(f"= {E3_psi:,.0f} psi")
        with mc3:
            D3_cm = st.number_input("D_SB (cm)", 1.0, 100.0, 20.0, 1.0, key="m_d")
            D3_in = D3_cm * CM_TO_INCH
            st.caption(f"= {D3_in:.3f} นิ้ว")

        if st.button("คำนวณ", type="primary", key="btn_C"):
            r = ck.mode_C(MR3, E3_psi, D3_in)
            st.success(f"**k∞ = {r['k_actual_pci']:.1f} pci**")

    else:  # Mode D
        with mc1:
            E3_MPa = st.number_input("E_SB (MPa)", 10, 5000, 850, 50, key="m_e")
            E3_psi = E3_MPa * MPa_TO_PSI
            st.caption(f"= {E3_psi:,.0f} psi")
        with mc2:
            D3_cm = st.number_input("D_SB (cm)", 1.0, 100.0, 20.0, 1.0, key="m_d")
            D3_in = D3_cm * CM_TO_INCH
            st.caption(f"= {D3_in:.3f} นิ้ว")
        with mc3:
            k3 = st.number_input("k∞ (pci)", 50, 2000, 200, 50, key="m_k")

        if st.button("คำนวณ", type="primary", key="btn_D"):
            r = ck.mode_D(E3_psi, D3_in, k3)
            cbr_back = r['MR_psi'] / 1500
            st.success(
                f"**M_R = {r['MR_psi']:,.0f} psi  →  CBR ≈ {cbr_back:.1f}%**"
            )
            st.info(f"k∞ check (จาก E+D) = {r['k_check_pci']:.1f} pci")

    # อ้างอิง
    st.divider()
    st.caption(
        "อ้างอิง: AASHTO Guide for Design of Pavement Structures 1993, Figure 3.3 | "
        "E_eq คำนวณโดย Odemark Weighted Method | "
        "Interpolation: RBF Thin-Plate Spline (scipy)"
    )
