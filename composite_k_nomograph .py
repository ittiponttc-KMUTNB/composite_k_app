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

# Upper Chart: k∞ curves
# แต่ละ curve คือ k∞ คงที่ มีจุด (E_SB psi, D_SB inches)
# Digitized จาก nomograph (log-log space)
UPPER_CHART_DATA = {
    50: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000],
        "D_in":  [19.5,  18.2,  16.5,  14.5,  13.0,  11.8,   10.2,   9.0]
    },
    100: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000],
        "D_in":  [16.0,  14.5,  12.8,  11.0,  9.8,   8.8,    7.6,    6.8,    5.8]
    },
    200: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000],
        "D_in":  [13.0,  11.5,  10.0,  8.5,   7.5,   6.7,    5.8,    5.2,    4.4,    3.5]
    },
    300: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000],
        "D_in":  [11.2,  9.8,   8.5,   7.2,   6.3,   5.6,    4.8,    4.3,    3.6,    2.9]
    },
    400: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000],
        "D_in":  [10.0,  8.8,   7.6,   6.4,   5.6,   5.0,    4.3,    3.8,    3.2,    2.5,    2.0]
    },
    500: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000],
        "D_in":  [9.2,   8.0,   6.9,   5.8,   5.1,   4.5,    3.9,    3.4,    2.9,    2.2,    1.8]
    },
    600: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000],
        "D_in":  [8.5,   7.5,   6.4,   5.4,   4.7,   4.2,    3.6,    3.2,    2.6,    2.0,    1.6,    1.3]
    },
    800: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000],
        "D_in":  [7.5,   6.6,   5.7,   4.8,   4.2,   3.7,    3.2,    2.8,    2.3,    1.8,    1.4,    1.1]
    },
    1000: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000],
        "D_in":  [6.8,   6.0,   5.2,   4.3,   3.8,   3.3,    2.8,    2.5,    2.1,    1.6,    1.2,    1.0]
    },
    1500: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000],
        "D_in":  [5.8,   5.1,   4.4,   3.7,   3.2,   2.8,    2.4,    2.1,    1.7,    1.3,    1.0,    0.8]
    },
    2000: {
        "E_psi": [15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 750000, 1000000],
        "D_in":  [5.2,   4.6,   3.9,   3.3,   2.8,   2.5,    2.1,    1.9,    1.5,    1.1,    0.9,    0.7]
    }
}

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
        points = []   # (log E_SB, log D_SB)
        values = []   # log k∞

        for k_val, data in UPPER_CHART_DATA.items():
            for e, d in zip(data["E_psi"], data["D_in"]):
                if e > 0 and d > 0:
                    points.append([np.log10(e), np.log10(d)])
                    values.append(np.log10(k_val))

        self._points = np.array(points)
        self._values = np.array(values)
        self._rbf = RBFInterpolator(self._points, self._values,
                                     kernel='thin_plate_spline', smoothing=0.01)

        # bounds สำหรับ warning
        self.E_min, self.E_max = 15000, 1000000   # psi
        self.D_min, self.D_max = 0.5,   20.0      # inches
        self.k_min, self.k_max = 50,    2000       # pci

    def get_k(self, E_psi: float, D_in: float) -> float:
        """(E_SB psi, D_SB inches) -> k∞ (pci)"""
        pt = np.array([[np.log10(E_psi), np.log10(D_in)]])
        log_k = float(self._rbf(pt)[0])
        k = 10 ** log_k
        return float(np.clip(k, self.k_min, self.k_max))

    def get_D(self, E_psi: float, k_target: float,
              D_bounds=(0.5, 20.0)) -> float:
        """
        กำหนด E_SB + k∞ target -> หา D_SB (inches)
        ใช้ bisection search
        """
        from scipy.optimize import brentq
        def obj(d):
            return self.get_k(E_psi, d) - k_target

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
              E_bounds=(15000, 1000000)) -> float:
        """
        กำหนด D_SB + k∞ target -> หา E_SB (psi)
        ใช้ bisection search
        """
        from scipy.optimize import brentq
        def obj(e):
            return self.get_k(e, D_in) - k_target

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
        k_actual = self.upper.get_k(E_psi, D_in)
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

        k_actual = self.upper.get_k(E_eq_psi, D_total_in)

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
            D_in  = self.upper.get_D(E_psi, k_target_pci)
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
if __name__ == "__main__":
    ck = CompositeK()

    print("=" * 60)
    print("AASHTO Figure 3.3 — Composite K Module Self Test")
    print("=" * 60)

    # Test Turning Line
    print("\n[Turning Line]")
    for mr in [3000, 5000, 7000, 10000]:
        d = ck.turning.mr_to_dsb(mr)
        print(f"  M_R={mr:6,} psi  ->  D_SB={d:.2f} inches")

    # Test Mode C (ตรวจสอบกับ nomograph)
    print("\n[Mode C: E_SB + D_SB -> k∞]")
    tests = [
        (20000, 10.0),
        (50000, 8.0),
        (100000, 6.0),
        (200000, 5.0),
    ]
    for e, d in tests:
        r = ck.mode_C(5000, e, d)
        print(f"  E={e:>8,} psi, D={d:.1f}\"  ->  k∞={r['k_actual_pci']:.0f} pci")

    # Test Mode A (ตัวอย่างจากการออกแบบ)
    print("\n[Mode A: M_R + k∞ + E_SB -> D_SB required]")
    cbr = 3
    MR  = cbr * 1500
    k_target = 200
    print(f"  CBR={cbr}% -> M_R={MR:,} psi, k∞ target={k_target} pci")
    for name, E_MPa in list(DEFAULT_MATERIALS.items())[:4]:
        E_psi = E_MPa * MPa_TO_PSI
        r = ck.mode_A(MR, k_target, E_psi)
        print(f"  {name[:35]:35s}  D_req={r['D_required_in']:.2f}\" ({r['D_required_cm']:.1f} cm)")

    # Test Multilayer
    print("\n[Multilayer: 2 ชั้น]")
    layers = [
        {"name": "หินคลุกผสมซีเมนต์ UCS 24.5 ksc",   "E_MPa": 850, "D_cm": 20},
        {"name": "รองพื้นทางวัสดุมวลรวม CBR 25%",     "E_MPa": 150, "D_cm": 15},
    ]
    r = ck.multilayer_k(layers, MR_psi=7000)
    print(f"  E_eq={r['E_eq_MPa']} MPa ({r['E_eq_psi']:,} psi)")
    print(f"  D_total={r['D_total_cm']} cm ({r['D_total_in']:.3f} in)")
    print(f"  k∞ actual={r['k_actual_pci']} pci")
