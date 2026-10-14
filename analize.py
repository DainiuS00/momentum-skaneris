"""
analize.py — GRYNOS skaičiavimo funkcijos ir strategijos parametrai.

Šį modulį naudoja ir app.py (Streamlit skaneris), ir backtest.py (istorinis
testas). Čia NĖRA jokio Streamlit ir jokių tinklo užklausų — tik matematika.
Todėl backtestas garantuotai skaičiuoja lygiai taip pat, kaip gyvas skaneris.
"""

import numpy as np
import pandas as pd

# ============================================================
# STRATEGIJOS PARAMETRAI
# ============================================================

POZICIJU_SKAICIUS = 8          # TOP N į portfelį
SEKTORIAUS_LUBOS = 3            # max pozicijų iš vieno sektoriaus
PRAMONES_LUBOS = 2               # max pozicijų iš vienos pramonės
LEISTI_RAUDONA = False           # ar raudonos (rizikingos) į portfelį
MIN_APYVARTA_GBP = 3_000_000     # min vidutinė 30 d. dienos apyvarta
M1_LUZIO_RIBA = -12.0            # jei m1 žemiau -> trendas lūžta

GLODINIMO_DIENOS = 10       # vidurkis abiejuose galuose vietoj vienos kainos
MOMENTUM_DIENOS = 63        # ~3 mėnesiai prekybos dienomis
MIN_R2 = 0.50               # kietas R² vartas
MIN_ISTORIJOS_DIENU = 252   # trumpesni -> "Nauji vardai"
EARNINGS_DIENOS = 7         # neimam, jei ataskaita per tiek dienų

MA_FILTRAS = 100            # 100, 200 arba 0 (išjungta)
LIPNUMO_RIBA = 25           # laikom, kol dar TOP 25
ANOMALIJOS_RIBA = 200.0     # >200 %/3 mėn. -> anomalija (rodoma, ne slepiama)

MIN_VOLATILUMAS = 8.0       # akcijai < 8 % metinio volatilumo fiziškai nebūna
MIN_UNIKALIU_KAINU = 0.60   # bent 60 % kainų per 63 d. turi būti skirtingos
MAX_RSI_PORTFELIUI = 85.0   # virš to — per karšta naujai pozicijai

BAZINE_VALIUTA = "GBP"
PENSU_PRIESAGOS = (".L", ".IL")   # kotiruoja pensais — dalinti iš 100

# Score svoriai (z-balai)
SVORIS_TRENDAS = 0.45   # glodintas m3 × R²
SVORIS_M6 = 0.30
SVORIS_M12 = 0.15
SVORIS_FUND = 0.10

# Kokybės vartai (v3.1)
IC_KIETA_RIBA = 1.5
IC_MINKSTA_RIBA = 3.0
FCF_NI_KIETA = 0.0
SHARES_KIETA_RIBA = 8.0
SHARES_MINKSTA_RIBA = 3.0
R2_MINKSTA_RIBA = 0.65
KOKYBES_BAUDA_TASKAI = 0.15
SPLIT_ITARIMO_RIBA = 40.0   # v3.2: akcijų sk. pokytis > 40 % be splito įrašo -> ❓, ne atmetimas

# Sektoriai, kuriems Interest Coverage ir D/E nėra prasmingi (v3.2)
FINANSU_SEKTORIAI = {"Financial Services", "Financial", "Financials"}

# ============================================================
# VALIUTOS
# ============================================================

VALIUTU_ZEMELAPIS = {
    ".L": "GBP",   ".IL": "GBP",
    ".DE": "EUR",  ".F": "EUR",   ".PA": "EUR",  ".AS": "EUR",
    ".BR": "EUR",  ".LS": "EUR",  ".MC": "EUR",  ".MI": "EUR",
    ".VI": "EUR",  ".IR": "EUR",  ".HE": "EUR",
    ".SW": "CHF",
    ".ST": "SEK",
    ".CO": "DKK",
    ".OL": "NOK",
    ".WA": "PLN",
    ".T":  "JPY",  ".JP": "JPY",
    ".HK": "HKD",
}


def nustatyti_valiuta(tickeris):
    for priesaga, val in VALIUTU_ZEMELAPIS.items():
        if tickeris.endswith(priesaga):
            return val
    return "USD"


def nustatyti_regiona(t):
    if t.endswith(".T") or t.endswith(".JP"):
        return "JP"
    if t.endswith(".HK"):
        return "HK"
    if any(t.endswith(s) for s in [".L", ".DE", ".PA", ".AS", ".MC", ".MI", ".ST",
                                   ".CO", ".OL", ".HE", ".BR", ".LS", ".VI",
                                   ".WA", ".IR", ".SW", ".F"]):
        return "EU"
    return "US"


def normalizuoti_indeksa(idx):
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def i_gbp(kainos, fx):
    """kainos vietine valiuta -> GBP. fx = kiek vietinės valiutos už 1 GBP."""
    if fx is None:
        return kainos
    kursas = fx.reindex(kainos.index).ffill().bfill()
    if kursas.isna().all():
        return kainos
    return kainos / kursas


# ============================================================
# SKAIČIAVIMAI
# ============================================================

def momentum_proc(kaina_pradzioje, kaina_dabar):
    if kaina_pradzioje is None or kaina_pradzioje == 0 or pd.isna(kaina_pradzioje):
        return None
    return (kaina_dabar - kaina_pradzioje) / kaina_pradzioje * 100


def skaiciuoti_momentum(kainos, dienos=MOMENTUM_DIENOS, glodinimas=GLODINIMO_DIENOS):
    """Grąžina (glodintas m3 %, R²). Bet kuris gali būti None."""
    reikia = dienos + glodinimas
    if len(kainos) < reikia:
        return None, None

    galas = float(kainos.iloc[-glodinimas:].mean())
    pradzia = float(kainos.iloc[-reikia:-dienos].mean())
    if pradzia <= 0:
        return None, None
    m3 = (galas - pradzia) / pradzia * 100

    langas = kainos.iloc[-dienos:].values.astype(float)
    if (langas <= 0).any():
        return round(m3, 1), None
    y = np.log(langas)
    x = np.arange(len(y), dtype=float)
    try:
        nuolydis, laisvasis = np.polyfit(x, y, 1)
    except Exception:
        return round(m3, 1), None
    prognoze = nuolydis * x + laisvasis
    ss_lieka = float(((y - prognoze) ** 2).sum())
    ss_visas = float(((y - y.mean()) ** 2).sum())
    if ss_visas <= 0:
        return round(m3, 1), None
    r2 = 1.0 - ss_lieka / ss_visas
    return round(m3, 1), round(min(max(r2, 0.0), 1.0), 3)


def paprastas_momentum(kainos, dienos):
    """Vieno taško momentum (m1/m6/m12)."""
    if len(kainos) <= dienos:
        return None
    return momentum_proc(float(kainos.iloc[-dienos - 1]), float(kainos.iloc[-1]))


def skaiciuoti_rsi(kainos, periodas=14):
    if len(kainos) < periodas + 1:
        return None
    pokyciai = kainos.diff().dropna()
    kilimai = pokyciai.clip(lower=0)
    kritimai = -pokyciai.clip(upper=0)
    vid_k = kilimai.ewm(alpha=1 / periodas, adjust=False).mean().iloc[-1]
    vid_kr = kritimai.ewm(alpha=1 / periodas, adjust=False).mean().iloc[-1]
    if vid_kr == 0:
        return 100.0
    rs = vid_k / vid_kr
    return round(100 - (100 / (1 + rs)), 1)


def skaiciuoti_volatiluma(kainos, dienos=63):
    """Grąžina (metinis vol %, None) arba (None, problemos tekstas)."""
    if len(kainos) < dienos:
        return None, "trumpa istorija"

    langas = kainos.iloc[-dienos:]

    unikaliu = langas.nunique() / len(langas)
    if unikaliu < MIN_UNIKALIU_KAINU:
        return None, f"pasikartojančios kainos ({unikaliu:.0%} unikalių)"

    graza = langas.pct_change().dropna()
    if len(graza) < 20:
        return None, "per mažai duomenų"

    nuliniu = (graza.abs() < 1e-9).sum() / len(graza)
    if nuliniu > 0.35:
        return None, f"{nuliniu:.0%} dienų kaina nejudėjo"

    vol = float(graza.std() * np.sqrt(252) * 100)

    if vol < MIN_VOLATILUMAS:
        return None, f"volatilumas {vol:.1f}% — neįmanomai žemas"

    return round(vol, 1), None


def z_balai(reiksmes):
    arr = np.array([np.nan if v is None else float(v) for v in reiksmes], dtype=float)
    galiojantys = arr[~np.isnan(arr)]
    if len(galiojantys) < 2:
        return np.zeros(len(arr))
    vid, std = galiojantys.mean(), galiojantys.std()
    if std == 0:
        return np.zeros(len(arr))
    z = (arr - vid) / std
    z = np.nan_to_num(z, nan=0.0)
    return np.clip(z, -3, 3)


def bazinis_score(rezultatai, su_fundamentais=True):
    """
    Bendras balas: 0.45 z(m3×R²) + 0.30 z(m6) + 0.15 z(m12) [+ 0.10 z(fund)]
    MINUS kokybės minkšta bauda. Įrašo r["TrendoBalas"] ir r["Score"], grąžina
    surikiuotą sąrašą. Naudojama ir skaneryje, ir backteste.
    """
    if not rezultatai:
        return rezultatai
    for r in rezultatai:
        r2 = r.get("R2")
        r["TrendoBalas"] = round(r["m3"] * r2, 1) if r2 is not None else None

    z_pagr = z_balai([r["TrendoBalas"] for r in rezultatai])
    z_m6 = z_balai([r.get("m6") for r in rezultatai])
    z_m12 = z_balai([r.get("m12") for r in rezultatai])
    z_fund = z_balai([r.get("FundBalas") for r in rezultatai]) if su_fundamentais \
        else np.zeros(len(rezultatai))

    for i, r in enumerate(rezultatai):
        s = (SVORIS_TRENDAS * z_pagr[i] + SVORIS_M6 * z_m6[i]
             + SVORIS_M12 * z_m12[i] + SVORIS_FUND * z_fund[i])
        s -= r.get("KokybesBauda", 0.0) or 0.0
        r["Score"] = round(float(s), 3)

    rezultatai.sort(key=lambda x: x["Score"], reverse=True)
    return rezultatai


def kainu_filtrai(kainos_gbp, kainos_vietine, kartele, ma_filtras=MA_FILTRAS,
                  min_r2=MIN_R2, taikyti_absoliutu=True):
    """
    Visi KAINŲ filtrai vienoje vietoje (naudoja ir skaneris, ir backtestas).

    Grąžina (rodikliai: dict, atmetimo_priezastis: str|None, kategorija: str|None).
    v3.2: m3/m6/m12 — GBP (tavo reali grąža); R², RSI, volatilumas — vietine
    valiuta (akcijos elgesys be svaro triukšmo).
    """
    m3, _ = skaiciuoti_momentum(kainos_gbp)
    _, r2 = skaiciuoti_momentum(kainos_vietine)
    if m3 is None:
        return None, "trumpa istorija", "istorija"

    kaina_dabar = float(kainos_gbp.iloc[-1])
    m1 = paprastas_momentum(kainos_gbp, 21)
    m6 = paprastas_momentum(kainos_gbp, 126)
    m12 = paprastas_momentum(kainos_gbp, 252)

    ma = None
    if ma_filtras and len(kainos_gbp) >= ma_filtras:
        ma = float(kainos_gbp.iloc[-ma_filtras:].mean())

    rsi = skaiciuoti_rsi(kainos_vietine)
    volatilumas, vol_problema = skaiciuoti_volatiluma(kainos_vietine)

    langas = kainos_gbp.iloc[-252:] if len(kainos_gbp) >= 252 else kainos_gbp
    virsune_52 = float(langas.max())
    nuo_virsunes = round((kaina_dabar - virsune_52) / virsune_52 * 100, 1) \
        if virsune_52 > 0 else None

    rod = {
        "kaina": kaina_dabar, "m1": m1, "m3": m3, "m6": m6, "m12": m12,
        "r2": r2, "ma": ma, "rsi": rsi, "vol": volatilumas,
        "nuo52": nuo_virsunes,
    }

    if taikyti_absoliutu:
        if m3 <= kartele:
            return rod, f"3 mėn. {m3:+.1f}% vs kartelė {kartele:+.1f}%", "absoliutus"
        if ma is not None and kaina_dabar < ma:
            return rod, f"kaina žemiau {ma_filtras} d. vidurkio", "absoliutus_ma"
    if r2 is not None and r2 < min_r2:
        return rod, f"R² {r2:.2f} (riba {min_r2:.2f}) — juda chaotiškai", "r2"
    if m1 is not None and m1 < M1_LUZIO_RIBA:
        return rod, f"1 mėn. {m1:+.1f}%", "luzis"
    if volatilumas is None:
        return rod, vol_problema, "duomenys"
    return rod, None, None


def isejimo_signalas(rod, ma_filtras=MA_FILTRAS):
    """
    v3.2 (6A): ta pati taisyklė LAIKOMOMS pozicijoms, kaip ir įeinant.
    Grąžina priežastį parduoti arba None.
    """
    if not rod:
        return None
    if rod.get("ma") is not None and rod["kaina"] < rod["ma"]:
        return f"kaina žemiau {ma_filtras} d. vidurkio"
    if rod.get("m1") is not None and rod["m1"] < M1_LUZIO_RIBA:
        return f"trendas lūžta (1 mėn. {rod['m1']:+.1f}%)"
    return None
