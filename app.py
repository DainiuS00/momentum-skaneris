"""
MOMENTUM SKANERIS v3.2
======================
PERTVARKYMAS v3.2 (2026-09-08, naujienos virš v3.1):
  1. Rezultatai saugomi st.session_state — nebedingsta spaudžiant/įvedant.
     Turimos pozicijos prisimenamos faile turimos_pozicijos.txt.
  2. Akcijų splitai koreguojami skaičiuojant diluciją (nebeatmeta NVDA/AVGO tipo vardų).
  3. Finansų sektoriui Interest Coverage ir D/E vartai netaikomi (bankams beprasmiai).
  4. Kainos siunčiamos paketais per yf.download (5–10× greičiau), cache 12 val.;
     info nebetraukiamas du kartus.
  5. Rinkų skydelis cache'inamas 15 min.
  6. Išėjimo taisyklė LAIKOMOMS pozicijoms: kaina < 100 d. MA arba m1 < −12 % -> parduoti.
  8. Kartelė (S&P 500) skaičiuojama TA PAČIA glodinta formule kaip akcijos.
  9. R², RSI ir volatilumas — vietine valiuta; momentum — GBP.
 10. JP/HK be FX kurso -> ❓ atmetama, ne spėjama.
 12. Suderinamumas su senesniu Streamlit (width/use_container_width).
  * Grynos funkcijos ir parametrai perkelti į analize.py (naudoja ir backtest.py).

--- v3.1 ---
PERTVARKYMAS v3.1 (naujienos virš v3.0):
  A. HIBRIDINIS KOKYBĖS VARTŲ MODELIS — nauji rodikliai PRIEŠ momentum
     rikiavimą, bet neįsigilinant per giliai visame universume:
       KIETI VARTAI (automatiškai atmeta, atskira lentelė su priežastimi):
         - Interest Coverage (EBIT / palūkanos) < 1.5
         - FCF / Net Income < 0
         - Akcijų skaičiaus pokytis per 12 mėn. > +8% (dilucija)
         - R² < 0.50 (nėra tvarkingo trendo)
       MINKŠTA NUOBAUDA (sumažina score, bet nepašalina):
         - Interest Coverage 1.5–3.0
         - Akcijų skaičiaus pokytis 3–8%
         - R² 0.50–0.65
  B. Duomenų trūkumas (❓) NIEKADA automatiškai neatmeta — tik pažymimas.
     Jei ❓ pozicija patenka į TOP 8, prie jos rodoma atsarginė alternatyva.
  C. ATSARGINIŲ POZICIJŲ SISTEMA — kiekvienai TOP 8 pozicijai laikomi
     2–3 sekantys geriausi kandidatai iš tos pačios pramonės/sektoriaus,
     rodomi atskiroje lentelėje. Naudinga, jei pagrindinė pozicija vėliau
     atmetama arba tiesiog norisi alternatyvos.
  D. POZICIJŲ_SKAIČIUS sumažintas nuo 15 iki 8 (atitinka realų portfelį).
  E. TOP 8 lentelė aiškiai atskirta spalva/antrašte nuo platesnio sąrašo
     (kuris liks studijoms/statistikai vėliau).

Ankstesnės v3.0 pertvarkos (glodintas momentum + R², pakopinis filtravimas
prieš rikiavimą, paskutinis uždarymas, atmestieji pagal priežastį) — visos
išlieka nepakeistos, žr. originalų aprašymą kode žemiau.
"""

import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from analize import *  # noqa: F401,F403 — strategijos parametrai ir skaičiavimai
import analize

# ============================================================
# NUSTATYMAI
# ============================================================

# --- APP LYGIO NUSTATYMAI (strategijos parametrai — analize.py) ---
RODYTI_ANOMALIJAS = True    # rodyti >200%/3men, o ne slepti
LYGINTI_SU_INDEKSU = True   # m3 turi lenkti indeksa, ne tik buti > 0
INDEKSO_ATSARGA = 0.0       # papildomi procentiniai punktai virs indekso
RSI_PREMIJA = False         # RSI dubliuoja momentum - pagal nutylejima isjungta
EKSPORTO_KATALOGAS = "skenavimai"
POZICIJU_FAILAS = "turimos_pozicijos.txt"   # v3.2: tavo pozicijos prisimenamos
MIN_KAPITALIZACIJA_GBP = 500_000_000   # min rinkos kapitalizacija
REIKALAUTI_SEKTORIAUS = True           # atmeta vardus be fundamentu duomenu
R2_KIETA_RIBA = MIN_R2
ATSARGU_SKAICIUS = 3         # kiek atsarginiu kandidatu laikyti kiekvienai TOP pozicijai
KAINU_CACHE_VAL = 12         # v3.2: kainu istorija cache'inama tiek valandu
PAKETO_DYDIS = 150           # v3.2: kiek tickeriu vienoje yf.download uzklausoje

# Streamlit suderinamumas (12): width="stretch" atsirado 1.45 versijoje
def _st_versija():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)
DF_KW = {"width": "stretch"} if _st_versija() >= (1, 45) else {"use_container_width": True}


st.set_page_config(
    page_title="Momentum Skaneris v3.2",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="📈",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    [data-testid="stMetricValue"] { font-size: 0.9rem; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem; }
    .stButton > button { width: 100%; padding: 0.6rem; font-size: 1rem; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Momentum Skaneris v3.2")
st.caption("🟢 Stiprus | 🟡 Įspėjimas | 🔴 Rizikinga | 🔵 Overbought | ⚫ Oversold | ❓ Trūksta kokybės duomenų")
st.caption("Visos kainos GBP · 3 mėn. lookback · hibridiniai kokybės vartai · sektoriaus lubos")


# ============================================================
# VALIUTOS (tinklo dalis; skaičiavimai — analize.py)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def gauti_fx_serija(valiuta):
    """Kiek vietinės valiutos už 1 GBP, dieninė serija (2 m.)."""
    if valiuta == BAZINE_VALIUTA:
        return None
    try:
        fx = yf.Ticker(f"GBP{valiuta}=X").history(period="2y", auto_adjust=True)
        if fx.empty:
            return None
        s = fx["Close"].dropna()
        s.index = normalizuoti_indeksa(s.index)
        s = s[~s.index.duplicated(keep="last")]
        return s
    except Exception:
        return None


# ============================================================
# v3.2 (4A): KAINŲ ISTORIJA PAKETAIS + CACHE
# ============================================================

def _isskaidyti_istorijas(df, tickeriai):
    """yf.download rezultatą (MultiIndex) paverčia į {ticker: DataFrame}."""
    istorijos = {}
    if df is None or df.empty:
        return istorijos
    if isinstance(df.columns, pd.MultiIndex):
        turimi = set(df.columns.get_level_values(0))
        for t in tickeriai:
            if t not in turimi:
                continue
            d = df[t].dropna(how="all")
            if d.empty or "Close" not in d.columns:
                continue
            istorijos[t] = d[["Close", "Volume"]] if "Volume" in d.columns else d[["Close"]]
    else:
        # vienas tickeris — paprastos kolonos
        if len(tickeriai) == 1 and "Close" in df.columns:
            d = df.dropna(how="all")
            istorijos[tickeriai[0]] = d[["Close", "Volume"]] if "Volume" in d.columns else d[["Close"]]
    return istorijos


@st.cache_data(ttl=KAINU_CACHE_VAL * 3600, show_spinner=False)
def gauti_kainu_paketa(tickeriai_tuple):
    """Vienas yf.download iki PAKETO_DYDIS tickerių. Cache'inama 12 val."""
    tickeriai = list(tickeriai_tuple)
    try:
        df = yf.download(
            tickeriai, period="2y", auto_adjust=True, group_by="ticker",
            threads=True, progress=False, actions=False,
        )
    except Exception:
        df = None
    istorijos = _isskaidyti_istorijas(df, tickeriai)

    # Ko nepavyko paketu — pabandom po vieną (Yahoo kartais nutyli dalį)
    truksta = [t for t in tickeriai if t not in istorijos]
    if truksta and len(truksta) <= max(10, len(tickeriai) // 5):
        for t in truksta:
            try:
                d = yf.Ticker(t).history(period="2y", auto_adjust=True)
                if not d.empty and "Close" in d.columns:
                    istorijos[t] = d[["Close", "Volume"]] if "Volume" in d.columns else d[["Close"]]
            except Exception:
                pass
    return istorijos


def gauti_kainu_istorijas(tickeriai, statusas_txt=None):
    """Visų tickerių istorijos paketais po PAKETO_DYDIS."""
    istorijos = {}
    paketai = [tickeriai[i:i + PAKETO_DYDIS] for i in range(0, len(tickeriai), PAKETO_DYDIS)]
    for i, paketas in enumerate(paketai, start=1):
        if statusas_txt is not None:
            statusas_txt.text(f"📥 Siunčiamos kainos: paketas {i}/{len(paketai)} "
                              f"({len(paketas)} tickerių)…")
        istorijos.update(gauti_kainu_paketa(tuple(paketas)))
    return istorijos


@st.cache_data(ttl=86400, show_spinner=False)
def gauti_info(tickeris):
    """Yahoo `info` žodynas, cache 24 val. (nebetraukiamas du kartus)."""
    try:
        info = yf.Ticker(tickeris).info
        return info if info else {}
    except Exception:
        return {}


# ============================================================
# FUNDAMENTAI
# ============================================================

def istraukti_fundamentalus(info):
    if not info or len(info) < 5:
        return {}
    return {
        "pavadinimas": info.get("shortName") or info.get("longName") or "—",
        "revenue_growth": info.get("revenueGrowth"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cash_flow": info.get("freeCashflow"),
        "roe": info.get("returnOnEquity"),
        "pe": info.get("trailingPE"),
        "sektorius": info.get("sector") or "—",
        "pramone": info.get("industry") or "—",
        "market_cap": info.get("marketCap"),
    }


def yra_finansu_sektorius(sektorius):
    """v3.2 (3A): bankams/draudikams IC ir D/E rodikliai beprasmiai."""
    return (sektorius or "") in FINANSU_SEKTORIAI


def vertinti_fundamentalus(f, regionas="US"):
    balas = 0
    ispejimai = []
    truksta = 0
    raudoni = 0
    geltoni = 0
    finansai = yra_finansu_sektorius(f.get("sektorius"))

    rg = f.get("revenue_growth")
    d2e_raw = f.get("debt_to_equity")
    fcf = f.get("free_cash_flow")
    roe = f.get("roe")

    if finansai:
        # D/E bankams visada "aukštas" — tai jų verslo modelis, ne rizika.
        # Vertinam neutraliai (0 balų, be įspėjimų, nelaikom trūkstamu).
        d2e_raw = "n/a"

    if rg is None:
        truksta += 1
    elif rg > 0.15:
        balas += 20
    elif rg > 0.05:
        balas += 10
    elif rg < -0.10:
        balas -= 15
        ispejimai.append("📉 Pajamos↓")
        raudoni += 1
    elif rg < 0:
        balas -= 5
        ispejimai.append("⚠️ Pajamos↓")
        geltoni += 1

    d2e = None
    if d2e_raw == "n/a":
        pass
    elif d2e_raw is None:
        truksta += 1
    else:
        d2e = d2e_raw / 100.0
        if d2e < 0.5:
            balas += 10
        elif d2e < 1.5:
            balas += 5
        elif d2e < 3.0:
            ispejimai.append("⚠️ Skola↑")
            geltoni += 1
        else:
            balas -= 15
            ispejimai.append("🔴 Skola aukšta")
            raudoni += 1

    if fcf is None:
        truksta += 1
    elif fcf > 0:
        balas += 10
    else:
        balas -= 5
        ispejimai.append("⚠️ FCF<0")
        geltoni += 1

    if roe is None:
        truksta += 1
    elif roe > 0.15:
        balas += 15
    elif roe > 0.05:
        balas += 5
    elif roe < -0.10:
        balas -= 20
        ispejimai.append("📉 Nuostolis")
        raudoni += 1
    elif roe < 0:
        balas -= 5
        ispejimai.append("⚠️ ROE<0")
        geltoni += 1

    if truksta >= 2:
        ispejimai.append(f"❓ {truksta}/4 trūksta")
        geltoni += 1

    veto = (fcf is not None and fcf < 0) and (d2e is not None and d2e > 3.0)
    if veto:
        ispejimai.append("⛔ FCF<0 + skola>3x")

    if raudoni >= 2:
        statusas = "raudona"
    elif raudoni == 1 and geltoni >= 1:
        statusas = "raudona"
    elif raudoni == 1 or geltoni >= 2:
        statusas = "geltona"
    elif geltoni == 1:
        statusas = "geltona"
    else:
        statusas = "zalia"

    return balas, ispejimai, veto, statusas


SPALVOS = {"zalia": "🟢", "geltona": "🟡", "raudona": "🔴"}


# ============================================================
# v3.1: KOKYBĖS VARTAI (kieti + minkšti)
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def gauti_kokybes_zaliavas(tickeris, fcf=None, ni=None, tikrinti_ic=True):
    """
    Papildomi duomenys kietiems/minkštiems kokybės vartams:
      - Interest Coverage = EBIT / |palūkanų išlaidos|  (praleidžiama finansams)
      - FCF / Net Income santykis (fcf, ni perduodami iš jau turimo `info`)
      - Akcijų skaičiaus pokytis per ~12 mėn. (dilucija), v3.2: koreguotas splitams

    SVARBU: jei duomenų nėra, grąžina None — NIEKADA neverčia/nespėlioja.
    Trūkstami duomenys vėliau žymimi ❓, ne automatiškai atmetami.
    """
    rezultatas = {"interest_coverage": None, "fcf_ni": None,
                  "shares_change_12m": None, "split_pastaba": None}
    try:
        akcija = yf.Ticker(tickeris)

        # --- Interest Coverage ---
        if tikrinti_ic:
            try:
                fin = akcija.financials
                if fin is not None and not fin.empty:
                    ebit = None
                    for eil in ("Operating Income", "EBIT"):
                        if eil in fin.index:
                            v = fin.loc[eil].iloc[0]
                            if pd.notna(v):
                                ebit = float(v)
                                break
                    palukanos = None
                    for eil in ("Interest Expense", "Interest Expense Non Operating"):
                        if eil in fin.index:
                            v = fin.loc[eil].iloc[0]
                            if pd.notna(v):
                                palukanos = abs(float(v))
                                break
                    if ebit is not None and palukanos and palukanos > 0:
                        rezultatas["interest_coverage"] = round(ebit / palukanos, 2)
            except Exception:
                pass

        # --- FCF / Net Income (iš jau turimo info — nebetraukiam antrą kartą) ---
        try:
            if fcf is not None and ni is not None:
                if fcf < 0:
                    # Neigiamas FCF - visada realus rizikos signalas.
                    rezultatas["fcf_ni"] = round(fcf / abs(ni), 2) if ni != 0 else -999.0
                elif ni <= 0:
                    # FCF teigiamas, NI neigiamas (nepiniginis nurašymas) -> ❓, ne atmetimas
                    rezultatas["fcf_ni"] = None
                else:
                    rezultatas["fcf_ni"] = round(fcf / ni, 2)
        except Exception:
            pass

        # --- Akcijų skaičiaus pokytis per ~12 mėn., v3.2 (2A): SPLITŲ KOREKCIJA ---
        try:
            iki = dt.date.today()
            nuo = iki - dt.timedelta(days=380)
            serija = akcija.get_shares_full(start=nuo.isoformat(), end=iki.isoformat())
            if serija is not None and len(serija) >= 2:
                serija = serija.dropna()
                pirma = float(serija.iloc[0])
                paskutine = float(serija.iloc[-1])
                pirma_data = pd.Timestamp(serija.index[0])
                paskutine_data = pd.Timestamp(serija.index[-1])

                # Splitai tame pačiame lange: pirmą reikšmę padauginam iš split
                # koeficientų, kad lygintume "tų pačių" akcijų skaičių.
                koef = 1.0
                try:
                    splitai = akcija.splits
                    if splitai is not None and len(splitai):
                        s_idx = pd.DatetimeIndex(splitai.index)
                        if s_idx.tz is not None:
                            s_idx = s_idx.tz_localize(None)
                        p_d = pirma_data.tz_localize(None) if pirma_data.tzinfo else pirma_data
                        pa_d = paskutine_data.tz_localize(None) if paskutine_data.tzinfo else paskutine_data
                        for d_, r_ in zip(s_idx, splitai.values):
                            if p_d < d_ <= pa_d and r_ and r_ > 0:
                                koef *= float(r_)
                except Exception:
                    koef = 1.0
                if koef != 1.0:
                    rezultatas["split_pastaba"] = f"split ×{koef:g} koreguotas"

                if pirma > 0:
                    pokytis = (paskutine - pirma * koef) / (pirma * koef) * 100
                    if abs(pokytis) > SPLIT_ITARIMO_RIBA:
                        # Per didelis pokytis be užfiksuoto splito — greičiausiai
                        # duomenų klaida / nepažymėtas splitas. ❓, ne atmetimas.
                        rezultatas["shares_change_12m"] = None
                        rezultatas["split_pastaba"] = f"akcijų sk. {pokytis:+.0f}% — įtartina, ignoruota"
                    else:
                        rezultatas["shares_change_12m"] = round(pokytis, 1)
        except Exception:
            pass

    except Exception:
        pass

    return rezultatas


def vertinti_kokybes_vartus(kokybe, r2, r2_kieta_riba=None, r2_minksta_riba=None,
                            sektorius=None):
    """
    v3.1 HIBRIDINIS MODELIS.

    r2_kieta_riba / r2_minksta_riba: jei perduota, naudoja šias vertes vietoj
    fiksuotų globalių R2_KIETA_RIBA/R2_MINKSTA_RIBA — leidžia suderinti su
    UI slankikliu (žr. paleisti()), kad ankstyvas filtras ir minkšta bauda
    visada naudotų TĄ PAČIĄ ribą.

    Grąžina:
      kietas_atmestas (bool)  — jei True, pozicija NEPATENKA į kandidatus
      priezastis (str|None)   — kodėl (kietiems atvejams)
      minksta_bauda (float)   — atimama iš galutinio score
      zenklai (list[str])     — trumpi žymekliai rodymui (⚠️/❓)
      truksta (bool)          — ar bent vieno rodiklio duomenų nėra
    """
    ic = kokybe.get("interest_coverage")
    fcf_ni = kokybe.get("fcf_ni")
    shares = kokybe.get("shares_change_12m")

    r2_kieta = r2_kieta_riba if r2_kieta_riba is not None else R2_KIETA_RIBA
    r2_minksta = r2_minksta_riba if r2_minksta_riba is not None else R2_MINKSTA_RIBA

    priezastys_kietos = []
    zenklai = []
    bauda = 0.0
    truksta = False

    # --- Interest Coverage ---
    if yra_finansu_sektorius(sektorius):
        # v3.2 (3A): bankams/draudikams IC beprasmis — netaikom, nelaikom trūkstamu
        zenklai.append("🏦IC n/a")
    elif ic is None:
        truksta = True
        zenklai.append("❓IC")
    elif ic < IC_KIETA_RIBA:
        priezastys_kietos.append(f"Interest Coverage {ic:.1f}x (< {IC_KIETA_RIBA:.1f}x)")
    elif ic < IC_MINKSTA_RIBA:
        bauda += KOKYBES_BAUDA_TASKAI
        zenklai.append(f"⚠️IC {ic:.1f}x")

    # --- FCF / Net Income ---
    # PASTABA (v3.1 taisymas): fcf_ni jau atėjo None iš gauti_kokybes_zaliavas(),
    # jei FCF teigiamas, o NI neigiamas/nulinis (santykis būtų prasmės neturintis,
    # matematiškai apsivertęs dėl nepiniginio nurašymo, pvz. GIS/KHC 2026) -
    # tokiu atveju čia patenka į "❓ trūksta", NE į kietą atmetimą.
    if fcf_ni is None:
        truksta = True
        zenklai.append("❓FCF/NI")
    elif fcf_ni < FCF_NI_KIETA:
        priezastys_kietos.append(f"FCF/NI {fcf_ni:.2f} (neigiamas FCF)")

    # --- Akcijų dilucija ---
    if kokybe.get("split_pastaba"):
        zenklai.append(f"ℹ️ {kokybe['split_pastaba']}")
    if shares is None:
        truksta = True
        zenklai.append("❓Dilucija")
    elif shares > SHARES_KIETA_RIBA:
        priezastys_kietos.append(f"Akcijų sk. +{shares:.1f}% (> {SHARES_KIETA_RIBA:.0f}%)")
    elif shares > SHARES_MINKSTA_RIBA:
        bauda += KOKYBES_BAUDA_TASKAI
        zenklai.append(f"⚠️Dilucija +{shares:.1f}%")

    # --- R² (jau apskaičiuotas momentum žingsnyje; ankstyvas filtras
    #     jau kerta ties r2_kieta, čia tik minkšta bauda tarpiniam zonai) ---
    if r2 is not None and r2_kieta <= r2 < r2_minksta:
        bauda += KOKYBES_BAUDA_TASKAI
        zenklai.append(f"⚠️R² {r2:.2f}")

    kietas = len(priezastys_kietos) > 0
    priezastis = " · ".join(priezastys_kietos) if priezastys_kietos else None
    return kietas, priezastis, round(bauda, 2), zenklai, truksta


# ============================================================
# SEKTORIU SPALVOS
# ============================================================

SEKTORIU_SPALVOS = {
    "Technology":             "#1E4D8C",
    "Financial Services":     "#1B5E4A",
    "Healthcare":             "#5B3A7E",
    "Energy":                 "#8C4A1E",
    "Industrials":            "#4A4A6A",
    "Consumer Cyclical":      "#8C1E4D",
    "Consumer Defensive":     "#6B5B1E",
    "Real Estate":            "#1E6B6B",
    "Communication Services": "#7A2E5C",
    "Communication":          "#7A2E5C",
    "Basic Materials":        "#5C4A2E",
    "Utilities":              "#2E5C7A",
}
NEZINOMO_SEKTORIAUS_SPALVA = "#3A3A3A"


def sektoriaus_spalva(sektorius):
    return SEKTORIU_SPALVOS.get(sektorius, NEZINOMO_SEKTORIAUS_SPALVA)


def pasviesinti(hex_spalva, kiek=0.35):
    h = hex_spalva.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * kiek * 0.35)
    g = int(g + (255 - g) * kiek * 0.35)
    b = int(b + (255 - b) * kiek * 0.35)
    return f"#{r:02X}{g:02X}{b:02X}"


def spalvinti_lentele(df, sektoriu_stulpelis="Sektorius", pramones_stulpelis="Pramonė"):
    def stilius(eilute):
        sekt = eilute.get(sektoriu_stulpelis, "")
        pilnas = next((k for k in SEKTORIU_SPALVOS if k.startswith(str(sekt)[:10])), sekt)
        spalva = sektoriaus_spalva(pilnas)
        stiliai = pd.Series("", index=eilute.index)
        if sektoriu_stulpelis in eilute.index:
            stiliai[sektoriu_stulpelis] = (
                f"background-color: {spalva}; color: #FFFFFF; font-weight: 600;"
            )
        if pramones_stulpelis in eilute.index:
            stiliai[pramones_stulpelis] = (
                f"background-color: {pasviesinti(spalva)}; color: #E8E8E8;"
            )
        return stiliai

    formatai = {}
    for stulpelis in df.columns:
        if stulpelis in ("Score",):
            formatai[stulpelis] = "{:.3f}"
        elif stulpelis in ("R²",):
            formatai[stulpelis] = "{:.2f}"
        elif stulpelis in ("Kaina £",):
            formatai[stulpelis] = "{:.2f}"
        elif stulpelis in ("1m%", "3m%", "6m%", "12m%", "Nuo52v%",
                           "RSI", "Vol%", "P/E", "P/E ⓘ"):
            formatai[stulpelis] = "{:.1f}"

    return df.style.apply(stilius, axis=1).format(formatai, na_rep="—")


# ============================================================
# TICKERIU SARASAI
# ============================================================

def _kelias(failas):
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), failas)
    except Exception:
        return failas


def _skaityti_atsargini(failas):
    try:
        with open(_kelias(failas), "r", encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    except Exception:
        return []


def _issaugoti_atsargini(failas, tickeriai):
    try:
        with open(_kelias(failas), "w", encoding="utf-8") as f:
            f.write("\n".join(tickeriai))
    except Exception:
        pass


@st.cache_data(ttl=86400, show_spinner=False)
def gauti_tickerius_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        lentele = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
        tickeriai = [str(t).replace(".", "-") for t in lentele["Symbol"].tolist()]
        if len(tickeriai) > 400:
            _issaugoti_atsargini("sp500.txt", tickeriai)
            return tickeriai
    except Exception as e:
        st.warning(f"⚠️ Wikipedia neprieinama ({e}) — naudojama vietinė kopija")

    atsarginis = _skaityti_atsargini("sp500.txt")
    if atsarginis:
        st.info(f"ℹ️ Naudojamas išsaugotas S&P 500 sąrašas ({len(atsarginis)} vnt.)")
        return atsarginis

    st.error("❌ S&P 500 sąrašo gauti nepavyko ir kopijos nėra")
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def gauti_tickerius_europa():
    tickeriai = []
    try:
        from pytickersymbols import PyTickerSymbols
        sd = PyTickerSymbols()
        indeksai = [
            "DAX", "CAC_40", "FTSE 100", "IBEX 35", "AEX", "BEL 20",
            "OMX Stockholm 30", "OMX Helsinki 25", "MDAX", "SDAX",
            "TecDAX", "EURO STOXX 50", "Switzerland 20",
        ]
        pirmenybe = [".DE", ".PA", ".L", ".AS", ".MC", ".MI", ".ST", ".CO",
                     ".OL", ".HE", ".BR", ".LS", ".VI", ".WA", ".IR", ".SW", ".F"]
        for idx in indeksai:
            try:
                for akcija in sd.get_stocks_by_index(idx):
                    simboliai = [s.get("yahoo", "") for s in akcija.get("symbols", []) if s.get("yahoo")]
                    for sf in pirmenybe:
                        t = next((x for x in simboliai if x.endswith(sf)), None)
                        if t:
                            tickeriai.append(t)
                            break
            except Exception:
                pass
    except ImportError:
        st.warning("⚠️ pytickersymbols neįdiegta (`pip install pytickersymbols`)")

    if len(tickeriai) > 100:
        _issaugoti_atsargini("europa.txt", list(dict.fromkeys(tickeriai)))
    else:
        atsarginis = _skaityti_atsargini("europa.txt")
        if atsarginis:
            st.info(f"ℹ️ Naudojamas išsaugotas Europos sąrašas ({len(atsarginis)} vnt.)")
            tickeriai = atsarginis

    return list(dict.fromkeys([t for t in tickeriai if t]))


@st.cache_data(ttl=86400, show_spinner=False)
def gauti_tickerius_japonija():
    t = _skaityti_atsargini("nikkei225.txt")
    if len(t) > 50:
        return t
    st.info("ℹ️ nikkei225.txt nerasta — naudojamas atsarginis sąrašas")
    return [
        "6758.T", "6861.T", "6954.T", "6976.T", "6762.T", "6971.T", "6702.T",
        "6501.T", "6503.T", "6752.T", "7735.T", "7741.T", "7751.T", "7752.T",
        "7974.T", "4689.T", "9984.T", "9433.T", "9432.T", "9434.T",
        "7203.T", "7267.T", "7269.T", "7270.T", "7201.T", "7211.T", "7261.T",
        "8306.T", "8316.T", "8411.T", "8766.T", "8801.T", "8802.T", "8591.T",
        "6301.T", "6305.T", "6902.T", "7011.T", "7012.T", "7013.T",
        "5401.T", "5411.T", "5713.T", "5802.T", "3407.T", "4063.T", "4452.T",
        "4502.T", "4503.T", "4519.T", "4523.T", "4568.T", "4578.T",
        "8001.T", "8002.T", "8031.T", "8035.T", "8053.T", "8058.T",
        "2914.T", "3382.T", "4901.T", "6367.T", "6645.T", "6857.T", "6098.T",
        "6273.T", "7733.T", "9020.T", "9021.T", "9022.T", "9101.T", "9104.T",
        "1925.T", "1928.T", "2502.T", "2503.T", "3659.T", "4661.T",
        "6506.T", "6594.T", "6723.T", "6981.T", "8015.T", "8725.T",
        "9735.T", "9766.T", "9983.T",
    ]


def gauti_japonijos_adr():
    return [
        "TM", "SONY", "MUFG", "SMFG", "MFG", "HMC", "TAK", "NMR", "IX",
        "NTTYY", "SFTBY", "MITSY", "ITOCY", "SSUMY", "MSBHF", "HTHIY",
        "DSNKY", "KYOCY",
    ]


def gauti_tickerius_azija():
    return [
        "0700.HK", "9988.HK", "9618.HK", "9888.HK", "9999.HK", "3690.HK",
        "1810.HK", "2382.HK", "0005.HK", "1299.HK", "0388.HK", "2318.HK",
        "0939.HK", "1398.HK", "3988.HK", "0941.HK", "0883.HK", "0857.HK",
        "1088.HK", "0016.HK", "0066.HK", "1211.HK", "2015.HK", "0175.HK",
        "2020.HK", "2269.HK", "2313.HK", "0688.HK", "0002.HK", "0003.HK",
    ]


def yra_etf(t):
    if t.startswith("IE00") or t.startswith("LU0"):
        return True
    return t in {
        "2800.HK", "2801.HK", "2802.HK", "2823.HK", "3188.HK", "3067.HK",
        "3033.HK", "2828.HK", "3037.HK", "9067.HK",
        "SPY", "QQQ", "IVV", "VOO", "VTI", "DIA",
    }



# ============================================================
# PAGRINDINE ANALIZE
# ============================================================

def analizuoti(tickeriai, progress, statusas_txt, taikyti_absoliutu_filtra=True,
               ma_filtras=MA_FILTRAS, kartele=0.0, leisti_raudona=LEISTI_RAUDONA,
               min_r2_riba=MIN_R2, tik_perkami=True):
    zali = []
    atmesti = {}
    visi_rodikliai = {}   # v3.2 (6A): kainų rodikliai KIEKVIENAM tickeriui — išėjimo taisyklei
    stats = {"ok": 0, "istorija": 0, "klaida": 0, "veto": 0, "etf": 0,
             "absoliutus": 0, "likvidumas": 0, "luzis": 0, "kokybe": 0,
             "duomenys": 0, "r2": 0, "rsi": 0, "nauji": 0, "kokybes_vartai": 0}

    def atmesti_i(kategorija, tickeris, detale, pavadinimas=None):
        atmesti.setdefault(kategorija, []).append({
            "Ticker": tickeris, "Pavadinimas": pavadinimas or "", "Priežastis": detale
        })
    klaidu_zurnalas = []
    regionu_stats = {}
    total = len(tickeriai)

    def zymeti(regionas, ivykis):
        d = regionu_stats.setdefault(regionas, {})
        d[ivykis] = d.get(ivykis, 0) + 1

    fx_cache = {}
    for val in set(nustatyti_valiuta(t) for t in tickeriai):
        fx_cache[val] = gauti_fx_serija(val)

    # --- v3.2 (4A): kainos visiems iš karto, paketais ---
    kandidatai = [t for t in tickeriai if not yra_etf(t)]
    stats["etf"] = total - len(kandidatai)
    if tik_perkami:
        neperkami = [t for t in kandidatai if nustatyti_regiona(t) in ("JP", "HK")]
        for t in neperkami:
            atmesti_i("🚫 Neperkami T212 (tik CFD)", t, "Japonijos/HK akcijos siūlomos tik kaip CFD")
        kandidatai = [t for t in kandidatai if t not in set(neperkami)]
    istorijos = gauti_kainu_istorijas(kandidatai, statusas_txt)
    siandien_ts = pd.Timestamp(dt.date.today())

    def atnaujinti(i):
        statusas_txt.text(
            f"⏳ {i+1}/{total} | ✅ {stats['ok']} | "
            f"📉 Absol.: {stats['absoliutus']} | 🔻 Lūžis: {stats['luzis']} | "
            f"💧 Likv.: {stats['likvidumas']} | 🧹 Kokybė: {stats['kokybe']} | "
            f"🩹 Duomenys: {stats['duomenys']} | 〰️ R²: {stats['r2']} | "
            f"🔵 RSI: {stats['rsi']} | 🌱 Nauji: {stats['nauji']} | "
            f"⛔ Fund.veto: {stats['veto']} | ⛔ Kok.vartai: {stats['kokybes_vartai']} | "
            f"📊 Trumpa: {stats['istorija']} | ❌ Klaidos: {stats['klaida']}"
        )

    total = max(len(kandidatai), 1)
    for i, tickeris in enumerate(kandidatai):
        try:
            regionas = nustatyti_regiona(tickeris)
            valiuta = nustatyti_valiuta(tickeris)

            duomenys = istorijos.get(tickeris)
            if duomenys is None or duomenys.empty or "Close" not in duomenys.columns:
                stats["istorija"] += 1
                continue

            duomenys = duomenys.copy()
            duomenys.index = normalizuoti_indeksa(pd.DatetimeIndex(duomenys.index))
            duomenys = duomenys[~duomenys.index.duplicated(keep="last")]

            # ignoruoti šiandienos nebaigtą žvakę
            if len(duomenys) and duomenys.index[-1] >= siandien_ts:
                duomenys = duomenys.iloc[:-1]

            kainos_vietine = duomenys["Close"].dropna()
            if tickeris.endswith(PENSU_PRIESAGOS):
                kainos_vietine = kainos_vietine / 100.0

            reikia = max(MIN_ISTORIJOS_DIENU, MOMENTUM_DIENOS + GLODINIMO_DIENOS,
                         (ma_filtras + 10) if taikyti_absoliutu_filtra else 0)
            if len(kainos_vietine) < reikia:
                if len(kainos_vietine) >= MOMENTUM_DIENOS + GLODINIMO_DIENOS:
                    stats["nauji"] += 1
                    atmesti_i("🌱 Nauji vardai (< 1 metų istorijos)", tickeris,
                              f"tik {len(kainos_vietine)} d. istorijos")
                else:
                    stats["istorija"] += 1
                continue

            fx_vietine = fx_cache.get(valiuta)
            if valiuta != BAZINE_VALIUTA and (fx_vietine is None or not len(fx_vietine)):
                # v3.2 (10): be kurso negalim nei kainų, nei kapitalizacijos į GBP — ❓
                stats["duomenys"] += 1
                zymeti(regionas, f"nėra {valiuta} kurso")
                atmesti_i("🩹 Duomenų problemos", tickeris, f"nėra GBP/{valiuta} kurso")
                continue
            kainos = i_gbp(kainos_vietine, fx_vietine)

            # --- v3.2 (9): visi kainų filtrai vienoje vietoje (analize.kainu_filtrai) ---
            rod, priezastis, kategorija = kainu_filtrai(
                kainos, kainos_vietine, kartele, ma_filtras, min_r2_riba,
                taikyti_absoliutu_filtra,
            )
            if rod is not None:
                visi_rodikliai[tickeris] = rod
            if kategorija == "istorija":
                stats["istorija"] += 1
                continue
            if kategorija in ("absoliutus", "absoliutus_ma"):
                stats["absoliutus"] += 1
                zymeti(regionas, f"m3 <= {kartele:.1f}%" if kategorija == "absoliutus" else f"po {ma_filtras}MA")
                atmesti_i("📉 Nelenkia indekso", tickeris, priezastis)
                continue
            if kategorija == "r2":
                stats["r2"] += 1
                zymeti(regionas, "R² per žemas")
                atmesti_i("〰️ Nėra tvarkingo trendo (kietas vartas)", tickeris, priezastis)
                continue
            if kategorija == "luzis":
                stats["luzis"] += 1
                zymeti(regionas, "lūžis")
                atmesti_i("🔻 Trendas lūžta", tickeris, priezastis)
                continue

            kaina_dabar = rod["kaina"]
            m1, m3, m6, m12 = rod["m1"], rod["m3"], rod["m6"], rod["m12"]
            r2, ma, rsi = rod["r2"], rod["ma"], rod["rsi"]

            apyvarta = None
            if "Volume" in duomenys.columns:
                vol = duomenys["Volume"].dropna()
                if len(vol) >= 30:
                    vol_suderinta = vol.reindex(kainos.index)
                    apyvarta_serija = (kainos * vol_suderinta).dropna()
                    if len(apyvarta_serija) >= 30:
                        apyvarta = float(apyvarta_serija.iloc[-30:].mean())
            if apyvarta is not None and apyvarta < MIN_APYVARTA_GBP:
                stats["likvidumas"] += 1
                zymeti(regionas, "likvidumas")
                atmesti_i("💧 Per maža apyvarta", tickeris,
                          f"£{apyvarta:,.0f}/d. (min £{MIN_APYVARTA_GBP:,})")
                continue

            if kategorija == "duomenys":
                stats["duomenys"] += 1
                zymeti(regionas, f"duomenys: {priezastis}")
                atmesti_i("🩹 Duomenų problemos", tickeris, priezastis)
                continue
            volatilumas = rod["vol"]
            nuo_virsunes = rod["nuo52"]

            # --- fundamentai (tik praėjusiems kainų filtrus; cache 24 val.) ---
            info = gauti_info(tickeris)
            f = istraukti_fundamentalus(info)

            sektorius = f.get("sektorius")
            kapitalizacija = f.get("market_cap")

            reikia_sektoriaus = REIKALAUTI_SEKTORIAUS and regionas not in ("JP", "HK")
            if reikia_sektoriaus and (not sektorius or sektorius == "—"):
                stats["kokybe"] += 1
                zymeti(regionas, "nėra sektoriaus")
                atmesti_i("🧹 Nėra fundamentų duomenų", tickeris, "nėra sektoriaus")
                continue

            if kapitalizacija is None or kapitalizacija <= 0:
                stats["kokybe"] += 1
                zymeti(regionas, "nėra kapitalizacijos")
                atmesti_i("🧹 Nėra fundamentų duomenų", tickeris, "nėra kapitalizacijos")
                continue

            if valiuta == BAZINE_VALIUTA:
                kapitalizacija_gbp = kapitalizacija
            else:
                kapitalizacija_gbp = kapitalizacija / float(fx_vietine.iloc[-1])

            if kapitalizacija_gbp < MIN_KAPITALIZACIJA_GBP:
                stats["kokybe"] += 1
                zymeti(regionas, "maža kapitalizacija")
                atmesti_i("🧹 Per maža įmonė", tickeris,
                          f"£{kapitalizacija_gbp/1e6:,.0f} mln. "
                          f"(min £{MIN_KAPITALIZACIJA_GBP/1e6:,.0f} mln.)",
                          f.get("pavadinimas"))
                continue

            fund_balas, ispejimai, veto, fund_statusas = vertinti_fundamentalus(f, regionas)
            if veto:
                stats["veto"] += 1
                atmesti_i("🔴 Rizikingi fundamentai", tickeris,
                          "FCF<0 kartu su skola >3x", f.get("pavadinimas"))
                continue

            if not leisti_raudona and fund_statusas == "raudona":
                priez = ", ".join(w for w in ispejimai if w.startswith(("📉", "🔴")))
                stats["veto"] += 1
                atmesti_i("🔴 Rizikingi fundamentai", tickeris,
                          priez or "keli prasti rodikliai", f.get("pavadinimas"))
                continue

            if rsi is not None and rsi > MAX_RSI_PORTFELIUI:
                stats["rsi"] += 1
                atmesti_i("🔵 Perkaitusios (RSI)", tickeris,
                          f"RSI {rsi} (riba {MAX_RSI_PORTFELIUI:.0f})",
                          f.get("pavadinimas"))
                continue

            # --- HIBRIDINIAI KOKYBĖS VARTAI (v3.2: info perduodamas, IC praleidžiamas finansams) ---
            kokybe = gauti_kokybes_zaliavas(
                tickeris, f.get("free_cash_flow"), info.get("netIncomeToCommon"),
                tikrinti_ic=not yra_finansu_sektorius(sektorius),
            )
            kietas_atmestas, kokybes_priezastis, kokybes_bauda, kokybes_zenklai, kokybe_truksta = \
                vertinti_kokybes_vartus(kokybe, r2, r2_kieta_riba=min_r2_riba, sektorius=sektorius)

            if kietas_atmestas:
                stats["kokybes_vartai"] += 1
                atmesti_i("⛔ Kietieji kokybės vartai", tickeris,
                          kokybes_priezastis, f.get("pavadinimas"))
                continue

            pe = f.get("pe")
            pe = round(pe, 1) if isinstance(pe, (int, float)) and pe > 0 else None

            zali.append({
                "Ticker": tickeris,
                "Pavadinimas": f.get("pavadinimas", "—"),
                "Kaina": round(kaina_dabar, 2),
                "Nuo52v%": nuo_virsunes,
                "Reg": regionas,
                "Val": valiuta,
                "Sektorius": f.get("sektorius", "—"),
                "Pramonė": f.get("pramone", "—"),
                "m1": None if m1 is None else round(m1, 1),
                "m3": round(m3, 1),
                "R2": r2,
                "m6": None if m6 is None else round(m6, 1),
                "m12": None if m12 is None else round(m12, 1),
                "RSI": rsi,
                "Vol%": volatilumas,
                "MA": round(ma, 2) if ma else None,
                "P/E": pe,
                "Apyvarta": apyvarta,
                "FundBalas": fund_balas,
                "Įspėjimai": ispejimai,
                "Statusas": fund_statusas,
                "Anomalija": m3 > ANOMALIJOS_RIBA,
                "Kokybe": kokybe,
                "KokybesBauda": kokybes_bauda,
                "KokybesZenklai": kokybes_zenklai,
                "KokybeTruksta": kokybe_truksta,
            })
            stats["ok"] += 1
            zymeti(regionas, "✅ praėjo")

        except Exception as e:
            stats["klaida"] += 1
            if len(klaidu_zurnalas) < 40:
                klaidu_zurnalas.append(f"{tickeris}: {type(e).__name__}: {e}")
        finally:
            progress.progress((i + 1) / total)
            atnaujinti(i)

    statusas_txt.text(
        f"✅ Baigta! Kandidatų: {stats['ok']} | "
        f"Absoliutus filtras: {stats['absoliutus']} | Lūžis: {stats['luzis']} | "
        f"Likvidumas: {stats['likvidumas']} | Kokybė: {stats['kokybe']} | "
        f"Sugadinti duomenys: {stats['duomenys']} | Nėra trendo (R²): {stats['r2']} | "
        f"Perkaitę: {stats['rsi']} | Nauji vardai: {stats['nauji']} | "
        f"Fund.veto: {stats['veto']} | Kok.vartai: {stats['kokybes_vartai']} | "
        f"Trumpa istorija: {stats['istorija']} | Klaidos: {stats['klaida']}"
    )
    stats["_klaidu_zurnalas"] = klaidu_zurnalas
    stats["_regionai"] = regionu_stats
    stats["_atmesti"] = atmesti
    stats["_rodikliai"] = visi_rodikliai
    stats["_ma_filtras"] = ma_filtras
    return zali, stats


def skaiciuoti_score(rezultatai, vol_metodas=None, rsi_premija=RSI_PREMIJA):
    """
    BALAS = analize.bazinis_score (glodintas m3 × R², m6, m12, fund — z-balai,
    minus kokybės bauda) + pasirinktinai RSI premija.
    """
    if not rezultatai:
        return rezultatai
    rezultatai = bazinis_score(rezultatai, su_fundamentais=True)
    if rsi_premija:
        for r in rezultatai:
            rsi = r.get("RSI")
            if rsi is None:
                continue
            s = r["Score"]
            if 50 <= rsi <= 65:
                s += 0.15
            elif 40 <= rsi < 50:
                s += 0.05
            elif rsi > 75:
                s -= 0.25
            elif rsi < 35:
                s -= 0.10
            r["Score"] = round(float(s), 3)
        rezultatai.sort(key=lambda x: x["Score"], reverse=True)
    return rezultatai


def sudaryti_atsargas(portfelis, visi_rezultatai, n=ATSARGU_SKAICIUS):
    """
    v3.1: Kiekvienai TOP pozicijai suranda iki n sekančių geriausių
    kandidatų iš TOS PAČIOS pramonės (arba, jei nepakanka, to paties
    sektoriaus), kurie nepateko į portfelį. Naudojama, jei pagrindinė
    pozicija vėliau atmetama (❓ duomenų trūkumas arba kietas vartas)
    arba tiesiog norisi alternatyvos palyginimui.
    """
    portfelio_tickeriai = {r["Ticker"] for r in portfelis}
    atsargos = {}
    for r in portfelis:
        pram = r.get("Pramonė")
        sekt = r.get("Sektorius")

        tos_pramones = [
            k for k in visi_rezultatai
            if k["Ticker"] not in portfelio_tickeriai and k.get("Pramonė") == pram
        ]
        if len(tos_pramones) < n:
            papildomi = [
                k for k in visi_rezultatai
                if k["Ticker"] not in portfelio_tickeriai
                and k.get("Sektorius") == sekt
                and k not in tos_pramones
            ]
            tos_pramones += papildomi

        tos_pramones.sort(key=lambda x: -x["Score"])
        # pasalinam dublikatus, islaikant tvarka
        matyti = set()
        unikalus = []
        for k in tos_pramones:
            if k["Ticker"] not in matyti:
                unikalus.append(k)
                matyti.add(k["Ticker"])
        atsargos[r["Ticker"]] = unikalus[:n]
    return atsargos


def atrinkti_portfeli(rezultatai, n=POZICIJU_SKAICIUS, lubos=SEKTORIAUS_LUBOS,
                      pramones_lubos=PRAMONES_LUBOS, tikrinti_earnings=True,
                      atmesti=None):
    portfelis, praleisti = [], []
    sekt_skaitliukas = {}
    pram_skaitliukas = {}
    dabar = pd.Timestamp.now(tz="UTC")
    if atmesti is None:
        atmesti = {}

    def i_atmestus(kategorija, r, detale):
        atmesti.setdefault(kategorija, []).append({
            "Ticker": r["Ticker"],
            "Pavadinimas": r.get("Pavadinimas", ""),
            "Priežastis": detale,
        })

    for r in rezultatai:
        if len(portfelis) >= n:
            break

        if r["Anomalija"] and not RODYTI_ANOMALIJAS:
            praleisti.append((r, f"anomalija >{ANOMALIJOS_RIBA:.0f}%/3mėn"))
            i_atmestus("🚀 Ekstremalus augimas", r, f"3 mėn. +{r['m3']}%")
            continue

        sekt = r.get("Sektorius") or "—"
        pram = r.get("Pramonė") or "—"

        if sekt != "—" and sekt_skaitliukas.get(sekt, 0) >= lubos:
            praleisti.append((r, f"sektoriaus lubos ({lubos}): {sekt}"))
            i_atmestus("🏭 Sektoriaus/pramonės lubos", r,
                       f"sektorius {sekt} jau turi {lubos}")
            continue

        if pram != "—" and pram_skaitliukas.get(pram, 0) >= pramones_lubos:
            praleisti.append((r, f"pramonės lubos ({pramones_lubos}): {pram}"))
            i_atmestus("🏭 Sektoriaus/pramonės lubos", r,
                       f"pramonė {pram} jau turi {pramones_lubos}")
            continue

        if tikrinti_earnings:
            ed = gauti_earnings(r["Ticker"])
            if ed is not None:
                try:
                    ed_tz = ed.tz_localize("UTC") if ed.tzinfo is None else ed
                    dienos = (ed_tz - dabar).days
                    if 0 <= dienos <= 60:
                        r["EarningsData"] = ed.strftime("%m-%d")
                        r["EarningsDienos"] = dienos
                    if 0 <= dienos <= EARNINGS_DIENOS:
                        praleisti.append((r, f"⚡ ataskaita {ed.strftime('%m-%d')} (po {dienos} d.)"))
                        i_atmestus("⚡ Ataskaita artėja", r,
                                   f"{ed.strftime('%m-%d')} — po {dienos} d.")
                        continue
                except Exception:
                    pass

        portfelis.append(r)
        sekt_skaitliukas[sekt] = sekt_skaitliukas.get(sekt, 0) + 1
        pram_skaitliukas[pram] = pram_skaitliukas.get(pram, 0) + 1

    return portfelis, praleisti


def gauti_earnings(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            data = cal.get("Earnings Date")
            if data and len(data) > 0:
                return pd.Timestamp(data[0])
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            return pd.Timestamp(cal.loc["Earnings Date"].iloc[0])
    except Exception:
        pass
    return None


# ============================================================
# EKSPORTAS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def gauti_rinkos_konteksta():
    """
    v3.2 (8): indeksų 3 mėn. grąža skaičiuojama TA PAČIA glodinta formule
    (skaiciuoti_momentum) ir be šiandienos nebaigtos žvakės — kaip akcijoms.
    """
    kontekstas = {}
    fx_usd = gauti_fx_serija("USD")
    fx_eur = gauti_fx_serija("EUR")
    siandien_ts = pd.Timestamp(dt.date.today())
    for pav, tk, fx in [("S&P 500", "^GSPC", fx_usd),
                        ("Nasdaq", "^IXIC", fx_usd),
                        ("STOXX 50", "^STOXX50E", fx_eur)]:
        try:
            d = yf.Ticker(tk).history(period="1y", auto_adjust=True)
            k = d["Close"].dropna()
            k.index = normalizuoti_indeksa(k.index)
            k = k[~k.index.duplicated(keep="last")]
            if len(k) and k.index[-1] >= siandien_ts:
                k = k.iloc[:-1]
            if len(k) < MOMENTUM_DIENOS + GLODINIMO_DIENOS:
                continue
            k = i_gbp(k, fx)
            m3, _ = skaiciuoti_momentum(k)
            m6 = paprastas_momentum(k, 126)
            if m3 is None:
                continue
            kontekstas[pav] = {"m3": m3, "m6": round(m6, 1) if m6 is not None else None}
        except Exception:
            pass
    return kontekstas


def gauti_ankstesni_skenavima(pavadinimas):
    try:
        katalogas = _kelias(EKSPORTO_KATALOGAS)
        if not os.path.isdir(katalogas):
            return None, None
        siandien = dt.date.today().isoformat()
        failai = sorted(
            f for f in os.listdir(katalogas)
            if f.endswith(".csv") and pavadinimas in f and not f.startswith(siandien)
        )
        if not failai:
            return None, None
        naujausias = failai[-1]
        df = pd.read_csv(os.path.join(katalogas, naujausias))
        return {
            "vietos": dict(zip(df["ticker"], df["vieta"])),
            "portfelyje": set(df.loc[df["portfelyje"] == "TAIP", "ticker"]),
        }, naujausias.split("_")[0]
    except Exception:
        return None, None


def pokycio_zyme(tickeris, vieta, ankstesnis):
    if not ankstesnis:
        return ""
    sena = ankstesnis["vietos"].get(tickeris)
    if sena is None:
        return "🆕"
    sk = int(sena) - vieta
    return f"↑{sk}" if sk > 0 else (f"↓{abs(sk)}" if sk < 0 else "=")


def velyva_stadija(m3, m6):
    if m3 is None or m6 is None or m6 <= 0:
        return False, None
    dalis = m3 / m6 * 100
    return dalis < 33.0, round(dalis, 0)


def skaiciuoti_koncentracija(portfelis):
    sekt, pram = {}, {}
    for r in portfelis:
        s = r.get("Sektorius") or "—"
        p = r.get("Pramonė") or "—"
        sekt[s] = sekt.get(s, 0) + 1
        pram[p] = pram.get(p, 0) + 1
    return (sorted(sekt.items(), key=lambda x: -x[1]),
            sorted(pram.items(), key=lambda x: -x[1]))


def eksportuoti_csv(rezultatai, portfelis, pavadinimas):
    data_str = dt.date.today().isoformat()
    eilutes = []
    portfelio_tickeriai = {r["Ticker"] for r in portfelis}

    for vieta, r in enumerate(rezultatai[:50], start=1):
        eilutes.append({
            "data": data_str,
            "skenavimas": pavadinimas,
            "vieta": vieta,
            "ticker": r["Ticker"],
            "pavadinimas": r.get("Pavadinimas"),
            "kaina_gbp": r.get("Kaina"),
            "portfelyje": "TAIP" if r["Ticker"] in portfelio_tickeriai else "",
            "statusas": r.get("Statusas"),
            "score": r["Score"],
            "sektorius": r.get("Sektorius"),
            "pramone": r.get("Pramonė"),
            "regionas": r["Reg"],
            "valiuta": r["Val"],
            "m1_gbp": r["m1"],
            "m3_gbp": r["m3"],
            "m6_gbp": r["m6"],
            "m12_gbp": r["m12"],
            "rsi": r["RSI"],
            "volatilumas": r["Vol%"],
            "r2": r.get("R2"),
            "trendo_balas": r.get("TrendoBalas"),
            "nuo_52sav_virsunes": r.get("Nuo52v%"),
            "ataskaita": r.get("EarningsData"),
            "pe": r["P/E"],
            "fund_balas": r["FundBalas"],
            "interest_coverage": (r.get("Kokybe") or {}).get("interest_coverage"),
            "fcf_ni": (r.get("Kokybe") or {}).get("fcf_ni"),
            "akciju_pokytis_12m": (r.get("Kokybe") or {}).get("shares_change_12m"),
            "kokybes_bauda": r.get("KokybesBauda"),
            "kokybe_truksta": r.get("KokybeTruksta"),
        })

    df = pd.DataFrame(eilutes)
    try:
        os.makedirs(_kelias(EKSPORTO_KATALOGAS), exist_ok=True)
        kelias = os.path.join(_kelias(EKSPORTO_KATALOGAS), f"{data_str}_{pavadinimas}.csv")
        df.to_csv(kelias, index=False, encoding="utf-8-sig")
    except Exception:
        kelias = None
    return df, kelias


# ============================================================
# RODYMAS
# ============================================================

def rodyti_diagnostika(stats):
    regionai = stats.get("_regionai") or {}
    if regionai:
        with st.expander("🌍 Kodėl regionai atrenkami nevienodai", expanded=False):
            eilutes = []
            for reg in sorted(regionai.keys()):
                d = regionai[reg]
                bendrai = sum(d.values())
                praejo = d.get("✅ praėjo", 0)
                eil = {"Regionas": reg, "Iš viso": bendrai, "Praėjo": praejo,
                       "Praėjo %": f"{praejo/bendrai*100:.0f}%" if bendrai else "—"}
                for k, v in sorted(d.items(), key=lambda x: -x[1]):
                    if k != "✅ praėjo":
                        eil[k] = v
                eilutes.append(eil)
            st.dataframe(pd.DataFrame(eilutes).fillna(0), hide_index=True, **DF_KW)
            st.caption(
                "Jei Europos „Praėjo %\" gerokai žemesnis nei JAV — priežastis matoma "
                "stulpeliuose. Dažniausiai tai absoliutus filtras (Europa atsilieka) "
                "arba trūkstami fundamentų duomenys."
            )

    zurnalas = stats.get("_klaidu_zurnalas") or []
    if stats.get("klaida", 0) > 0:
        with st.expander(f"🔧 Diagnostika — {stats['klaida']} klaidų", expanded=False):
            if stats["klaida"] > 50:
                st.warning(
                    "⚠️ Daug klaidų. Jei jos vienodo tipo — tai kodo problema, "
                    "ne rinkos. Jei skirtingos — greičiausiai Yahoo apribojimai."
                )
            for eil in zurnalas:
                st.code(eil, language=None)


def rodyti_rezultatus(rezultatai, stats, pavadinimas, tikrinti_earnings=True,
                      leisti_raudona=LEISTI_RAUDONA, vol_metodas=None,
                      rsi_premija=RSI_PREMIJA):
    rodyti_diagnostika(stats)

    if not rezultatai:
        st.error("⚠️ Nė viena akcija nepraėjo filtrų.")
        st.info(
            "**Tai gali būti teisingas atsakymas.** Absoliutus filtras "
            "(3 mėn. grąža > 0 IR kaina virš slankiojo vidurkio) atmeta viską "
            "krentančioje rinkoje. Sistema sako: šį mėnesį nepirk arba pirk mažiau."
        )
        return

    rezultatai = skaiciuoti_score(rezultatai, vol_metodas, rsi_premija)
    # v3.2: kopija — rodymas gali kartotis (session_state), atmestieji nesidubliuoja
    atmesti = {k: list(v) for k, v in (stats.get("_atmesti") or {}).items()}
    rodikliai = stats.get("_rodikliai") or {}
    ma_filtras_sken = stats.get("_ma_filtras", MA_FILTRAS)
    portfelis, praleisti = atrinkti_portfeli(
        rezultatai, tikrinti_earnings=tikrinti_earnings, atmesti=atmesti
    )
    portfelio_tickeriai = {r["Ticker"] for r in portfelis}
    praleistu_priezastys = {r["Ticker"]: p for r, p in praleisti}

    # v3.1: atsarginės pozicijos kiekvienai TOP vietai
    atsargos = sudaryti_atsargas(portfelis, rezultatai)

    kontekstas = gauti_rinkos_konteksta()
    if kontekstas:
        st.markdown("##### 📐 Rinkos kontekstas — su kuo lyginti")
        cols = st.columns(len(kontekstas) + 1)
        for i, (pav, d) in enumerate(kontekstas.items()):
            cols[i].metric(f"{pav} 3 mėn.", f"{d['m3']:+.1f}%")
        med_m3 = float(np.median([r["m3"] for r in rezultatai[:POZICIJU_SKAICIUS]]))
        etalonas = kontekstas.get("S&P 500", {}).get("m3")
        delta = f"{med_m3 - etalonas:+.1f} p.p. vs S&P" if etalonas is not None else None
        cols[-1].metric("Tavo TOP mediana", f"{med_m3:+.1f}%", delta)
        if etalonas is not None and med_m3 <= etalonas:
            st.warning(
                "⚠️ **Tavo atranka nelenkia indekso.** Z-balas yra santykinis — "
                "aukštas balas silpnoje aibėje reiškia geriausias iš prastų, "
                "o ne stiprus signalas. Apsvarstyk mažesnę ekspoziciją."
            )

    ankstesnis, ankstesne_data = gauti_ankstesni_skenavima(pavadinimas.replace(" ", "_"))

    rodomi = portfelis
    st.subheader(f"🏆 PORTFELIS — {len(rodomi)} pozicijos (TOP {POZICIJU_SKAICIUS})")
    st.success(
        "✅ **Visi žemiau esantys vardai praėjo VISUS filtrus** (įskaitant v3.1 "
        "hibridinius kokybės vartus). Perkaitusios, rizikingos, su artėjančiomis "
        "ataskaitomis ar prastais duomenimis jau atmestos — jas rasi žemiau, "
        "suskirstytas pagal priežastį."
    )
    antraste = (
        f"Kandidatų praėjo filtrus: **{len(rezultatai)}** · "
        f"Portfelis: **{len(portfelis)}** poz. po {100/max(len(portfelis),1):.1f}% · "
        f"Metodas: glodintas 3 mėn. momentum × R² − kokybės bauda"
    )
    if ankstesne_data:
        antraste += f" · Palyginta su **{ankstesne_data}**"
    st.caption(antraste)

    if len(portfelis) < POZICIJU_SKAICIUS:
        st.warning(
            f"⚠️ Portfeliui rasta tik {len(portfelis)} iš {POZICIJU_SKAICIUS}. "
            "Laikyk mažiau pozicijų — likusį kapitalą palik grynais."
        )

    sekt_sarasas, pram_sarasas = skaiciuoti_koncentracija(portfelis)
    if sekt_sarasas:
        st.markdown("##### 🎨 Sektorių pasiskirstymas portfelyje")
        n = max(len(portfelis), 1)
        eilutes_html = []
        for sektorius, kiekis in sekt_sarasas:
            proc = kiekis / n * 100
            spalva = sektoriaus_spalva(sektorius)
            eilutes_html.append(
                f'<span style="background-color:{spalva};color:#fff;'
                f'padding:4px 10px;border-radius:4px;margin-right:6px;'
                f'display:inline-block;margin-bottom:5px;font-size:0.85rem;">'
                f'{sektorius}: <b>{kiekis}</b> ({proc:.0f}%)</span>'
            )
        st.markdown("".join(eilutes_html), unsafe_allow_html=True)

        didziausias, kiek = sekt_sarasas[0]
        if kiek / n >= 0.33:
            st.warning(
                f"⚠️ **{didziausias}: {kiek} iš {n} ({kiek/n*100:.0f}%).** "
                "Jei šie vardai kyla dėl tos pačios priežasties, tai ne "
                "diversifikacija, o viena pozicija keliuose tikeriuose. "
                "Apsvarstyk dalį pakeisti kandidatais iš kitų sektorių."
            )

        dubl_pram = [(p, k) for p, k in pram_sarasas if k > 1 and p != "—"]
        if dubl_pram:
            st.caption(
                "🏭 Tos pačios pramonės: "
                + " · ".join(f"**{p}**: {k}" for p, k in dubl_pram)
                + " — siauresnis pjūvis nei sektorius"
            )

    eilutes = []
    for i, r in enumerate(rodomi, start=1):
        spalva = SPALVOS.get(r.get("Statusas", "geltona"), "🟡")
        rsi = r["RSI"]
        z = ""
        if rsi is not None and rsi > 75:
            z += "🔵"
        elif rsi is not None and rsi < 30:
            z += "⚫"
        if r["Vol%"] is not None and r["Vol%"] > 60:
            z += "⚡"
        if r.get("Anomalija"):
            z += "🚀"
        if r.get("EarningsDienos") is not None and r["EarningsDienos"] <= 30:
            z += "📅"
        velyva, dalis = velyva_stadija(r["m3"], r.get("m6"))
        if velyva:
            z += "⏰"
        if r.get("KokybeTruksta"):
            z += "❓"

        eilutes.append({
            "#": i,
            "Δ": pokycio_zyme(r["Ticker"], i, ankstesnis),
            "St": spalva + z,
            "Ticker": r["Ticker"],
            "Pavadinimas": (r.get("Pavadinimas") or "—")[:22],
            "Sektorius": r.get("Sektorius") or "—",
            "Pramonė": (r.get("Pramonė") or "—")[:22],
            "Kaina £": r.get("Kaina"),
            "Val": r.get("Val"),
            "Score": r["Score"],
            "3m%": r["m3"],
            "R²": r.get("R2"),
            "6m%": r["m6"],
            "Nuo52v%": r.get("Nuo52v%"),
            "RSI": rsi,
            "Vol%": r["Vol%"],
            "P/E ⓘ": r["P/E"],
            "Ataskaita": r.get("EarningsData") or "",
        })

    df = pd.DataFrame(eilutes)
    for c in ["Kaina £", "Score", "3m%", "R²", "6m%", "Nuo52v%", "RSI", "Vol%", "P/E ⓘ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    st.dataframe(spalvinti_lentele(df), hide_index=True, **DF_KW)

    st.caption("**Δ**: ↑ pakilo · ↓ nukrito · 🆕 naujas · = nepakito")
    st.caption(
        "**🟢** fundamentai geri · **🟡** įspėjimas · **🔴** rizikinga · "
        "**🔵** overbought (RSI>75) · **⚫** oversold (RSI<30) · "
        "**⚡** volat.>60% · **🚀** ekstremalus augimas · **📅** ataskaita ≤30 d. · "
        "**⏰** vėlyva stadija · **❓** trūksta kokybės duomenų (IC/FCF-NI/dilucija) — "
        "žr. atsarginę poziciją žemiau"
    )
    st.caption(
        "**3m%** — glodintas 3 mėn. pokytis · **R²** — kaip TVARKINGAI kilo "
        "(kietas vartas <0,50, minkšta bauda 0,50–0,65) · **Score** = 3m%×R² "
        "z-balas MINUS kokybės bauda (Interest Coverage, dilucija, R²) · "
        "**Nuo52v%** — atstumas nuo 52 sav. viršūnės"
    )

    # ---- v3.1: ATSARGINĖS POZICIJOS ----
    with st.expander("🧰 Atsarginės pozicijos (kiekvienai TOP vietai)", expanded=False):
        st.caption(
            "Jei pagrindinė pozicija turi ❓ (trūksta kokybės duomenų) arba vėliau "
            "atmetama dėl kieto vartų pažeidimo, čia rasi kitą geriausią kandidatą "
            "iš tos pačios pramonės/sektoriaus, kuris jau praėjo visus filtrus."
        )
        for r in portfelis:
            rezervas = atsargos.get(r["Ticker"], [])
            zyme_klaustukas = " ❓" if r.get("KokybeTruksta") else ""
            st.markdown(f"**{r['Ticker']}{zyme_klaustukas}** ({r.get('Pramonė','—')}):")
            if rezervas:
                eil_txt = " · ".join(
                    f"{k['Ticker']} (Score {k['Score']:.2f})" for k in rezervas
                )
                st.caption(f"　↳ Atsargos: {eil_txt}")
            else:
                st.caption("　↳ _Nerasta tinkamo atsarginio tos pačios grupės kandidato_")

    if ankstesnis:
        nauji = [r["Ticker"] for r in portfelis if r["Ticker"] not in ankstesnis["portfelyje"]]
        iskrito = [t for t in ankstesnis["portfelyje"] if t not in portfelio_tickeriai]
        with st.expander(f"🔄 Kas pasikeitė nuo {ankstesne_data}", expanded=True):
            c1, c2 = st.columns(2)
            c1.markdown("**🆕 Nauji portfelyje**")
            c1.markdown("\n".join(f"- {t}" for t in nauji) if nauji else "_nėra_")
            c2.markdown("**❌ Iškrito iš portfelio**")
            c2.markdown("\n".join(f"- {t}" for t in iskrito) if iskrito else "_nėra_")
            if nauji or iskrito:
                st.caption(
                    f"Apyvarta: {len(nauji)}/{POZICIJU_SKAICIUS} pozicijų. "
                    "Kuo didesnė — tuo daugiau FX ir spread kaštų."
                )

    ispejimu_sarasas = []
    for r in rodomi:
        w = list(r["Įspėjimai"])
        rsi = r["RSI"]
        if rsi is not None and rsi > 75:
            w.append(f"🔵 Overbought (RSI {rsi}) — gali atsitraukti")
        if rsi is not None and rsi < 30:
            w.append(f"⚫ Oversold (RSI {rsi})")
        if r["Vol%"] is not None and r["Vol%"] > 60:
            w.append(f"⚡ Volatilumas {r['Vol%']}%")
        if r["Vol%"] is not None and r["Vol%"] > 100:
            w.append("🔴 Volatilumas ekstremalus")
        if r.get("Anomalija"):
            w.append(f"🚀 3 mėn. +{r['m3']}% — ekstremalu, tikrink priežastį")
        if r.get("Nuo52v%") is not None and r["Nuo52v%"] < -20:
            w.append(f"📉 {abs(r['Nuo52v%'])}% žemiau 52 sav. viršūnės")
        if r.get("EarningsDienos") is not None and r["EarningsDienos"] <= 30:
            w.append(f"📅 Ataskaita {r['EarningsData']} (po {r['EarningsDienos']} d.)")
        if r.get("m1") is not None and r["m1"] < 0:
            w.append(f"⚠️ 1 mėn. {r['m1']}% — trendas slopsta")
        if r.get("R2") is not None and r["R2"] < R2_MINKSTA_RIBA:
            w.append(f"〰️ R² {r['R2']:.2f} — kilimas ne visai tolygus (minkšta bauda)")
        for kz in r.get("KokybesZenklai", []):
            w.append(kz)
        if w:
            spalva = SPALVOS.get(r.get("Statusas", "geltona"), "🟡")
            zyme = "✅" if r["Ticker"] in portfelio_tickeriai else "⏳"
            ispejimu_sarasas.append(f"**{zyme} {spalva} {r['Ticker']}:** {' | '.join(w)}")

    if ispejimu_sarasas:
        with st.expander(f"⚠️ Įspėjimai ir pastabos ({len(ispejimu_sarasas)})", expanded=True):
            for t in ispejimu_sarasas:
                st.markdown(t)
            st.caption(
                "📉 Pajamos↓ = pajamos krenta >10% · 🔴 Skola aukšta = D/E >3x · "
                "⚠️ FCF<0 = neigiamas laisvas pinigų srautas · 📉 Nuostolis = ROE <-10% · "
                "❓ trūksta = nėra dalies fundamentų/kokybės duomenų · "
                "⚠️IC/⚠️Dilucija/⚠️R² = v3.1 minkšta kokybės bauda"
            )

    with st.expander("💷 Ką pirkti šį mėnesį (pagal tavo taisyklę)", expanded=False):
        st.caption(
            "Taisyklė: £100/mėn. eina pozicijai, kuri (a) vis dar TOP "
            f"{POZICIJU_SKAICIUS} ir (b) turi mažiausią svorį. "
            "Įklijuok turimas pozicijas: `TICKER,vertė` po vieną eilutėje."
        )
        # v3.2 (1B): pozicijos prisimenamos faile — kitą mėnesį nereikės vesti iš naujo
        if "turimos_tekstas" not in st.session_state:
            st.session_state["turimos_tekstas"] = "\n".join(_skaityti_atsargini(POZICIJU_FAILAS))
        tekstas = st.text_area("Turimos pozicijos", key="turimos_tekstas",
                               placeholder="PANW,31.50\nFTNT,28.10", height=110)
        imoka = st.number_input("Mėnesio įnašas (£)", min_value=10.0, value=100.0, step=10.0)
        if tekstas.strip():
            _issaugoti_atsargini(POZICIJU_FAILAS,
                                 [e.strip() for e in tekstas.strip().splitlines() if e.strip()])
            st.caption(f"💾 Pozicijos išsaugotos į `{POZICIJU_FAILAS}` — kitą kartą jau bus įvestos.")
            turimos = {}
            for eil in tekstas.strip().splitlines():
                dalys = eil.replace(";", ",").split(",")
                if len(dalys) >= 2:
                    try:
                        turimos[dalys[0].strip().upper()] = float(dalys[1].strip().replace("£", ""))
                    except ValueError:
                        pass
            if turimos:
                bendra = sum(turimos.values()) + imoka
                tikslinis = bendra / max(len(portfelis), 1)
                kvalifikuoti = [
                    (t, turimos.get(t, 0.0), tikslinis - turimos.get(t, 0.0))
                    for t in portfelio_tickeriai
                ]
                kvalifikuoti.sort(key=lambda x: -x[2])
                st.markdown(f"**Tikslinis svoris:** £{tikslinis:,.2f} pozicijai")
                st.dataframe(pd.DataFrame([
                    {"Ticker": t, "Turima £": round(v, 2), "Trūksta £": round(d, 2)}
                    for t, v, d in kvalifikuoti[:8]
                ]), hide_index=True, **DF_KW)
                if kvalifikuoti:
                    st.success(f"➡️ Šį mėnesį pirk: **{kvalifikuoti[0][0]}** "
                               f"(atsilieka £{kvalifikuoti[0][2]:,.2f})")

                visos_vietos = {r["Ticker"]: i for i, r in enumerate(rezultatai, start=1)}
                laikyti, parduoti, luzta, nera_duomenu = [], [], [], []
                for t in turimos:
                    # v3.2 (6A): IŠĖJIMO TAISYKLĖ visoms laikomoms — net TOP 8
                    rod = rodikliai.get(t)
                    if rod is None:
                        if t not in visos_vietos:
                            nera_duomenu.append(t)
                    else:
                        priez = isejimo_signalas(rod, ma_filtras_sken)
                        if priez:
                            luzta.append((t, priez))
                            continue
                    if t in portfelio_tickeriai:
                        continue
                    vieta = visos_vietos.get(t)
                    if vieta is not None and vieta <= LIPNUMO_RIBA:
                        laikyti.append((t, vieta))
                    else:
                        parduoti.append((t, vieta))

                if luzta:
                    st.error(
                        "🛑 **Parduoti — išėjimo taisyklė** (ta pati, kaip įeinant: kaina "
                        f"žemiau {ma_filtras_sken} d. vidurkio arba 1 mėn. < {M1_LUZIO_RIBA:.0f} %): "
                        + " · ".join(f"**{t}** ({p})" for t, p in luzta)
                    )
                if nera_duomenu:
                    st.caption("ℹ️ Nėra šio skenavimo duomenų (ne šiame universume?): "
                               + ", ".join(nera_duomenu))

                if laikyti:
                    st.info(
                        "🤝 **Laikyti (lipnumo taisyklė)** — iškrito iš TOP "
                        f"{POZICIJU_SKAICIUS}, bet dar TOP {LIPNUMO_RIBA}: "
                        + ", ".join(f"**{t}** ({v} vieta)" for t, v in laikyti)
                        + ". Naujų pinigų neduodam, bet ir neparduodam — "
                        "pozicija turi realiai susilpnėti, ne tik pastumti per ribą."
                    )
                if parduoti:
                    st.error(
                        f"❌ **Parduoti** — iškrito iš TOP {LIPNUMO_RIBA}: "
                        + ", ".join(
                            f"**{t}**" + (f" ({v} vieta)" if v else " (nebe kandidatas)")
                            for t, v in parduoti
                        )
                    )
                if not laikyti and not parduoti and not luzta and turimos:
                    st.success("✅ Visos turimos pozicijos vis dar TOP "
                               f"{POZICIJU_SKAICIUS} — parduoti nieko nereikia.")

    with st.expander("📋 Portfelio sąrašas (kopijuoti naujienų patikrai)", expanded=False):
        st.caption(
            "Nukopijuok ir atsiųsk Claude — patikrins kiekvieną vardą keliuose "
            "šaltiniuose ir grąžins tik tai, kas sutampa (su data ir šaltiniu)."
        )
        eilutes_txt = [f"{i}. {r['Ticker']} — {r.get('Pavadinimas','')} "
                       f"({r.get('Sektorius','')})"
                       for i, r in enumerate(portfelis, start=1)]
        st.code("\n".join(eilutes_txt), language=None)

    if atmesti:
        bendrai = sum(len(v) for v in atmesti.values())
        st.markdown(f"##### 🗂️ Atmesta: {bendrai} vardų")
        st.caption(
            "Kiekviena skiltis — atskira priežastis. Jei prireiks pakaitalo ar "
            "norėsi pasidomėti, viskas išsaugota, ne dingę."
        )
        eiliskumas = sorted(
            atmesti.items(),
            key=lambda x: (x[0].startswith("🌱"), -len(x[1]))
        )
        for kategorija, sarasas in eiliskumas:
            with st.expander(f"{kategorija} — {len(sarasas)}"):
                df_a = pd.DataFrame(sarasas)
                if "Pavadinimas" in df_a.columns:
                    df_a["Pavadinimas"] = df_a["Pavadinimas"].fillna("").str[:26]
                st.dataframe(df_a.head(60), hide_index=True, **DF_KW)
                if len(sarasas) > 60:
                    st.caption(f"Rodoma 60 iš {len(sarasas)}")

    # ---- v3.1: PLATESNIS SĄRAŠAS — praėjo bazinius filtrus, bet ne TOP N ----
    with st.expander(f"📋 Visi kandidatai, praėję filtrus ({min(len(rezultatai), 100)}) "
                     "— tolimesnei analizei/statistikai"):
        st.caption(
            "Šie vardai praėjo visus filtrus (įskaitant kietuosius kokybės vartus), "
            f"bet nepateko į TOP {POZICIJU_SKAICIUS} dėl sektoriaus/pramonės lubų arba "
            "žemesnio score. Spalvos tos pačios — ieškant pakaitalo iškart matai sektorių."
        )
        df_visi = pd.DataFrame([{
            "St": SPALVOS.get(r.get("Statusas", "geltona"), "🟡") + ("❓" if r.get("KokybeTruksta") else ""),
            "Ticker": r["Ticker"],
            "Pavadinimas": (r.get("Pavadinimas") or "—")[:20],
            "Sektorius": r.get("Sektorius") or "—",
            "Pramonė": (r.get("Pramonė") or "—")[:20],
            "Reg": r["Reg"],
            "Kaina £": r.get("Kaina"),
            "Score": r["Score"],
            "1m%": r["m1"], "3m%": r["m3"], "R²": r.get("R2"),
            "6m%": r["m6"], "12m%": r["m12"], "Nuo52v%": r.get("Nuo52v%"),
            "RSI": r["RSI"], "Vol%": r["Vol%"], "P/E": r["P/E"],
        } for r in rezultatai[:100]])
        for c in ["Kaina £", "Score", "1m%", "3m%", "R²", "6m%", "12m%",
                  "Nuo52v%", "RSI", "Vol%", "P/E"]:
            df_visi[c] = pd.to_numeric(df_visi[c], errors="coerce")
        st.dataframe(spalvinti_lentele(df_visi), hide_index=True, **DF_KW)

    df_csv, kelias = eksportuoti_csv(rezultatai, portfelis, pavadinimas.replace(" ", "_"))
    st.download_button(
        "💾 Atsisiųsti skenavimo CSV",
        data=df_csv.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{dt.date.today().isoformat()}_{pavadinimas.replace(' ', '_')}.csv",
        mime="text/csv",
    )
    if kelias:
        st.caption(f"✅ Išsaugota: `{kelias}` — kitą kartą matysi, kas pasikeitė")


# ============================================================
# SASAJA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def gauti_rodikli(ticker):
    """v3.2 (5): rinkų skydelio rodiklis cache'inamas 15 min. — nebesiunčiama kas paspaudimą."""
    try:
        d = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        k = d["Close"].dropna()
        if len(k) >= 2:
            return float(k.iloc[-1]), round((k.iloc[-1] - k.iloc[-2]) / k.iloc[-2] * 100, 2)
        if len(k) == 1:
            return float(k.iloc[-1]), None
    except Exception:
        pass
    return None, None


with st.expander("📊 Rinkų skydelis", expanded=False):

    def rodyti_rodikli(stulpelis, pav, ticker, formatas="{:,.0f}",
                       paaiskinimas=None, atvirksciai=False):
        verte, p = gauti_rodikli(ticker)
        if verte is None:
            stulpelis.metric(pav, "—", help=paaiskinimas)
        elif p is None:
            stulpelis.metric(pav, formatas.format(verte), help=paaiskinimas)
        else:
            stulpelis.metric(
                pav, formatas.format(verte), f"{p:+.2f}%",
                delta_color="inverse" if atvirksciai else "normal",
                help=paaiskinimas,
            )

    st.markdown("**🏛️ Akcijų indeksai**")
    e1 = st.columns(4)
    rodyti_rodikli(e1[0], "🇺🇸 S&P 500", "^GSPC",
                   paaiskinimas="500 didžiausių JAV bendrovių. Tavo skanerio kartelė.")
    rodyti_rodikli(e1[1], "🇺🇸 Nasdaq 100", "^NDX",
                   paaiskinimas="Technologijų indeksas.")
    rodyti_rodikli(e1[2], "🇪🇺 STOXX 50", "^STOXX50E",
                   paaiskinimas="50 didžiausių euro zonos bendrovių.")
    rodyti_rodikli(e1[3], "🇬🇧 FTSE 100", "^FTSE",
                   paaiskinimas="100 didžiausių Londono biržos bendrovių.")

    e2 = st.columns(4)
    rodyti_rodikli(e2[0], "🇯🇵 Nikkei 225", "^N225",
                   paaiskinimas="Japonijos pagrindinis indeksas.")
    rodyti_rodikli(e2[1], "🔧 Puslaidininkiai", "^SOX",
                   paaiskinimas="Philadelphia Semiconductor Index — visos šakos "
                                "sveikata. Jei krenta visas, tai ne MU problema.")
    rodyti_rodikli(e2[2], "🏦 JAV bankai", "^BKX",
                   paaiskinimas="KBW Bank Index — finansų sektoriaus barometras.")
    rodyti_rodikli(e2[3], "🌍 Besivystančios", "EEM",
                   paaiskinimas="Besivystančių rinkų ETF.")

    st.divider()

    st.markdown("**⛽ Nafta ir žaliavos**")
    e3 = st.columns(4)
    rodyti_rodikli(e3[0], "🛢️ Nafta (Brent)", "BZ=F", "${:,.2f}",
                   paaiskinimas="Europos/pasaulinė naftos kaina. Reaguoja į "
                                "geopolitiką — Hormūzo sąsiaurį ir pan.")
    rodyti_rodikli(e3[1], "🛢️ Nafta (WTI)", "CL=F", "${:,.2f}",
                   paaiskinimas="JAV naftos kaina. Jei Brent smarkiai atsiplėšia "
                                "nuo WTI — tiekimo problema, ne paklausos šuolis.")
    rodyti_rodikli(e3[2], "🥇 Auksas", "GC=F", "${:,.0f}",
                   paaiskinimas="Saugus prieglobstis. Kyla su akcijomis = "
                                "rinkoje nervingumas, kurio VIX dar nerodo.")
    rodyti_rodikli(e3[3], "🥈 Sidabras", "SI=F", "${:,.2f}",
                   paaiskinimas="Pusiau pramoninis metalas. Lenkia auksą = "
                                "pramonės paklausa, ne baimė.")

    st.divider()

    st.markdown("**🌡️ Rizikos nuotaika**")
    e4 = st.columns(4)

    try:
        vix_v, _ = gauti_rodikli("^VIX")
        if vix_v is None:
            raise ValueError("VIX nėra")
        vix = round(float(vix_v), 1)
        if vix < 15:
            nuotaika, ikona = "Godumas", "😎"
        elif vix < 20:
            nuotaika, ikona = "Ramu", "😊"
        elif vix < 25:
            nuotaika, ikona = "Neutralu", "😐"
        elif vix < 35:
            nuotaika, ikona = "Baimė", "😟"
        else:
            nuotaika, ikona = "Panika", "😱"
        e4[0].metric(f"{ikona} Baimės indeksas", vix, nuotaika, delta_color="off",
                     help="VIX. Žemiau 15 — ramu (bet ir savimi patenkinta). "
                          "Virš 30 — panika, dažnai geriausias pirkimo metas.")
    except Exception:
        e4[0].metric("😐 Baimės indeksas", "—")

    rodyti_rodikli(e4[1], "₿ Bitcoin", "BTC-USD", "${:,.0f}",
                   paaiskinimas="Rizikos apetito rodiklis. Krenta pirmas, kai "
                                "investuotojai bėga iš rizikos.")
    rodyti_rodikli(e4[2], "📈 JAV 10m. palūk.", "^TNX", "{:,.2f}%",
                   paaiskinimas="Svarbiausias vienas skaičius rinkoje. Kylant — "
                                "technologijos krenta, bankai kyla.",
                   atvirksciai=True)
    rodyti_rodikli(e4[3], "💵 Dolerio indeksas", "DX-Y.NYB", "{:,.2f}",
                   paaiskinimas="Dolerio stiprumas. Stiprėjant — JAV akcijos "
                                "svarais atrodo geriau, nei yra iš tikrųjų.")

    st.divider()

    st.markdown("**💱 Valiutos** — svarbu, nes skaneris viską verčia į svarus")
    e5 = st.columns(3)
    rodyti_rodikli(e5[0], "💷 Svaras → doleris", "GBPUSD=X", "{:,.4f}",
                   paaiskinimas="Kiek dolerių už 1 svarą. Kylant — JAV akcijos "
                                "tau atsipigina, bet grąža svarais mažėja.")
    rodyti_rodikli(e5[1], "💶 Svaras → euras", "GBPEUR=X", "{:,.4f}",
                   paaiskinimas="Kiek eurų už 1 svarą.")
    rodyti_rodikli(e5[2], "💴 Svaras → jena", "GBPJPY=X", "{:,.2f}",
                   paaiskinimas="Kiek jenų už 1 svarą.")

    st.caption(
        "💡 Užvesk pelę ant ⓘ prie kiekvieno pavadinimo — paaiškina, ką rodiklis "
        "reiškia ir kodėl svarbus. **Baimės indeksui ir palūkanoms** kilimas rodomas "
        "raudonai, nes jiems kylant akcijoms paprastai blogiau."
    )

with st.expander("⚙️ Nustatymai", expanded=False):
    st.markdown("**Pagrindiniai**")
    absoliutus = st.checkbox(
        "Absoliutus filtras (momentum turi lenkti kartelę + kaina virš slankiojo vidurkio)",
        value=True,
        help="Išjungus skaneris visada duos TOP N, net krentančioje rinkoje."
    )
    lyginti_su_indeksu = st.checkbox(
        "Kartelė = S&P 500 grąža (o ne nulis)", value=LYGINTI_SU_INDEKSU,
        help="Momentum yra SANTYKINIS stiprumas. Akcija +1% per 3 mėn., kai "
             "indeksas +8%, yra atsilikėja. Išjungus kartelė bus 0%."
    )
    earnings_filtras = st.checkbox(
        f"Praleisti, jei ataskaita per {EARNINGS_DIENOS} d.", value=True
    )
    leisti_raudona = st.checkbox(
        "Leisti 🔴 (rizikingus fundamentus) į portfelį", value=LEISTI_RAUDONA
    )

    with st.expander("🔬 Išplėstiniai (keisti nebūtina)", expanded=False):
        ma_filtras = st.radio(
            "Trendo filtro ilgis",
            options=[100, 200, 0],
            format_func=lambda x: {100: "100 dienų (tinka 3 mėn. strategijai)",
                                   200: "200 dienų (ilgo laikotarpio)",
                                   0: "Išjungta"}[x],
            index=0,
        )
        min_r2 = st.slider(
            "Minimalus R² — KIETAS vartas (trendo tvarkingumas)", 0.0, 0.8, MIN_R2, 0.05,
            help="v3.1: numatytoji 0,50 — žemiau AUTOMATIŠKAI atmeta (kietas vartas). "
                 "Minkšta bauda balui taikoma juostai [ši riba, ši riba + 0,15) — "
                 "juosta juda kartu su slankikliu, kad visada liktų suderinta."
        )
        rsi_premija = st.checkbox("RSI premija bale", value=RSI_PREMIJA)
        vol_metodas = None

    st.caption(
        f"Pozicijų: {POZICIJU_SKAICIUS} · "
        f"Sektoriaus lubos: {SEKTORIAUS_LUBOS} · Pramonės lubos: {PRAMONES_LUBOS} · "
        f"RSI riba: {MAX_RSI_PORTFELIUI:.0f} · Min. R² (kietas): {MIN_R2} · "
        f"Min. istorijos: {MIN_ISTORIJOS_DIENU} d. · Glodinimas: {GLODINIMO_DIENOS} d. · "
        f"Min. apyvarta: £{MIN_APYVARTA_GBP:,} · "
        f"Min. kapitalizacija: £{MIN_KAPITALIZACIJA_GBP:,} · "
        f"Kokybės vartai: IC<{IC_KIETA_RIBA} kietas/{IC_MINKSTA_RIBA} minkštas · "
        f"Dilucija>{SHARES_KIETA_RIBA}% kietas/{SHARES_MINKSTA_RIBA}% minkštas"
    )

st.divider()

c1, c2 = st.columns(2)
with c1:
    btn_sp = st.button("🇺🇸 S&P 500")
    btn_eu = st.button("🌍 Europa")
with c2:
    btn_jp = st.button("🇯🇵 Japonija")
    btn_az = st.button("🌏 Azija (HK)")

btn_all = st.button("🎯 JAV + Europa (T212 prieinama)", type="primary")
btn_global = st.button("🌐 VISAS PASAULIS (JAV + Europa + Japonija + HK)")
st.caption(
    "🎯 — tik tai, ką T212 tikrai siūlo. "
    "🌐 — viskas, įskaitant Japoniją ir HK (patikrink prieinamumą prieš pirkdamas)."
)


def paleisti(tickeriai, pavadinimas, tik_perkami=True):
    if not tickeriai:
        st.error("Tickerių sąrašas tuščias.")
        return

    kartele = 0.0
    kartele_txt = None
    if lyginti_su_indeksu and absoliutus:
        kontekstas = gauti_rinkos_konteksta()
        etalonas = kontekstas.get("S&P 500", {}).get("m3")
        if etalonas is not None:
            kartele = float(etalonas) + INDEKSO_ATSARGA
            kartele_txt = (f"📐 **Kartelė: {kartele:+.1f}%** (S&P 500 glodinta 3 mėn. grąža, GBP). "
                           "Į kandidatus pateks tik akcijos, lenkiančios indeksą.")
            st.info(kartele_txt)
        else:
            st.warning("⚠️ Indekso grąžos gauti nepavyko — kartelė 0%.")

    st.info(f"Skenuojama: {len(tickeriai)} akcijų...")
    progress = st.progress(0)
    statusas = st.empty()
    rezultatai, stats = analizuoti(tickeriai, progress, statusas,
                                   absoliutus, ma_filtras, kartele, leisti_raudona,
                                   min_r2, tik_perkami)
    # v3.2 (1A): rezultatai saugomi sesijoje — nebedingsta spaudžiant/įvedant
    st.session_state["skenavimas"] = {
        "rezultatai": rezultatai,
        "stats": stats,
        "pavadinimas": pavadinimas,
        "laikas": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kartele_txt": kartele_txt,
        "nustatymai": (absoliutus, lyginti_su_indeksu, ma_filtras, min_r2, leisti_raudona),
    }
    st.rerun()


def rodyti_issaugota():
    """v3.2 (1A): piešia paskutinį skenavimą iš session_state (kiekvieną perkrovimą)."""
    sk = st.session_state.get("skenavimas")
    if not sk:
        return
    c1, c2 = st.columns([4, 1])
    c1.caption(f"🕒 Rodomas skenavimas **{sk['pavadinimas']}** ({sk['laikas']}). "
               "Gali įvesti pozicijas ar keisti rodinius — rezultatai išliks.")
    if c2.button("🗑️ Išvalyti"):
        del st.session_state["skenavimas"]
        st.rerun()
    if sk.get("kartele_txt"):
        st.info(sk["kartele_txt"])
    dabartiniai = (absoliutus, lyginti_su_indeksu, ma_filtras, min_r2, leisti_raudona)
    if dabartiniai != sk.get("nustatymai"):
        st.warning("⚙️ Nustatymai pakeisti po skenavimo — rezultatai iš senų nustatymų. "
                   "Paspausk skenavimo mygtuką iš naujo.")
    # rezultatai kopijuojami — score/rikiavimas nepakeis išsaugoto sąrašo
    rezultatai = [dict(r) for r in sk["rezultatai"]]
    rodyti_rezultatus(rezultatai, sk["stats"], sk["pavadinimas"], earnings_filtras,
                      leisti_raudona, vol_metodas, rsi_premija)


if btn_sp:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_sp500()
    paleisti(t, "S&P 500")

if btn_eu:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_europa()
    paleisti(t, "Europa")

if btn_jp:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_japonija()
    paleisti(t, "Japonija", tik_perkami=False)

if btn_az:
    paleisti(gauti_tickerius_azija(), "Azija HK", tik_perkami=False)

if btn_all:
    with st.spinner("Kraunami sąrašai..."):
        t = list(dict.fromkeys(
            gauti_tickerius_sp500() +
            gauti_tickerius_europa() +
            gauti_japonijos_adr()
        ))
    st.caption(
        f"ℹ️ Įtraukta {len(gauti_japonijos_adr())} japoniškų ADR "
        "(Toyota, Sony, Mitsubishi UFJ ir kt.) — jos kotiruojamos NYSE doleriais "
        "ir perkamos kaip tikros akcijos, ne CFD."
    )
    paleisti(t, "JAV + Europa")

if btn_global:
    with st.spinner("Kraunami visi sąrašai..."):
        t = list(dict.fromkeys(
            gauti_tickerius_sp500() +
            gauti_tickerius_europa() +
            gauti_tickerius_japonija() +
            gauti_tickerius_azija()
        ))
    st.warning(
        "⚠️ Globalus skenavimas užtruks ilgiau (~1200 tikerių). "
        "Japonijos ir HK akcijoms Yahoo dažnai neturi fundamentų duomenų, "
        "tad jų praeis mažiau — priežastis bus matoma „🌍 Kodėl regionai\" skydelyje."
    )
    paleisti(t, "Globalus", tik_perkami=False)

rodyti_issaugota()
