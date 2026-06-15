import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="Momentum Skaneris",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="📈"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    [data-testid="stMetricValue"] { font-size: 0.9rem; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem; }
    .stButton > button {
        width: 100%;
        padding: 0.6rem;
        font-size: 1rem;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Momentum Skaneris")
st.caption("🟢 Stiprus | 🟡 Įspėjimas | 🔴 Praleisk | 🔵 Overbought | ⚫ Oversold")


# ============================================
# PAGRINDINĖS FUNKCIJOS
# ============================================

def momentum_proc(kaina_pradzioje, kaina_dabar):
    if kaina_pradzioje == 0:
        return 0
    return (kaina_dabar - kaina_pradzioje) / kaina_pradzioje * 100


def skaiciuoti_rsi(kainos, periodas=14):
    pokyciai = kainos.diff()
    kilimai = pokyciai.where(pokyciai > 0, 0)
    kritimai = -pokyciai.where(pokyciai < 0, 0)
    vid_kilimas = kilimai.rolling(periodas).mean().iloc[-1]
    vid_kritimas = kritimai.rolling(periodas).mean().iloc[-1]
    if vid_kritimas == 0:
        return 100
    rs = vid_kilimas / vid_kritimas
    return round(100 - (100 / (1 + rs)), 1)


def gauti_fundamentalus(ticker_obj):
    try:
        info = ticker_obj.info
        if not info or len(info) < 5:
            return None, None, None, None
    except:
        return None, None, None, None
    revenue_growth = info.get("revenueGrowth")
    debt_to_equity = info.get("debtToEquity")
    free_cash_flow = info.get("freeCashflow")
    roe = info.get("returnOnEquity")
    return revenue_growth, debt_to_equity, free_cash_flow, roe


def vertinti_fundamentalus(revenue_growth, debt_to_equity, free_cash_flow, roe, regionas="US"):
    score_papildas = 0
    ispejimai = []
    raudoni = 0
    geltoni = 0
    truksta = 0

    if revenue_growth is None:
        truksta += 1
    elif revenue_growth > 0.15:
        score_papildas += 20
    elif revenue_growth > 0.05:
        score_papildas += 10
    elif revenue_growth < -0.10:
        score_papildas -= 15
        ispejimai.append("📉 Pajamos↓")
        raudoni += 1
    elif revenue_growth < 0:
        score_papildas -= 5
        ispejimai.append("⚠️ Pajamos↓ šiek tiek")
        geltoni += 1

    if debt_to_equity is None:
        truksta += 1
    else:
        de = debt_to_equity / 100 if debt_to_equity > 10 else debt_to_equity
        if de < 0.5:
            score_papildas += 10
        elif de < 1.5:
            score_papildas += 5
        elif de < 3.0:
            ispejimai.append("⚠️ Skola↑")
            geltoni += 1
        else:
            score_papildas -= 15
            ispejimai.append("🔴 Skola kritinė")
            raudoni += 1

    if free_cash_flow is None:
        truksta += 1
    elif free_cash_flow > 0:
        score_papildas += 10
    else:
        ispejimai.append("⚠️ FCF<0")
        geltoni += 1
        score_papildas -= 5

    if roe is None:
        truksta += 1
    elif roe > 0.15:
        score_papildas += 15
    elif roe > 0.05:
        score_papildas += 5
    elif roe < -0.10:
        ispejimai.append("📉 Nuostolis")
        raudoni += 1
        score_papildas -= 20
    elif roe < 0:
        ispejimai.append("⚠️ ROE neigiamas")
        geltoni += 1
        score_papildas -= 5

    if truksta == 4:
        return 0, ["❓ Nėra fund. duomenų"], "geltona"

    if truksta >= 2:
        ispejimai.append(f"❓ {truksta}/4 trūksta")
        if regionas not in ("JP", "HK"):
            geltoni += 1

    if raudoni >= 2:
        statusas = "raudona"
    elif raudoni == 1 and geltoni >= 2:
        statusas = "raudona"
    elif geltoni >= 3:
        statusas = "geltona"
    elif geltoni >= 1 or truksta >= 2:
        statusas = "geltona"
    else:
        statusas = "zalia"

    return score_papildas, ispejimai, statusas


def skaiciuoti_score(m1, m3, m6, m12, rsi, pe, fund_papildas):
    score = 0
    score += m6 * 0.35
    score += m3 * 0.25
    score += m12 * 0.20
    score += m1 * 0.10
    score += fund_papildas * 0.10

    if 50 <= rsi <= 65:
        score += 10
    elif 40 <= rsi < 50:
        score += 5
    elif rsi > 75:
        score -= 10
    elif rsi < 35:
        score -= 5

    if pe is not None:
        if pe < 20:
            score += 15
        elif pe < 35:
            score += 8
        elif pe > 100:
            score -= 10

    return round(score, 1)


def gauti_naujienas(ticker):
    try:
        naujienos = yf.Ticker(ticker).news
        if not naujienos:
            return False, []
        turi = any(
            n.get("providerPublishTime", 0) > pd.Timestamp.now().timestamp() - 48 * 3600
            for n in naujienos
        )
        return turi, naujienos[:3] if turi else []
    except:
        return False, []


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
    except:
        pass
    return None


def yra_etf_filtras(tickeris):
    if tickeris.startswith("IE00") or tickeris.startswith("LU0"):
        return True

    etf_sarasas = {
        "2800.HK", "2801.HK", "2802.HK", "2823.HK", "3188.HK",
        "3067.HK", "3033.HK", "2828.HK", "3037.HK", "9067.HK",
        "SPY", "QQQ", "IVV", "VOO", "VTI", "DIA",
    }
    if tickeris in etf_sarasas:
        return True

    return False


def nustatyti_regiona(tickeris):
    if tickeris.endswith(".T") or tickeris.endswith(".JP"):
        return "JP"
    elif tickeris.endswith(".HK"):
        return "HK"
    elif any(tickeris.endswith(s) for s in [
        ".L", ".DE", ".PA", ".AS", ".MC", ".MI", ".ST",
        ".CO", ".OL", ".HE", ".BR", ".LS", ".VI",
        ".WA", ".IR", ".SW", ".F"
    ]):
        return "EU"
    else:
        return "US"


def analizuoti(tickeriai, progress, statusas_txt):
    rezultatai = []
    sekmingai = 0
    praleista_klaida = 0
    praleista_istorija = 0
    praleista_raudona = 0
    praleista_etf = 0

    total = len(tickeriai)

    def atnaujinti(i):
        statusas_txt.text(
            f"⏳ {i+1}/{total} | "
            f"✅ {sekmingai} | "
            f"📊 Trumpa: {praleista_istorija} | "
            f"❌ Klaidos: {praleista_klaida} | "
            f"🔴 Filtruota: {praleista_raudona} | "
            f"🏷️ ETF: {praleista_etf}"
        )

    for i, tickeris in enumerate(tickeriai):
        if yra_etf_filtras(tickeris):
            praleista_etf += 1
            progress.progress((i + 1) / total)
            atnaujinti(i)
            continue

        regionas = nustatyti_regiona(tickeris)

        try:
            akcija = yf.Ticker(tickeris)
            duomenys = akcija.history(period="2y")

            if duomenys.empty or "Close" not in duomenys.columns:
                praleista_istorija += 1
                progress.progress((i + 1) / total)
                atnaujinti(i)
                continue

            kainos = duomenys["Close"].dropna()

            min_dienos = {
                "US": 200,
                "EU": 150,
                "JP": 150,
                "HK": 150,
            }
            reikia = min_dienos.get(regionas, 150)

            if len(kainos) < reikia:
                praleista_istorija += 1
                progress.progress((i + 1) / total)
                atnaujinti(i)
                continue

            kaina_dabar = kainos.iloc[-1]

            def saugus_momentum(dienos):
                idx = min(dienos, len(kainos) - 1)
                return momentum_proc(kainos.iloc[-idx], kaina_dabar)

            m1 = saugus_momentum(21)
            m3 = saugus_momentum(63)
            m6 = saugus_momentum(126)
            m12 = saugus_momentum(min(252, len(kainos) - 1))

            rsi = skaiciuoti_rsi(kainos)

            try:
                info = akcija.info
                pe = info.get("trailingPE")
                if pe is not None:
                    pe = round(pe, 1)
            except:
                info = {}
                pe = None

            rev_g, d2e, fcf, roe = gauti_fundamentalus(akcija)
            fund_papildas, ispejimai, fund_statusas = vertinti_fundamentalus(
                rev_g, d2e, fcf, roe, regionas=regionas
            )

            if fund_statusas == "raudona":
                praleista_raudona += 1
                progress.progress((i + 1) / total)
                atnaujinti(i)
                continue

            anomalija = m6 > 300
            score = skaiciuoti_score(m1, m3, m6, m12, rsi, pe, fund_papildas)
            sekmingai += 1

            if d2e is not None:
                de_display = d2e / 100 if d2e > 10 else d2e
                de_str = f"{round(de_display, 2)}"
            else:
                de_str = "—"

            rezultatai.append({
                "Ticker":    tickeris,
                "Regionas":  regionas,
                "1m%":       round(m1, 1),
                "3m%":       round(m3, 1),
                "6m%":       round(m6, 1),
                "12m%":      round(m12, 1),
                "P/E":       pe if pe else "—",
                "RSI":       rsi,
                "Rev":       f"{round(rev_g*100,1)}%" if rev_g is not None else "—",
                "D/E":       de_str,
                "FCF":       "✅" if fcf and fcf > 0 else ("❌" if fcf is not None and fcf <= 0 else "—"),
                "ROE":       f"{round(roe*100,1)}%" if roe is not None else "—",
                "Įspėjimai": ispejimai,
                "Statusas":  fund_statusas,
                "Score":     score,
                "Anomalija": anomalija,
            })

        except Exception:
            praleista_klaida += 1

        progress.progress((i + 1) / total)
        atnaujinti(i)

    statusas_txt.text(
        f"✅ Baigta! Įtraukta: {sekmingai} | "
        f"Trumpa istorija: {praleista_istorija} | "
        f"Klaidos: {praleista_klaida} | "
        f"Filtruota: {praleista_raudona} | "
        f"ETF: {praleista_etf}"
    )
    return rezultatai


def rodyti_rezultatus(rezultatai, pavadinimas):
    rezultatai.sort(key=lambda x: x["Score"], reverse=True)
    normalus = [r for r in rezultatai if not r["Anomalija"]]
    anomalijos = [r for r in rezultatai if r["Anomalija"]]

    st.subheader(f"🏆 TOP 20 {pavadinimas}")
    st.caption(f"Iš viso rastų akcijų: {len(rezultatai)}")

    if not normalus:
        st.warning("⚠️ Nerasta tinkamų akcijų! Patikrink tickerių sąrašą.")
        return normalus, anomalijos

    nr1 = normalus[0]
    st.success(
        f"🥇 **{nr1['Ticker']}** ({nr1['Regionas']}) — "
        f"Score: **{nr1['Score']}** | RSI: {nr1['RSI']} | 6m: {nr1['6m%']}%"
    )

    top20 = normalus[:20]
    dabar = pd.Timestamp.now(tz="UTC")

    df_trumpa = pd.DataFrame([{
        "":       ("🟢" if r["Statusas"] == "zalia" else "🟡"),
        "Ticker": r["Ticker"],
        "Score":  r["Score"],
        "6m%":    r["6m%"],
        "3m%":    r["3m%"],
        "RSI":    r["RSI"],
        "P/E":    r["P/E"],
    } for r in top20])

    st.dataframe(df_trumpa, hide_index=True, use_container_width=True)

    with st.expander("📊 Pilna lentelė"):
        df_pilna = pd.DataFrame([{
            "Ticker": r["Ticker"],
            "Reg":    r["Regionas"],
            "Score":  r["Score"],
            "1m%":    r["1m%"],
            "3m%":    r["3m%"],
            "6m%":    r["6m%"],
            "12m%":   r["12m%"],
            "RSI":    r["RSI"],
            "P/E":    r["P/E"],
            "Rev":    r["Rev"],
            "D/E":    r["D/E"],
            "FCF":    r["FCF"],
            "ROE":    r["ROE"],
        } for r in top20])
        st.dataframe(df_pilna, hide_index=True, use_container_width=True)

    ispejimai_list = []
    for r in top20:
        ticker = r["Ticker"]
        rsi = r["RSI"]
        warns = list(r["Įspėjimai"])

        if rsi > 75:
            warns.append("🔵 Overbought")
        if rsi < 30:
            warns.append("⚫ Oversold")

        if top20.index(r) < 10:
            turi_nauj, naujienos = gauti_naujienas(ticker)
            if turi_nauj:
                warns.append("📰 Naujienos 48h")
            ed = gauti_earnings(ticker)
            if ed:
                try:
                    ed_tz = ed.tz_localize("UTC") if ed.tzinfo is None else ed
                    if 0 <= (ed_tz - dabar).days <= 14:
                        warns.append(f"⚡ Earnings {ed.strftime('%m-%d')}")
                except:
                    pass

        if warns:
            spalva = "🟢" if r["Statusas"] == "zalia" else "🟡"
            ispejimai_list.append(f"**{spalva} {ticker}:** {' | '.join(warns)}")

    if ispejimai_list:
        with st.expander("⚠️ Įspėjimai ir pastabos", expanded=True):
            for txt in ispejimai_list:
                st.markdown(txt)

    if anomalijos:
        with st.expander(f"⚠️ Anomalijos ({len(anomalijos)} vnt.) — augimas >300%/6mėn"):
            for r in anomalijos[:5]:
                st.markdown(
                    f"**{r['Ticker']}** ({r['Regionas']}) — "
                    f"6m: {r['6m%']}% | Score: {r['Score']}"
                )

    return normalus, anomalijos


def gauti_tickerius_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        lentele = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
        return [t.replace(".", "-") for t in lentele["Symbol"].tolist()]
    except Exception as e:
        st.error(f"S&P 500 klaida: {e}")
        return []


def gauti_tickerius_europa():
    tickeriai = []

    try:
        from pytickersymbols import PyTickerSymbols
        stock_data = PyTickerSymbols()
        indeksai = [
            "DAX", "CAC_40", "FTSE 100", "IBEX 35", "AEX",
            "BEL 20", "OMX Stockholm 30", "OMX Helsinki 25",
            "MDAX", "SDAX", "TecDAX", "EURO STOXX 50",
            "Switzerland 20",
        ]
        pirmenybe = [
            ".DE", ".PA", ".L", ".AS", ".MC", ".MI",
            ".ST", ".CO", ".OL", ".HE", ".BR", ".LS",
            ".VI", ".WA", ".IR", ".LU", ".SW", ".F",
        ]
        for indeksas in indeksai:
            try:
                akcijos = list(stock_data.get_stocks_by_index(indeksas))
                for akcija in akcijos:
                    simboliai = [
                        sym.get("yahoo", "")
                        for sym in akcija.get("symbols", [])
                        if sym.get("yahoo")
                    ]
                    rastas = None
                    for sf in pirmenybe:
                        t = next((x for x in simboliai if x.endswith(sf)), None)
                        if t:
                            rastas = t
                            break
                    if rastas:
                        tickeriai.append(rastas)
            except:
                pass
    except ImportError:
        st.warning("⚠️ pytickersymbols neįdiegta — naudojamas atsarginis sąrašas")

    if len(tickeriai) < 50:
        fallback_eu = [
            "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MBG.DE",
            "BMW.DE", "BAS.DE", "MUV2.DE", "ADS.DE", "IFX.DE",
            "AIR.DE", "DHL.DE", "RWE.DE", "HEN3.DE", "VOW3.DE",
            "BEI.DE", "EOAN.DE", "FRE.DE", "MTX.DE", "SHL.DE",
            "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AI.PA",
            "AIR.PA", "SU.PA", "BNP.PA", "CS.PA", "DG.PA",
            "KER.PA", "RI.PA", "CAP.PA", "SGO.PA", "STM.PA",
            "BN.PA", "EN.PA", "VIV.PA", "ACA.PA", "GLE.PA",
            "SHEL.L", "AZN.L", "ULVR.L", "HSBA.L", "BP.L",
            "GSK.L", "RIO.L", "LSEG.L", "DGE.L", "REL.L",
            "BATS.L", "AAL.L", "NG.L", "VOD.L", "LLOY.L",
            "BARC.L", "ANTO.L", "PRU.L", "CRH.L", "CPG.L",
            "ASML.AS", "PHIA.AS", "UNA.AS", "INGA.AS", "AD.AS",
            "NESN.SW", "NOVN.SW", "ROG.SW", "ABBN.SW", "UBSG.SW",
            "ITX.MC", "IBE.MC", "SAN.MC", "BBVA.MC", "TEF.MC",
            "ISP.MI", "UCG.MI", "ENI.MI", "ENEL.MI", "G.MI",
            "VOLV-B.ST", "ERIC-B.ST", "ABB.ST", "SAND.ST",
            "NOVO-B.CO", "CARL-B.CO", "MAERSK-B.CO",
            "NESTE.HE", "FORTUM.HE", "NOKIA.HE",
        ]
        esami = set(tickeriai)
        for t in fallback_eu:
            if t not in esami:
                tickeriai.append(t)

    return list(dict.fromkeys([t for t in tickeriai if t]))


def gauti_tickerius_japonija():
    import os

    try:
        failo_kelias = os.path.join(os.path.dirname(__file__), "nikkei225.txt")
        with open(failo_kelias, "r") as f:
            tickeriai = [line.strip() for line in f if line.strip()]
        if len(tickeriai) > 50:
            return tickeriai
    except:
        pass

    st.info("ℹ️ nikkei225.txt nerasta — naudojamas atsarginis sąrašas (100 akcijų)")
    return [
        "6758.T", "6861.T", "6954.T", "6976.T", "6762.T",
        "6971.T", "6702.T", "6501.T", "6503.T", "6752.T",
        "7735.T", "7741.T", "7751.T", "7752.T", "7974.T",
        "4689.T", "9984.T", "9433.T", "9432.T", "9434.T",
        "7203.T", "7267.T", "7269.T", "7270.T", "7201.T",
        "7211.T", "7261.T",
        "8306.T", "8316.T", "8411.T", "8766.T", "8801.T",
        "8802.T", "8591.T", "8697.T",
        "6301.T", "6305.T", "6902.T", "7011.T", "7012.T",
        "7013.T", "5401.T", "5411.T", "5713.T", "5802.T",
        "3407.T", "4063.T", "4452.T", "4502.T", "4503.T",
        "4519.T", "4523.T", "4568.T", "4578.T",
        "8001.T", "8002.T", "8031.T", "8035.T", "8053.T",
        "8058.T",
        "2914.T", "3382.T", "4901.T", "4911.T", "6367.T",
        "6645.T", "6857.T", "6098.T", "6273.T", "7733.T",
        "9020.T", "9021.T", "9022.T", "9101.T", "9104.T",
        "9107.T", "1925.T", "1928.T", "2502.T", "2503.T",
        "2801.T", "2802.T", "3659.T", "4307.T", "4661.T",
        "6506.T", "6594.T", "6723.T", "6981.T", "8015.T",
        "8725.T", "9735.T", "9766.T", "9983.T",
    ]


def gauti_tickerius_azija():
    return [
        "0700.HK", "9988.HK", "9618.HK", "9888.HK", "9999.HK",
        "3690.HK", "1024.HK", "1810.HK", "2382.HK", "0268.HK",
        "0285.HK", "0992.HK", "2018.HK",
        "0005.HK", "1299.HK", "0388.HK", "2318.HK", "0939.HK",
        "1398.HK", "3988.HK", "2628.HK", "2388.HK", "0011.HK",
        "1109.HK",
        "0941.HK", "0762.HK", "0728.HK",
        "0883.HK", "0857.HK", "0386.HK", "1088.HK", "0016.HK",
        "0066.HK", "0027.HK", "1113.HK",
        "1211.HK", "2015.HK", "9866.HK", "9868.HK",
        "0175.HK", "0669.HK", "1044.HK", "0960.HK", "1876.HK",
        "2020.HK", "2269.HK", "2313.HK", "6098.HK", "0267.HK",
        "0688.HK", "1093.HK", "0868.HK", "0017.HK", "0002.HK",
        "0003.HK", "0006.HK", "0012.HK", "0101.HK", "0823.HK",
        "1038.HK", "1997.HK",
    ]


with st.expander("📊 Rinkų skydelis", expanded=False):
    indeksai = {
        "🇺🇸 S&P":   "^GSPC",
        "💻 Nasdaq":  "^IXIC",
        "🇪🇺 STOXX":  "^STOXX50E",
        "🇯🇵 Nikkei": "^N225",
    }
    cols = st.columns(4)
    for idx, (pav, ticker) in enumerate(indeksai.items()):
        try:
            d = yf.Ticker(ticker).history(period="5d")
            if len(d) >= 2:
                pokytis = round(
                    (d["Close"].iloc[-1] - d["Close"].iloc[-2])
                    / d["Close"].iloc[-2] * 100, 2
                )
                cols[idx].metric(pav, f"{d['Close'].iloc[-1]:,.0f}", f"{pokytis:+.2f}%")
            else:
                cols[idx].metric(pav, "—")
        except:
            cols[idx].metric(pav, "—")

    try:
        vix_d = yf.Ticker("^VIX").history(period="5d")
        vix = round(vix_d["Close"].iloc[-1], 1)
        if vix < 15:
            fg = "😎 Godumas"
        elif vix < 20:
            fg = "😊 Ramu"
        elif vix < 25:
            fg = "😐 Neutralu"
        elif vix < 35:
            fg = "😟 Baimė"
        else:
            fg = "😱 Panika"
        st.metric("VIX", vix, fg)
    except:
        pass

st.divider()


col1, col2 = st.columns(2)
with col1:
    btn_sp = st.button("🇺🇸 S&P 500")
    btn_eu = st.button("🌍 Europa")
with col2:
    btn_jp = st.button("🇯🇵 Japonija")
    btn_az = st.button("🌏 Azija (HK)")

btn_all = st.button("🌐 Viską analizuoti", type="primary")


def paleisti_analize(tickeriai, pavadinimas):
    st.info(f"Randama akcijų: {len(tickeriai)}. Pradedama analizė...")
    progress = st.progress(0)
    statusas = st.empty()
    rezultatai = analizuoti(tickeriai, progress, statusas)
    rodyti_rezultatus(rezultatai, pavadinimas)


if btn_sp:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_sp500()
    paleisti_analize(t, "S&P 500")

if btn_eu:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_europa()
    paleisti_analize(t, "Europa")

if btn_jp:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_japonija()
    paleisti_analize(t, "Japonija")

if btn_az:
    with st.spinner("Kraunama..."):
        t = gauti_tickerius_azija()
    paleisti_analize(t, "Azija (Hang Seng)")

if btn_all:
    with st.spinner("Kraunami visi sąrašai..."):
        t = list(dict.fromkeys(
            gauti_tickerius_sp500() +
            gauti_tickerius_europa() +
            gauti_tickerius_japonija() +
            gauti_tickerius_azija()
        ))
    paleisti_analize(t, "🌐 Globalus")
