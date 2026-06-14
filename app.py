import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="Momentum Skaneris", layout="wide")
st.title("📈 Globalus Momentum Skaneris")
st.caption("ℹ️ Spalvų legenda: 🟢 Stiprus signalas | 🟡 Yra įspėjimų | 🔴 Praleisk | 🔵 Overbought (RSI>70) | ⚫ Oversold (RSI<35) | ⚡ Earnings <14d | 📰 Naujienos 48h")

def momentum_proc(kaina_pradzioje, kaina_dabar):
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
    except:
        return None, None, None, None
    revenue_growth = info.get("revenueGrowth")
    debt_to_equity = info.get("debtToEquity")
    free_cash_flow = info.get("freeCashflow")
    roe            = info.get("returnOnEquity")
    return revenue_growth, debt_to_equity, free_cash_flow, roe

def vertinti_fundamentalus(revenue_growth, debt_to_equity, free_cash_flow, roe):
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
    elif revenue_growth < 0:
        score_papildas -= 15
        ispejimai.append("📉 Pajamos mažėja")
        raudoni += 1

    if debt_to_equity is None:
        truksta += 1
    else:
        de = debt_to_equity / 100
        if de < 0.5:
            score_papildas += 10
        elif de < 1.5:
            score_papildas += 5
        elif de < 3.0:
            ispejimai.append("⚠️ Didelė skola")
            geltoni += 1
        else:
            score_papildas -= 20
            ispejimai.append("🔴 Kritiškai didelė skola")
            raudoni += 1

    if free_cash_flow is None:
        truksta += 1
    elif free_cash_flow > 0:
        score_papildas += 10
    else:
        ispejimai.append("⚠️ Neigiamas cash flow")
        geltoni += 1
        score_papildas -= 5

    if roe is None:
        truksta += 1
    elif roe > 0.15:
        score_papildas += 15
    elif roe > 0.05:
        score_papildas += 5
    elif roe < 0:
        ispejimai.append("📉 Nuostolingas")
        raudoni += 1
        score_papildas -= 20

    if truksta == 4:
        return 0, ["❓ Fundamentalių duomenų nėra"], "geltona"

    if truksta >= 2:
        ispejimai.append(f"❓ Trūksta {truksta}/4 rodiklių")
        geltoni += 1

    if raudoni >= 1:
        statusas = "raudona"
    elif geltoni >= 2 or truksta >= 1:
        statusas = "geltona"
    else:
        statusas = "zalia"

    return score_papildas, ispejimai, statusas

def skaiciuoti_score(m1, m3, m6, m12, rsi, pe, fund_papildas):
    score = 0
    score += m6  * 0.35
    score += m3  * 0.25
    score += m12 * 0.20
    score += m1  * 0.10
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
        turi = len(naujienos) > 0 and any(
            n.get("providerPublishTime", 0) > pd.Timestamp.now().timestamp() - 48*3600
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
    """
    Filtruoja ETF/ETP tickerius.
    - IE0..., LU0... — visada ETF
    - 0... + .HK pabaiga — Honkongo ETF/ETP
    - Japonijos tickeriai (pvz. 6976.T) NĖRA filtruojami net jei prasideda skaičiumi
    """
    if tickeris.startswith("IE0") or tickeris.startswith("LU0"):
        return True
    if tickeris.startswith("0") and tickeris.endswith(".HK"):
        return True
    return False

def analizuoti(tickeriai, progress, statusas_txt):
    rezultatai = []
    sekmingai = 0
    praleista_klaida = 0
    praleista_istorija = 0
    praleista_raudona = 0

    def atnaujinti_statusa(i):
        statusas_txt.text(
            f"⏳ {i+1}/{len(tickeriai)} | "
            f"✅ Įtraukta: {sekmingai} | "
            f"⚠️ Trumpa istorija: {praleista_istorija} | "
            f"❌ Klaidos: {praleista_klaida} | "
            f"🔴 Filtruota: {praleista_raudona}"
        )

    for i, tickeris in enumerate(tickeriai):
        if yra_etf_filtras(tickeris):
            praleista_istorija += 1
            progress.progress((i + 1) / len(tickeriai))
            atnaujinti_statusa(i)
            continue
        try:
            akcija = yf.Ticker(tickeris)
            duomenys = akcija.history(period="2y")
            kainos = duomenys["Close"]

            if len(kainos) < 200:
                praleista_istorija += 1
                progress.progress((i + 1) / len(tickeriai))
                atnaujinti_statusa(i)
                continue

            kaina_dabar = kainos.iloc[-1]
            m1  = momentum_proc(kainos.iloc[-21],  kaina_dabar)
            m3  = momentum_proc(kainos.iloc[-63],  kaina_dabar)
            m6  = momentum_proc(kainos.iloc[-126], kaina_dabar)
            m12 = momentum_proc(kainos.iloc[-252], kaina_dabar)
            rsi = skaiciuoti_rsi(kainos)

            info = akcija.info
            pe = info.get("trailingPE")
            if pe is not None:
                pe = round(pe, 1)

            rev_g, d2e, fcf, roe = gauti_fundamentalus(akcija)
            fund_papildas, ispejimai, fund_statusas = vertinti_fundamentalus(rev_g, d2e, fcf, roe)

            if fund_statusas == "raudona":
                praleista_raudona += 1
                progress.progress((i + 1) / len(tickeriai))
                atnaujinti_statusa(i)
                continue

            anomalija = m6 > 300
            score = skaiciuoti_score(m1, m3, m6, m12, rsi, pe, fund_papildas)
            sekmingai += 1

            rezultatai.append({
                "Ticker":    tickeris,
                "1mėn %":    round(m1, 1),
                "3mėn %":    round(m3, 1),
                "6mėn %":    round(m6, 1),
                "12mėn %":   round(m12, 1),
                "P/E":       pe if pe is not None else "n/a",
                "RSI":       rsi,
                "Rev":       f"{round(rev_g*100,1)}%" if rev_g is not None else "n/a",
                "D/E":       f"{round(d2e/100,2)}" if d2e is not None else "n/a",
                "FCF":       "✅" if fcf and fcf > 0 else ("❌" if fcf and fcf <= 0 else "n/a"),
                "ROE":       f"{round(roe*100,1)}%" if roe is not None else "n/a",
                "Įspėjimai": ispejimai,
                "Statusas":  fund_statusas,
                "Score":     score,
                "Anomalija": anomalija,
            })
        except Exception:
            praleista_klaida += 1

        progress.progress((i + 1) / len(tickeriai))
        atnaujinti_statusa(i)

    statusas_txt.text(
        f"✅ Baigta! Įtraukta: {sekmingai} | "
        f"Trumpa istorija: {praleista_istorija} | "
        f"Klaidos: {praleista_klaida} | "
        f"Filtruota (raudona): {praleista_raudona}"
    )
    return rezultatai

def rodyti_rezultatus(rezultatai, pavadinimas):
    rezultatai.sort(key=lambda x: x["Score"] if isinstance(x["Score"], (int, float)) else 0, reverse=True)
    normalus   = [r for r in rezultatai if not r["Anomalija"]]
    anomalijos = [r for r in rezultatai if r["Anomalija"]]

    st.subheader(f"🏆 TOP 20 {pavadinimas}")
    dabar = pd.Timestamp.now(tz="UTC")

    top20 = normalus[:20]
    eilutes = []
    for r in top20:
        ticker  = r["Ticker"]
        rsi     = r["RSI"]
        statusas = r["Statusas"]

        if statusas == "raudona":
            spalva = "🔴"
        elif rsi > 75:
            spalva = "🔵"
        elif rsi < 30:
            spalva = "⚫"
        elif statusas == "zalia":
            spalva = "🟢"
        else:
            spalva = "🟡"

        turi_naujienu, naujienos = gauti_naujienas(ticker)
        earnings_data = gauti_earnings(ticker)
        earnings_artimas = False
        earnings_str = ""
        if earnings_data is not None:
            try:
                ed = earnings_data.tz_localize("UTC") if earnings_data.tzinfo is None else earnings_data
                if 0 <= (ed - dabar).days <= 14:
                    earnings_artimas = True
                    earnings_str = f"⚡ {earnings_data.strftime('%m-%d')}"
            except:
                pass

        ispejimai = list(r["Įspėjimai"])
        if rsi > 75:
            ispejimai.append("🔵 RSI>75 — overbought")
        if rsi < 30:
            ispejimai.append("⚫ RSI<30 — oversold")
        if turi_naujienu:
            ispejimai.append("📰 Naujienos 48h")
        if earnings_artimas:
            ispejimai.append(earnings_str)

        eilutes.append({
            "r": r,
            "spalva": spalva,
            "ispejimai": ispejimai,
            "naujienos": naujienos,
            "turi_naujienu": turi_naujienu,
        })

    lentelės_duomenys = []
    for e in eilutes:
        r = e["r"]
        lentelės_duomenys.append({
            "":        e["spalva"],
            "Ticker":  r["Ticker"],
            "Score":   r["Score"],
            "1mėn%":   r["1mėn %"],
            "3mėn%":   r["3mėn %"],
            "6mėn%":   r["6mėn %"],
            "12mėn%":  r["12mėn %"],
            "RSI":     r["RSI"],
            "P/E":     r["P/E"],
            "Rev":     r["Rev"],
            "D/E":     r["D/E"],
            "FCF":     r["FCF"],
            "ROE":     r["ROE"],
        })

    df = pd.DataFrame(lentelės_duomenys)
    st.dataframe(df, width="stretch", hide_index=True)

    ispejimai_visi = []
    for e in eilutes:
        if e["ispejimai"]:
            ticker = e["r"]["Ticker"]
            txt = f"**{e['spalva']} {ticker}:** " + " | ".join(e["ispejimai"])
            ispejimai_visi.append(txt)

    if ispejimai_visi:
        with st.expander("⚠️ Įspėjimai ir pastabos", expanded=True):
            for txt in ispejimai_visi:
                st.markdown(txt)
            for e in eilutes:
                if e["turi_naujienu"] and e["naujienos"]:
                    st.markdown(f"**📰 {e['r']['Ticker']} naujienos:**")
                    for n in e["naujienos"]:
                        st.markdown(f"- [{n.get('title','')}]({n.get('link','#')})")

    if anomalijos:
        st.markdown("---")
        st.markdown(
            "<div style='background:#2d1a00;border-left:4px solid #ff6600;"
            "padding:10px;border-radius:6px;margin-bottom:8px'>"
            "<b>⚠️ ANOMALIJOS — augimas >300% per 6 mėn</b><br>"
            "<small>Dėmesio: šios kompanijos auga nenormaliai greitai. "
            "Gali būti spekuliacinis burbulas arba tikras proveržis. Tikrinti atidžiai.</small>"
            "</div>",
            unsafe_allow_html=True
        )
        df_an = pd.DataFrame(anomalijos[:5])[
            ["Ticker","1mėn %","3mėn %","6mėn %","12mėn %","P/E","RSI","Rev","D/E","FCF","ROE","Score"]
        ]
        st.dataframe(df_an, width="stretch", hide_index=True)

    if normalus:
        st.success(f"✅ {pavadinimas} signalas: **{normalus[0]['Ticker']}** (Score: {normalus[0]['Score']})")

    return normalus, anomalijos


SALIES_SUFIKSAS = {
    "United Kingdom": ".L",  "Switzerland": ".SW", "France": ".PA",
    "Germany": ".DE",        "Netherlands": ".AS", "Spain": ".MC",
    "Italy": ".MI",          "Sweden": ".ST",      "Denmark": ".CO",
    "Norway": ".OL",         "Finland": ".HE",     "Belgium": ".BR",
    "Portugal": ".LS",       "Austria": ".VI",     "Poland": ".WA",
    "Ireland": ".IR",        "Luxembourg": ".LU",
}

def gauti_tickerius_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    lentele = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
    return [t.replace(".", "-") for t in lentele["Symbol"].tolist()]

def gauti_tickerius_europa():
    try:
        from pytickersymbols import PyTickerSymbols
        stock_data = PyTickerSymbols()
        indeksai = [
            "DAX", "CAC_40", "CAC Mid 60", "FTSE 100", "IBEX 35",
            "AEX", "BEL 20", "OMX Stockholm 30", "OMX Helsinki 25",
            "MDAX", "SDAX", "TecDAX", "EURO STOXX 50", "Switzerland 20",
        ]
        pirmenybe = [
            ".DE", ".PA", ".L", ".AS", ".MC", ".MI",
            ".ST", ".CO", ".OL", ".HE", ".BR", ".LS",
            ".VI", ".WA", ".IR", ".LU", ".SW", ".F",
        ]
        tickeriai = []
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
        return list(dict.fromkeys([t for t in tickeriai if t]))
    except Exception as e:
        st.error(f"Europa klaida: {e}")
        return []

def gauti_tickerius_japonija():
    import os
    failo_kelias = os.path.join(os.path.dirname(__file__), "nikkei225.txt")
    try:
        with open(failo_kelias, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("❌ Nerasta nikkei225.txt! Įdėk failą į tą patį folderį kaip app.py")
        return []

def gauti_tickerius_azija():
    return [
        "0700.HK","9988.HK","0941.HK","1299.HK","0005.HK",
        "0939.HK","1398.HK","2318.HK","3690.HK","0388.HK",
        "0883.HK","2628.HK","1810.HK","0016.HK","0027.HK",
        "0066.HK","0175.HK","0267.HK","0386.HK","0688.HK",
        "0762.HK","0857.HK","0868.HK","0960.HK","1088.HK",
        "1093.HK","1109.HK","1113.HK","1211.HK","1876.HK",
        "2020.HK","2313.HK","2382.HK","6098.HK","9618.HK",
        "2388.HK","3988.HK","0669.HK","1044.HK","9999.HK",
        "9888.HK","1024.HK","2269.HK","9868.HK",
    ]


def rodyti_rinku_skydeli():
    try:
        indeksai = {
            "🇺🇸 S&P 500":   "^GSPC",
            "💻 Nasdaq":      "^IXIC",
            "🏭 Dow Jones":   "^DJI",
            "🇪🇺 STOXX 50":  "^STOXX50E",
            "💵 USD/EUR":     "USDEUR=X",
            "🇯🇵 Nikkei":    "^N225",
        }

        cols = st.columns(7)

        for idx, (pav, ticker) in enumerate(indeksai.items()):
            try:
                d = yf.Ticker(ticker).history(period="5d")
                if len(d) >= 2:
                    kaina   = d["Close"].iloc[-1]
                    pokytis = round((d["Close"].iloc[-1] - d["Close"].iloc[-2]) / d["Close"].iloc[-2] * 100, 2)

                    if ticker == "USDEUR=X":
                        kaina_str = f"{kaina:.4f}"
                    elif kaina > 1000:
                        kaina_str = f"{kaina:,.0f}"
                    else:
                        kaina_str = f"{kaina:.2f}"

                    cols[idx].metric(
                        label=pav,
                        value=kaina_str,
                        delta=f"{pokytis:+.2f}%",
                    )
                else:
                    cols[idx].metric(label=pav, value="n/a")
            except:
                cols[idx].metric(label=pav, value="n/a")

        try:
            vix_d = yf.Ticker("^VIX").history(period="5d")
            vix   = round(vix_d["Close"].iloc[-1], 1)
            if vix < 15:
                fg_label = "😎 Ekstr. godumas"
                fg_spalva = "🟢"
            elif vix < 20:
                fg_label = "😊 Godumas"
                fg_spalva = "🟡"
            elif vix < 25:
                fg_label = "😐 Neutralus"
                fg_spalva = "🟡"
            elif vix < 35:
                fg_label = "😟 Baimė"
                fg_spalva = "🔴"
            else:
                fg_label = "😱 Ekstr. baimė"
                fg_spalva = "🔴"
            cols[6].metric(
                label=f"{fg_spalva} VIX (F&G proxy)",
                value=f"{vix}",
                delta=fg_label,
            )
        except:
            cols[6].metric(label="VIX", value="n/a")

    except Exception as e:
        st.warning(f"Rinkų skydelis neprieinamas: {e}")

# Rinkų skydelis
rodyti_rinku_skydeli()
st.divider()

# ============================================
# DEBUG SEKCIJA — laikinai, diagnostikai
# ============================================
with st.expander("🔍 DEBUG — testuoti tickerius", expanded=True):
    debug_ticker = st.text_input("Įvesk tickerį testavimui", value="0700.HK")

    if st.button("🧪 Testuoti"):
        st.write(f"### Testuojama: `{debug_ticker}`")

        st.write("**1. Bandymas gauti history (2y):**")
        try:
            akcija = yf.Ticker(debug_ticker)
            duomenys = akcija.history(period="2y")
            if duomenys.empty:
                st.error("❌ history() grąžino TUŠČIĄ DataFrame")
            else:
                st.success(f"✅ history() OK — {len(duomenys)} eilučių")
                st.write(f"Paskutinė kaina: {duomenys['Close'].iloc[-1]}")
                st.write(f"Pirma data: {duomenys.index[0]}, paskutinė: {duomenys.index[-1]}")
        except Exception as e:
            st.error(f"❌ history() KLAIDA: {type(e).__name__}: {e}")

        st.write("**2. Bandymas gauti .info:**")
        try:
            akcija2 = yf.Ticker(debug_ticker)
            info = akcija2.info
            if not info or len(info) < 2:
                st.error(f"❌ .info tuščias arba minimalus: {info}")
            else:
                st.success(f"✅ .info OK — {len(info)} laukų")
                st.write(f"trailingPE: {info.get('trailingPE', 'nėra')}")
                st.write(f"shortName: {info.get('shortName', 'nėra')}")
        except Exception as e:
            st.error(f"❌ .info KLAIDA: {type(e).__name__}: {e}")

        st.write("**3. Bandymas gauti fast_info:**")
        try:
            akcija3 = yf.Ticker(debug_ticker)
            fi = akcija3.fast_info
            st.success(f"✅ fast_info OK")
            try:
                st.write(f"last_price: {fi.get('lastPrice', 'nėra')}")
            except:
                st.write(f"fast_info objektas: {fi}")
        except Exception as e:
            st.error(f"❌ fast_info KLAIDA: {type(e).__name__}: {e}")

        st.write("**4. yfinance versija:**")
        st.write(f"yfinance: {yf.__version__}")

st.divider()

if st.button("🚀 Analizuoti S&P 500"):
    with st.spinner("Kraunamas S&P 500 sąrašas..."):
        tickeriai = gauti_tickerius_sp500()
    st.info(f"Randama akcijų: {len(tickeriai)}. Analizė užtruks ~15-20 min.")
    progress = st.progress(0)
    statusas = st.empty()
    rezultatai = analizuoti(tickeriai, progress, statusas)
    rodyti_rezultatus(rezultatai, "S&P 500")

if st.button("🌍 Analizuoti Europa"):
    with st.spinner("Kraunamas Europos akcijų sąrašas..."):
        tickeriai_eu = gauti_tickerius_europa()
    st.info(f"Randama akcijų: {len(tickeriai_eu)}. Analizė užtruks ~20-25 min.")
    progress_eu = st.progress(0)
    statusas_eu = st.empty()
    rezultatai_eu = analizuoti(tickeriai_eu, progress_eu, statusas_eu)
    rodyti_rezultatus(rezultatai_eu, "Europa")

if st.button("🗾 Analizuoti Japonija (Nikkei 225)"):
    with st.spinner("Kraunamas Nikkei 225 sąrašas..."):
        tickeriai_jp = gauti_tickerius_japonija()
    st.info(f"Randama akcijų: {len(tickeriai_jp)}. Analizė užtruks ~10-15 min.")
    progress_jp = st.progress(0)
    statusas_jp = st.empty()
    rezultatai_jp = analizuoti(tickeriai_jp, progress_jp, statusas_jp)
    rodyti_rezultatus(rezultatai_jp, "Japonija")

if st.button("🌏 Analizuoti Azija (Hang Seng)"):
    with st.spinner("Kraunamas Hang Seng sąrašas..."):
        tickeriai_az = gauti_tickerius_azija()
        tickeriai_az = list(dict.fromkeys(tickeriai_az))
    st.info(f"Randama akcijų: {len(tickeriai_az)}. Analizė užtruks ~5-8 min.")
    progress_az = st.progress(0)
    statusas_az = st.empty()
    rezultatai_az = analizuoti(tickeriai_az, progress_az, statusas_az)
    rodyti_rezultatus(rezultatai_az, "Azija (Hang Seng)")

if st.button("🌐 Analizuoti Viską (Globalus nugalėtojas)"):
    with st.spinner("Kraunami visi sąrašai..."):
        tickeriai_visi = (
            gauti_tickerius_sp500() +
            gauti_tickerius_europa() +
            gauti_tickerius_japonija() +
            gauti_tickerius_azija()
        )
        tickeriai_visi = list(dict.fromkeys(tickeriai_visi))
    st.info(f"Iš viso akcijų: {len(tickeriai_visi)}. Analizė užtruks ~50-70 min.")
    progress_v = st.progress(0)
    statusas_v = st.empty()
    visi_rezultatai = analizuoti(tickeriai_visi, progress_v, statusas_v)
    rodyti_rezultatus(visi_rezultatai, "🌐 Globalus")
