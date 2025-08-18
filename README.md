# Gestión de Riesgos Financieros

## Librerías


```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from scipy.stats import chi2
```

## 1. Elección de los Activos

Se lista un conjunto de activos que sean candidatos para armar el portafolio:


- Renta variable USA: ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'TSLA','UNH', 'RGC', 'QUBT', 'DFDV', 'QBTS', 'RGTI', 'MNPR', 'DGNX', 'TGEN', 'NNNN',...]

- Renta fija: ['TLT' (bonos largos), 'IEF' (bonos 7–10Y), 'SHY' (bonos cortos)]

- Commodities y refugios: ['GLD' (oro), 'SLV' (plata), 'DBC' (mixto)]

- Internacional/mercados emergentes: ['VEA' (desarrollados), 'VWO' (emergentes)]

- Alternativos o defensivos: ['XLU' (utilities), 'XLP' (consumo básico)]

- Mercado FX: ['USD/EUR', 'USD/JPY', 'USD/PEN']


Se define un Benchmark con el cual comparar el portafolio

- Benchmark: ['SPY']


```python
Acciones = ['AAPL', 'ABBV', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADP', 'AFL', 'ALL', 'AMAT', 'AMGN', 'AMD', 'AMZN', 'ANET', 'APO', 'APP', 'ATLN', 'AVGO', 'AXP', 'BA', 'BAC', 'BAP', 'BK', 'BKNG', 'BLK', 'BMY',
            'BRK-B', 'BSX', 'C', 'CB', 'CAT', 'CEG', 'CHTR', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DGNX', 'DHI', 'DHR', 'DE', 'DELL', 'DIS', 'DFDV', 'DUK', 'ELV', 'EME', 'EOG', 'EPD',
            'ET', 'F', 'FI', 'GE', 'GEV', 'GILD','GOOG', 'GM', 'GS', 'HCA', 'HD', 'HON', 'IBM', 'INTC', 'INTU', 'IFS', 'ISRG', 'IOBT', 'JNJ', 'JPM', 'KKR', 'KLAC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'LRCX',
            'MA', 'MCD', 'MDLZ', 'MDT', 'MET','META', 'MNPR', 'MMC', 'MO', 'MPLX', 'MRK', 'MS', 'MSFT', 'MTNB', 'MU', 'NEE', 'NFLX', 'NKE', 'NNNN', 'NOW', 'NSRGY', 'NVDA', 'ORCL', 'PANW', 'PEP', 'PFE', 'PG',
            'PGEN', 'PGR', 'PLD','PLTR', 'PM', 'PNC', 'PPSI', 'PWR', 'QCOM', 'QBTS', 'QUBT', 'RGC', 'REGN', 'RGTI', 'RTX', 'SAP', 'SBUX', 'SCHW', 'SLB', 'SO', 'SPGI', 'SYK', 'T', 'TGEN', 'TMO', 'TM', 'TJX',
            'TMUS', 'TRV', 'TSLA', 'TXN', 'UBER', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VRTX', 'VZ', 'WFC', 'WMT', 'XOM']

Bonos = ['TLT', 'IEF', 'SHY']

Commodities = ['GLD', 'SLV', 'DBC']

Emergentes = ['VEA', 'VWO']

Defensivos = ['XLU', 'XLP']

FX = ['USD/EUR', 'USD/JPY', 'USD/PEN']

Benchmark = ['SPY']
```

### Análisis Fundamental de las Acciones


```python
def obtener_datos_fundamentales(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        'Ticker': ticker,
        'Nombre': info.get('shortName'),
        'Sector': info.get('sector'),
        'Industria': info.get('industry'),
        'Capitalización Bursátil': info.get('marketCap'),
        'PER': info.get('trailingPE'),
        'Forward PER': info.get('forwardPE'),
        'P/E': info.get('trailingPE'),
        'P/B': info.get('priceToBook'),
        'ROE': info.get('returnOnEquity'),
        'ROA': info.get('returnOnAssets'),
        'Deuda/Patrimonio': info.get('debtToEquity'),
        'Crec. Ingresos (%)': info.get('revenueGrowth'),
        'Crec. Utilidad (%)': info.get('earningsGrowth'),
        'Dividend Yield': info.get('dividendYield'),
        'Beta': info.get('beta')
    }

datos_fundamentales = [obtener_datos_fundamentales(tk) for tk in Acciones]
df_fundamental = pd.DataFrame(datos_fundamentales)
df_fundamental
```





  <div id="df-0981a2dc-0e5d-429a-80b9-07c9ded05974" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>Nombre</th>
      <th>Sector</th>
      <th>Industria</th>
      <th>Capitalización Bursátil</th>
      <th>PER</th>
      <th>Forward PER</th>
      <th>P/E</th>
      <th>P/B</th>
      <th>ROE</th>
      <th>ROA</th>
      <th>Deuda/Patrimonio</th>
      <th>Crec. Ingresos (%)</th>
      <th>Crec. Utilidad (%)</th>
      <th>Dividend Yield</th>
      <th>Beta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAPL</td>
      <td>Apple Inc.</td>
      <td>Technology</td>
      <td>Consumer Electronics</td>
      <td>3436888195072</td>
      <td>35.089394</td>
      <td>27.868832</td>
      <td>35.089394</td>
      <td>52.265850</td>
      <td>1.49814</td>
      <td>0.24546</td>
      <td>154.486</td>
      <td>0.096</td>
      <td>0.121</td>
      <td>0.45</td>
      <td>1.165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABBV</td>
      <td>AbbVie Inc.</td>
      <td>Healthcare</td>
      <td>Drug Manufacturers - General</td>
      <td>365130285056</td>
      <td>97.957350</td>
      <td>17.039572</td>
      <td>97.957350</td>
      <td>-1987.403800</td>
      <td>1.12854</td>
      <td>0.08869</td>
      <td>NaN</td>
      <td>0.066</td>
      <td>-0.324</td>
      <td>3.17</td>
      <td>0.503</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABT</td>
      <td>Abbott Laboratories</td>
      <td>Healthcare</td>
      <td>Medical Devices</td>
      <td>229305614336</td>
      <td>16.530740</td>
      <td>25.532946</td>
      <td>16.530740</td>
      <td>4.534816</td>
      <td>0.30931</td>
      <td>0.06617</td>
      <td>26.501</td>
      <td>0.074</td>
      <td>0.365</td>
      <td>1.79</td>
      <td>0.705</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACGL</td>
      <td>Arch Capital Group Ltd.</td>
      <td>Financial Services</td>
      <td>Insurance - Diversified</td>
      <td>33858519040</td>
      <td>9.381593</td>
      <td>10.035398</td>
      <td>9.381593</td>
      <td>1.527067</td>
      <td>0.17087</td>
      <td>0.03793</td>
      <td>11.840</td>
      <td>0.233</td>
      <td>-0.021</td>
      <td>NaN</td>
      <td>0.475</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture plc</td>
      <td>Technology</td>
      <td>Information Technology Services</td>
      <td>153850920960</td>
      <td>19.650755</td>
      <td>17.555792</td>
      <td>19.650755</td>
      <td>5.034342</td>
      <td>0.26928</td>
      <td>0.11245</td>
      <td>25.881</td>
      <td>0.077</td>
      <td>0.147</td>
      <td>2.40</td>
      <td>1.290</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>146</th>
      <td>VRTX</td>
      <td>Vertex Pharmaceuticals Incorpor</td>
      <td>Healthcare</td>
      <td>Biotechnology</td>
      <td>100707827712</td>
      <td>27.936699</td>
      <td>20.926477</td>
      <td>27.936699</td>
      <td>5.861225</td>
      <td>0.22771</td>
      <td>0.13094</td>
      <td>8.893</td>
      <td>0.121</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>147</th>
      <td>VZ</td>
      <td>Verizon Communications Inc.</td>
      <td>Communication Services</td>
      <td>Telecom Services</td>
      <td>186530004992</td>
      <td>10.288372</td>
      <td>9.353065</td>
      <td>10.288372</td>
      <td>1.809851</td>
      <td>0.18454</td>
      <td>0.05204</td>
      <td>167.439</td>
      <td>0.052</td>
      <td>0.083</td>
      <td>6.13</td>
      <td>0.362</td>
    </tr>
    <tr>
      <th>148</th>
      <td>WFC</td>
      <td>Wells Fargo &amp; Company</td>
      <td>Financial Services</td>
      <td>Banks - Diversified</td>
      <td>246985228288</td>
      <td>13.247422</td>
      <td>14.043716</td>
      <td>13.247422</td>
      <td>1.509338</td>
      <td>0.11498</td>
      <td>0.01059</td>
      <td>NaN</td>
      <td>0.019</td>
      <td>0.203</td>
      <td>2.33</td>
      <td>1.196</td>
    </tr>
    <tr>
      <th>149</th>
      <td>WMT</td>
      <td>Walmart Inc.</td>
      <td>Consumer Defensive</td>
      <td>Discount Stores</td>
      <td>798042030080</td>
      <td>42.735043</td>
      <td>36.764706</td>
      <td>42.735043</td>
      <td>9.531072</td>
      <td>0.21783</td>
      <td>0.07151</td>
      <td>75.780</td>
      <td>0.025</td>
      <td>-0.111</td>
      <td>0.94</td>
      <td>0.664</td>
    </tr>
    <tr>
      <th>150</th>
      <td>XOM</td>
      <td>Exxon Mobil Corporation</td>
      <td>Energy</td>
      <td>Oil &amp; Gas Integrated</td>
      <td>453993463808</td>
      <td>15.126420</td>
      <td>13.531131</td>
      <td>15.126420</td>
      <td>1.728874</td>
      <td>0.11831</td>
      <td>0.05277</td>
      <td>14.442</td>
      <td>-0.123</td>
      <td>-0.236</td>
      <td>3.72</td>
      <td>0.502</td>
    </tr>
  </tbody>
</table>
<p>151 rows × 16 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0981a2dc-0e5d-429a-80b9-07c9ded05974')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0981a2dc-0e5d-429a-80b9-07c9ded05974 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0981a2dc-0e5d-429a-80b9-07c9ded05974');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-92fbb42e-a53c-4730-9c40-12b12bed55dd">
      <button class="colab-df-quickchart" onclick="quickchart('df-92fbb42e-a53c-4730-9c40-12b12bed55dd')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-92fbb42e-a53c-4730-9c40-12b12bed55dd button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_fbf86be1-f1b3-4aa9-880b-b50965dca538">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_fundamental')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_fbf86be1-f1b3-4aa9-880b-b50965dca538 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_fundamental');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Se realiza un filtrado de las acciones
# Aplicamos los filtros
df_filtradas = df_fundamental.copy()

filtros = (
    (df_filtradas['PER'] < 50) &
    (df_filtradas['Forward PER'] < df_filtradas['PER']) &
    (df_filtradas['ROE'] > 0.10) &
    (df_filtradas['ROA'] > 0.02) &
    (df_filtradas['Deuda/Patrimonio'] < 150) &
    (df_filtradas['Crec. Ingresos (%)'] > 0) &
    (df_filtradas['Crec. Utilidad (%)'] > 0) &
    (df_filtradas['Dividend Yield'] >= 0.01) &
    (df_filtradas['Beta'] >= 0.7) & (df_filtradas['Beta'] <= 1.7)
)

# Ordenar por calidad (ROE * Crec. Ingresos)
df_filtradas['Score Calidad'] = df_filtradas['ROE'] *df_filtradas['Crec. Ingresos (%)']
df_ordenado = df_filtradas[filtros].sort_values(by='Score Calidad', ascending=False)
df_ordenado
```





  <div id="df-6e71f7d7-68c8-4529-b3f2-4c0d1c032e40" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>Nombre</th>
      <th>Sector</th>
      <th>Industria</th>
      <th>Capitalización Bursátil</th>
      <th>PER</th>
      <th>Forward PER</th>
      <th>P/E</th>
      <th>P/B</th>
      <th>ROE</th>
      <th>ROA</th>
      <th>Deuda/Patrimonio</th>
      <th>Crec. Ingresos (%)</th>
      <th>Crec. Utilidad (%)</th>
      <th>Dividend Yield</th>
      <th>Beta</th>
      <th>Score Calidad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>KLAC</td>
      <td>KLA Corporation</td>
      <td>Technology</td>
      <td>Semiconductor Equipment &amp; Materials</td>
      <td>115457957888</td>
      <td>28.818840</td>
      <td>26.658745</td>
      <td>28.818840</td>
      <td>24.616380</td>
      <td>1.00775</td>
      <td>0.19897</td>
      <td>129.746</td>
      <td>0.236</td>
      <td>0.467</td>
      <td>0.87</td>
      <td>1.432</td>
      <td>0.237829</td>
    </tr>
    <tr>
      <th>81</th>
      <td>LRCX</td>
      <td>Lam Research Corporation</td>
      <td>Technology</td>
      <td>Semiconductor Equipment &amp; Materials</td>
      <td>125941841920</td>
      <td>23.978313</td>
      <td>23.304451</td>
      <td>23.978313</td>
      <td>12.802008</td>
      <td>0.58238</td>
      <td>0.18399</td>
      <td>48.233</td>
      <td>0.336</td>
      <td>0.722</td>
      <td>0.92</td>
      <td>1.661</td>
      <td>0.195680</td>
    </tr>
    <tr>
      <th>145</th>
      <td>V</td>
      <td>Visa Inc.</td>
      <td>Financial Services</td>
      <td>Credit Services</td>
      <td>668595585024</td>
      <td>33.672535</td>
      <td>27.209322</td>
      <td>33.672535</td>
      <td>17.590258</td>
      <td>0.51755</td>
      <td>0.17049</td>
      <td>65.017</td>
      <td>0.143</td>
      <td>0.121</td>
      <td>0.69</td>
      <td>0.940</td>
      <td>0.074010</td>
    </tr>
    <tr>
      <th>94</th>
      <td>MSFT</td>
      <td>Microsoft Corporation</td>
      <td>Technology</td>
      <td>Software - Infrastructure</td>
      <td>3866511802368</td>
      <td>38.163610</td>
      <td>34.793980</td>
      <td>38.163610</td>
      <td>11.258116</td>
      <td>0.33281</td>
      <td>0.14203</td>
      <td>32.661</td>
      <td>0.181</td>
      <td>0.237</td>
      <td>0.64</td>
      <td>1.055</td>
      <td>0.060239</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ADP</td>
      <td>Automatic Data Processing, Inc.</td>
      <td>Technology</td>
      <td>Software - Application</td>
      <td>122237935616</td>
      <td>30.209211</td>
      <td>27.763570</td>
      <td>30.209211</td>
      <td>19.766178</td>
      <td>0.76003</td>
      <td>0.06315</td>
      <td>148.765</td>
      <td>0.075</td>
      <td>0.106</td>
      <td>2.04</td>
      <td>0.820</td>
      <td>0.057002</td>
    </tr>
    <tr>
      <th>139</th>
      <td>TXN</td>
      <td>Texas Instruments Incorporated</td>
      <td>Technology</td>
      <td>Semiconductors</td>
      <td>176890789888</td>
      <td>35.570385</td>
      <td>33.090137</td>
      <td>35.570385</td>
      <td>10.782489</td>
      <td>0.29991</td>
      <td>0.10406</td>
      <td>85.612</td>
      <td>0.164</td>
      <td>0.156</td>
      <td>2.80</td>
      <td>1.022</td>
      <td>0.049185</td>
    </tr>
    <tr>
      <th>96</th>
      <td>MU</td>
      <td>Micron Technology, Inc.</td>
      <td>Technology</td>
      <td>Semiconductors</td>
      <td>135269244928</td>
      <td>21.778378</td>
      <td>9.391608</td>
      <td>21.778378</td>
      <td>2.665211</td>
      <td>0.13109</td>
      <td>0.06550</td>
      <td>31.948</td>
      <td>0.366</td>
      <td>4.602</td>
      <td>0.38</td>
      <td>1.374</td>
      <td>0.047979</td>
    </tr>
    <tr>
      <th>117</th>
      <td>QCOM</td>
      <td>QUALCOMM Incorporated</td>
      <td>Technology</td>
      <td>Semiconductors</td>
      <td>170320166912</td>
      <td>15.251207</td>
      <td>12.906788</td>
      <td>15.251207</td>
      <td>6.288594</td>
      <td>0.44615</td>
      <td>0.14174</td>
      <td>54.350</td>
      <td>0.103</td>
      <td>0.294</td>
      <td>2.26</td>
      <td>1.228</td>
      <td>0.045953</td>
    </tr>
    <tr>
      <th>89</th>
      <td>MMC</td>
      <td>Marsh &amp; McLennan Companies, Inc</td>
      <td>Financial Services</td>
      <td>Insurance Brokers</td>
      <td>101869412352</td>
      <td>24.875150</td>
      <td>21.973490</td>
      <td>24.875150</td>
      <td>6.461582</td>
      <td>0.28408</td>
      <td>0.07864</td>
      <td>135.265</td>
      <td>0.121</td>
      <td>0.079</td>
      <td>1.74</td>
      <td>0.775</td>
      <td>0.034374</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AMAT</td>
      <td>Applied Materials, Inc.</td>
      <td>Technology</td>
      <td>Semiconductor Equipment &amp; Materials</td>
      <td>129808064512</td>
      <td>19.302505</td>
      <td>16.692984</td>
      <td>19.302505</td>
      <td>6.618183</td>
      <td>0.35635</td>
      <td>0.15884</td>
      <td>32.106</td>
      <td>0.077</td>
      <td>0.083</td>
      <td>1.14</td>
      <td>1.694</td>
      <td>0.027439</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACN</td>
      <td>Accenture plc</td>
      <td>Technology</td>
      <td>Information Technology Services</td>
      <td>153850920960</td>
      <td>19.650755</td>
      <td>17.555792</td>
      <td>19.650755</td>
      <td>5.034342</td>
      <td>0.26928</td>
      <td>0.11245</td>
      <td>25.881</td>
      <td>0.077</td>
      <td>0.147</td>
      <td>2.40</td>
      <td>1.290</td>
      <td>0.020735</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BLK</td>
      <td>BlackRock, Inc.</td>
      <td>Financial Services</td>
      <td>Asset Management</td>
      <td>175759704064</td>
      <td>27.475430</td>
      <td>23.116295</td>
      <td>27.475430</td>
      <td>3.574283</td>
      <td>0.13921</td>
      <td>0.03609</td>
      <td>28.507</td>
      <td>0.129</td>
      <td>0.020</td>
      <td>1.84</td>
      <td>1.432</td>
      <td>0.017958</td>
    </tr>
    <tr>
      <th>39</th>
      <td>CSCO</td>
      <td>Cisco Systems, Inc.</td>
      <td>Technology</td>
      <td>Communication Equipment</td>
      <td>262151995392</td>
      <td>25.363985</td>
      <td>16.974358</td>
      <td>25.363985</td>
      <td>5.563960</td>
      <td>0.22583</td>
      <td>0.06542</td>
      <td>59.625</td>
      <td>0.076</td>
      <td>0.320</td>
      <td>2.48</td>
      <td>0.909</td>
      <td>0.017163</td>
    </tr>
    <tr>
      <th>124</th>
      <td>SAP</td>
      <td>SAP  SE</td>
      <td>Technology</td>
      <td>Software - Application</td>
      <td>325993070592</td>
      <td>42.746155</td>
      <td>41.162964</td>
      <td>42.746155</td>
      <td>7.892569</td>
      <td>0.15841</td>
      <td>0.08730</td>
      <td>21.126</td>
      <td>0.089</td>
      <td>0.970</td>
      <td>0.91</td>
      <td>0.934</td>
      <td>0.014098</td>
    </tr>
    <tr>
      <th>38</th>
      <td>CRM</td>
      <td>Salesforce, Inc.</td>
      <td>Technology</td>
      <td>Software - Application</td>
      <td>231772635136</td>
      <td>37.881250</td>
      <td>21.782570</td>
      <td>37.881250</td>
      <td>3.828443</td>
      <td>0.10312</td>
      <td>0.05087</td>
      <td>19.813</td>
      <td>0.076</td>
      <td>0.019</td>
      <td>0.69</td>
      <td>1.370</td>
      <td>0.007837</td>
    </tr>
    <tr>
      <th>129</th>
      <td>SPGI</td>
      <td>S&amp;P Global Inc.</td>
      <td>Financial Services</td>
      <td>Financial Data &amp; Stock Exchanges</td>
      <td>169890283520</td>
      <td>42.739628</td>
      <td>33.103508</td>
      <td>42.739628</td>
      <td>5.088051</td>
      <td>0.11291</td>
      <td>0.06261</td>
      <td>31.651</td>
      <td>0.058</td>
      <td>0.084</td>
      <td>0.69</td>
      <td>1.188</td>
      <td>0.006549</td>
    </tr>
    <tr>
      <th>77</th>
      <td>LIN</td>
      <td>Linde plc</td>
      <td>Basic Materials</td>
      <td>Specialty Chemicals</td>
      <td>225114308608</td>
      <td>34.120823</td>
      <td>28.323301</td>
      <td>34.120823</td>
      <td>5.844869</td>
      <td>0.17313</td>
      <td>0.06949</td>
      <td>64.823</td>
      <td>0.028</td>
      <td>0.084</td>
      <td>1.25</td>
      <td>0.929</td>
      <td>0.004848</td>
    </tr>
    <tr>
      <th>133</th>
      <td>TMO</td>
      <td>Thermo Fisher Scientific Inc</td>
      <td>Healthcare</td>
      <td>Diagnostics &amp; Research</td>
      <td>184656052224</td>
      <td>28.250145</td>
      <td>20.817795</td>
      <td>28.250145</td>
      <td>3.655685</td>
      <td>0.13418</td>
      <td>0.04973</td>
      <td>69.618</td>
      <td>0.030</td>
      <td>0.059</td>
      <td>0.35</td>
      <td>0.748</td>
      <td>0.004025</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6e71f7d7-68c8-4529-b3f2-4c0d1c032e40')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6e71f7d7-68c8-4529-b3f2-4c0d1c032e40 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6e71f7d7-68c8-4529-b3f2-4c0d1c032e40');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-5cdf056c-007b-4b88-9eee-6e50705ed96a">
      <button class="colab-df-quickchart" onclick="quickchart('df-5cdf056c-007b-4b88-9eee-6e50705ed96a')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-5cdf056c-007b-4b88-9eee-6e50705ed96a button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_c36fb34a-17f3-4477-bb7c-30bfe1201f2d">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_ordenado')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c36fb34a-17f3-4477-bb7c-30bfe1201f2d button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_ordenado');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# === Selección automática del TOP 3 para 'base' ===

topN = 3

cols_minimas = {'Ticker', 'Nombre', 'Sector', 'Score Calidad'}
faltantes = cols_minimas - set(df_ordenado.columns)
if faltantes:
    raise KeyError(f"Faltan columnas en df_ordenado: {faltantes}")

# Tomar los 3 primeros (ya está ordenado desc por 'Score Calidad')
top3 = (
    df_ordenado.loc[:, ['Ticker', 'Nombre', 'Sector', 'Score Calidad']]
    .dropna(subset=['Ticker'])
    .head(topN)
)

# Lista 'base' con los tickers del TOP 3
base = top3['Ticker'].astype(str).tolist()

print("Activos base seleccionados automáticamente (TOP 3 por Score):", base)
display(top3.style.format({'Score Calidad': '{:.4f}'}))
```

    Activos base seleccionados automáticamente (TOP 3 por Score): ['KLAC', 'LRCX', 'V']
    


<style type="text/css">
</style>
<table id="T_fbe55" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fbe55_level0_col0" class="col_heading level0 col0" >Ticker</th>
      <th id="T_fbe55_level0_col1" class="col_heading level0 col1" >Nombre</th>
      <th id="T_fbe55_level0_col2" class="col_heading level0 col2" >Sector</th>
      <th id="T_fbe55_level0_col3" class="col_heading level0 col3" >Score Calidad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fbe55_level0_row0" class="row_heading level0 row0" >75</th>
      <td id="T_fbe55_row0_col0" class="data row0 col0" >KLAC</td>
      <td id="T_fbe55_row0_col1" class="data row0 col1" >KLA Corporation</td>
      <td id="T_fbe55_row0_col2" class="data row0 col2" >Technology</td>
      <td id="T_fbe55_row0_col3" class="data row0 col3" >0.2378</td>
    </tr>
    <tr>
      <th id="T_fbe55_level0_row1" class="row_heading level0 row1" >81</th>
      <td id="T_fbe55_row1_col0" class="data row1 col0" >LRCX</td>
      <td id="T_fbe55_row1_col1" class="data row1 col1" >Lam Research Corporation</td>
      <td id="T_fbe55_row1_col2" class="data row1 col2" >Technology</td>
      <td id="T_fbe55_row1_col3" class="data row1 col3" >0.1957</td>
    </tr>
    <tr>
      <th id="T_fbe55_level0_row2" class="row_heading level0 row2" >145</th>
      <td id="T_fbe55_row2_col0" class="data row2 col0" >V</td>
      <td id="T_fbe55_row2_col1" class="data row2 col1" >Visa Inc.</td>
      <td id="T_fbe55_row2_col2" class="data row2 col2" >Financial Services</td>
      <td id="T_fbe55_row2_col3" class="data row2 col3" >0.0740</td>
    </tr>
  </tbody>
</table>



### Análisis para Renta Fija


```python
data = yf.download(Bonos, start='2020-01-01', interval='1d')['Close']
returns = data.pct_change().dropna()

# Gráfico de desempeño acumulado
desempeno_acum = (1 + returns).cumprod()
desempeno_acum.plot(figsize=(10,6), title='Desempeño Acumulado de ETFs de Renta Fija')
plt.ylabel('Crecimiento $1')
plt.grid(True)
plt.show()
```

    /tmp/ipython-input-1242455989.py:1: FutureWarning: YF.download() has changed argument auto_adjust default to True
      data = yf.download(Bonos, start='2020-01-01', interval='1d')['Close']
    [*********************100%***********************]  3 of 3 completed
    


    
![png](output_11_1.png)
    



```python
# === Cuadro comparativo de renta fija (TLT, IEF, SHY) ===

def max_drawdown(cum_series: pd.Series) -> float:
    """Max Drawdown como porcentaje positivo (ej. 0.20 = -20%)."""
    peak = cum_series.cummax()
    drawdown = (cum_series / peak) - 1.0
    return -drawdown.min()

n_days = len(returns)

# Métricas por ticker
rows = []
for tk in returns.columns:
    ret_d = returns[tk].dropna()

    # Desempeño acumulado (crec. de $1)
    cum = (1 + ret_d).cumprod()
    total_return = cum.iloc[-1] - 1.0

    # Anualización usando duración real de la muestra
    ann_return = (1.0 + total_return) ** (252 / len(ret_d)) - 1.0
    ann_vol = ret_d.std(ddof=1) * np.sqrt(252)

    # Peor día y max drawdown
    worst_day = ret_d.min()
    mdd = max_drawdown(cum)  # positivo (e.g., 0.23 = -23%)

    rows.append({
        "Ticker": tk,
        "Rend. acumulado %": total_return * 100,
        "Retorno anual %": ann_return * 100,
        "Volatilidad anual %": ann_vol * 100,
        "Peor día %": worst_day * 100,
        "Max Drawdown %": mdd * 100
    })

df_bonos_cmp = pd.DataFrame(rows).set_index("Ticker")

# Ordena por retorno anual
df_bonos_cmp = df_bonos_cmp.sort_values("Retorno anual %", ascending=False)

print("Cuadro comparativo — ETFs de Renta Fija")
display(df_bonos_cmp.round(2))

# Mejor ETF según retorno anual
mejor_bono_ticker = df_bonos_cmp.index[0]
print(f"Mejor ETF (por retorno anual): {mejor_bono_ticker} — {df_bonos_cmp.iloc[0]['Retorno anual %']:.2f}%")

# Conservar el mejor ETF:
mejor_bono = [mejor_bono_ticker]
```

    Cuadro comparativo — ETFs de Renta Fija
    



  <div id="df-003ff892-8cee-45cb-89d3-0f01173f1202" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rend. acumulado %</th>
      <th>Retorno anual %</th>
      <th>Volatilidad anual %</th>
      <th>Peor día %</th>
      <th>Max Drawdown %</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SHY</th>
      <td>9.73</td>
      <td>1.67</td>
      <td>1.88</td>
      <td>-0.51</td>
      <td>5.71</td>
    </tr>
    <tr>
      <th>IEF</th>
      <td>-2.58</td>
      <td>-0.47</td>
      <td>7.81</td>
      <td>-2.51</td>
      <td>23.92</td>
    </tr>
    <tr>
      <th>TLT</th>
      <td>-26.52</td>
      <td>-5.35</td>
      <td>17.52</td>
      <td>-6.67</td>
      <td>48.35</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-003ff892-8cee-45cb-89d3-0f01173f1202')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-003ff892-8cee-45cb-89d3-0f01173f1202 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-003ff892-8cee-45cb-89d3-0f01173f1202');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-94d26fa3-d29a-4643-86ad-0a916af5ca92">
      <button class="colab-df-quickchart" onclick="quickchart('df-94d26fa3-d29a-4643-86ad-0a916af5ca92')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-94d26fa3-d29a-4643-86ad-0a916af5ca92 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    Mejor ETF (por retorno anual): SHY — 1.67%
    

### Elección del Activo Diversificador


```python
# Candidatos a evaluar como quinto activo
candidatos = ['GLD', 'SLV', 'DBC', 'VEA', 'VWO', 'XLU', 'XLP', 'EUR=X', 'JPY=X', 'PEN=X']
tickers = base + candidatos

# Descargar precios mensuales
data = yf.download(tickers, start="2020-01-01", interval="1d")['Close']
returns = data.pct_change().dropna()

# Correlación con los activos base
correlaciones = returns.corr().loc[candidatos, base]

# Volatilidad anualizada de los candidatos
volatilidad = returns[candidatos].std() * np.sqrt(252)

# Unir métricas
resumen = correlaciones.copy()
resumen['Volatilidad anual'] = volatilidad
resumen = resumen.sort_values('Volatilidad anual')

display(resumen.round(4))
```

    /tmp/ipython-input-2540013830.py:6: FutureWarning: YF.download() has changed argument auto_adjust default to True
      data = yf.download(tickers, start="2020-01-01", interval="1d")['Close']
    [*********************100%***********************]  13 of 13 completed
    /tmp/ipython-input-2540013830.py:7: FutureWarning: The default fill_method='pad' in DataFrame.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
      returns = data.pct_change().dropna()
    



  <div id="df-2406918e-403a-4d8e-9549-2eabecf16797" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Ticker</th>
      <th>KLAC</th>
      <th>LRCX</th>
      <th>V</th>
      <th>Volatilidad anual</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EUR=X</th>
      <td>-0.0201</td>
      <td>-0.0071</td>
      <td>-0.0191</td>
      <td>0.0762</td>
    </tr>
    <tr>
      <th>JPY=X</th>
      <td>-0.0293</td>
      <td>-0.0074</td>
      <td>-0.0100</td>
      <td>0.0945</td>
    </tr>
    <tr>
      <th>GLD</th>
      <td>0.1162</td>
      <td>0.1411</td>
      <td>0.0523</td>
      <td>0.1563</td>
    </tr>
    <tr>
      <th>XLP</th>
      <td>0.4202</td>
      <td>0.3990</td>
      <td>0.5996</td>
      <td>0.1644</td>
    </tr>
    <tr>
      <th>DBC</th>
      <td>0.2396</td>
      <td>0.2593</td>
      <td>0.2456</td>
      <td>0.1929</td>
    </tr>
    <tr>
      <th>VEA</th>
      <td>0.6575</td>
      <td>0.6635</td>
      <td>0.6969</td>
      <td>0.1955</td>
    </tr>
    <tr>
      <th>PEN=X</th>
      <td>-0.0251</td>
      <td>-0.0225</td>
      <td>-0.0212</td>
      <td>0.2061</td>
    </tr>
    <tr>
      <th>VWO</th>
      <td>0.6310</td>
      <td>0.6334</td>
      <td>0.5853</td>
      <td>0.2070</td>
    </tr>
    <tr>
      <th>XLU</th>
      <td>0.3556</td>
      <td>0.3376</td>
      <td>0.5206</td>
      <td>0.2236</td>
    </tr>
    <tr>
      <th>SLV</th>
      <td>0.2145</td>
      <td>0.2465</td>
      <td>0.1791</td>
      <td>0.3066</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2406918e-403a-4d8e-9549-2eabecf16797')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2406918e-403a-4d8e-9549-2eabecf16797 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2406918e-403a-4d8e-9549-2eabecf16797');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-c924ef36-815a-4329-907f-df216b11fd94">
      <button class="colab-df-quickchart" onclick="quickchart('df-c924ef36-815a-4329-907f-df216b11fd94')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-c924ef36-815a-4329-907f-df216b11fd94 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
# Calcular la correlación promedio con los activos del portafolio base
resumen['Correlación Promedio'] = resumen[base].mean(axis=1)

# Definir la fórmula de score (pondera diversificación y riesgo)
peso_corr = 0.9  # Peso de la correlación promedio
peso_vol = 1 - peso_corr   # Peso de la volatilidad

# Normalizar variables para poder combinarlas
resumen['Z_Correlación'] = (resumen['Correlación Promedio'] - resumen['Correlación Promedio'].mean()) / resumen['Correlación Promedio'].std()
resumen['Z_Volatilidad'] = (resumen['Volatilidad anual'] - resumen['Volatilidad anual'].mean()) / resumen['Volatilidad anual'].std()

# Score final
resumen['Score Total'] = peso_corr * resumen['Z_Correlación'] + peso_vol * resumen['Z_Volatilidad']

# Selección del mejor candidato
mejor_activo = resumen.sort_values('Score Total').iloc[0]

# Mostrar top 3
print("\n Top 3 activos más recomendables:")
display(resumen.sort_values('Score Total').head(3))
```

    
     Top 3 activos más recomendables:
    



  <div id="df-4639a443-d481-4142-82ea-73ce3618a5ac" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Ticker</th>
      <th>KLAC</th>
      <th>LRCX</th>
      <th>V</th>
      <th>Volatilidad anual</th>
      <th>Correlación Promedio</th>
      <th>Z_Correlación</th>
      <th>Z_Volatilidad</th>
      <th>Score Total</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EUR=X</th>
      <td>-0.020054</td>
      <td>-0.007120</td>
      <td>-0.019073</td>
      <td>0.076219</td>
      <td>-0.015416</td>
      <td>-1.077344</td>
      <td>-1.620657</td>
      <td>-1.131675</td>
    </tr>
    <tr>
      <th>JPY=X</th>
      <td>-0.029342</td>
      <td>-0.007351</td>
      <td>-0.009986</td>
      <td>0.094494</td>
      <td>-0.015559</td>
      <td>-1.077890</td>
      <td>-1.341502</td>
      <td>-1.104252</td>
    </tr>
    <tr>
      <th>PEN=X</th>
      <td>-0.025081</td>
      <td>-0.022521</td>
      <td>-0.021182</td>
      <td>0.206102</td>
      <td>-0.022928</td>
      <td>-1.105924</td>
      <td>0.363365</td>
      <td>-0.958995</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4639a443-d481-4142-82ea-73ce3618a5ac')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4639a443-d481-4142-82ea-73ce3618a5ac button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4639a443-d481-4142-82ea-73ce3618a5ac');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-a575d055-aa71-446b-bd58-32ba9041982b">
      <button class="colab-df-quickchart" onclick="quickchart('df-a575d055-aa71-446b-bd58-32ba9041982b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-a575d055-aa71-446b-bd58-32ba9041982b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
# Selección del mejor candidato
mejor_activo_row = resumen.sort_values('Score Total').iloc[0]
mejor_ticker = mejor_activo_row.name  # El índice es el ticker

# Se guarda el mejor activo diversificador:
activo_diversificador = [mejor_ticker]
```

### Activos Seleccionados


```python
# Activos seleccionados + benchmark
activos = base + mejor_bono + activo_diversificador
benchmark = 'SPY'
tickers = activos + [benchmark]
tickers

# Horizonte de análisis
start_date = "2020-01-02"

# Precios mensuales
data = yf.download(tickers, start=start_date, interval="1d")['Close'].ffill()
```

    /tmp/ipython-input-4199177591.py:11: FutureWarning: YF.download() has changed argument auto_adjust default to True
      data = yf.download(tickers, start=start_date, interval="1d")['Close'].ffill()
    [*********************100%***********************]  6 of 6 completed
    


```python
# Tomar el último valor de cada año calendario (close anual “a fin de año”)
px_y = data.resample("Y").last()
# Asegurar intersección de años para todos
px_y = px_y.dropna(how="all")

# Normalizar cada serie a base 1 en su primer año disponible
px_y_norm = px_y.apply(lambda s: s / s.dropna().iloc[0])

# 2) Parámetros de la grilla
n = len(activos)
ncols = 3  # ajusta a 3 o 4 según prefieras
nrows = ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.3*nrows), sharex=True)
axes = np.atleast_2d(axes)

years = px_y_norm.index.year

for i, tk in enumerate(activos):
    r = i // ncols
    c = i % ncols
    ax = axes[r, c]

    # Series del activo y benchmark alineadas por años disponibles
    s_act = px_y_norm[tk].dropna()
    s_bmk = px_y_norm[benchmark].dropna()

    # Intersección de años
    idx = s_act.index.intersection(s_bmk.index)
    s_act = s_act.loc[idx]
    s_bmk = s_bmk.loc[idx]

    # Plot
    ax.plot(s_act.index.year, s_act.values, marker='o', linewidth=2)
    ax.plot(s_bmk.index.year, s_bmk.values, linestyle='--', linewidth=1.8)

    ax.set_title(f"{tk} por Año")
    ax.grid(True, alpha=0.3)
    if r == nrows - 1:
        ax.set_xlabel("Año")
    ax.set_ylabel("")  # limpio para que el panel sea más visual

# Si sobran ejes (cuando n no llena la grilla), los ocultamos
for j in range(i + 1, nrows * ncols):
    r = j // ncols
    c = j % ncols
    axes[r, c].axis("off")

# Leyenda global (una vez)
lines_labels = [("Activo", dict(linewidth=2, linestyle='-')),
                (benchmark, dict(linewidth=1.8, linestyle='--'))]
handles = [plt.Line2D([0], [0], **style) for _, style in lines_labels]
labels = [lbl for lbl, _ in lines_labels]
fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
```

    /tmp/ipython-input-1439723770.py:2: FutureWarning: 'Y' is deprecated and will be removed in a future version, please use 'YE' instead.
      px_y = data.resample("Y").last()
    


    
![png](output_19_1.png)
    


## 2. Restricciones Regulatorias

- No apalancamiento ni ventas en corto:	Pesos ≥ 0
- Máx. 25% por emisor (empresa individual):	Peso máx. por activo: 0.25
- Máx. 80% en conjunto de emisores >10%:	Suma de pesos > 0.1 debe ser ≤ 0.80
- Mín. 30% en instrumentos de renta fija/garantizados:	SHY u otro instrumento conservador ≥ 0.30
- Máx. 20% en activos no tradicionales:	Activo diversificador ≤ 0.20
- Suma de pesos = 100%:	Restricción de igualdad


```python
bounds = [
    (0, 0.25),  # acción 1: mínimo 0%, máximo 25%
    (0, 0.25),  # acción 2: igual
    (0, 0.25),  # acción 3: igual
    (0.30, 1.00),  # SHY: mínimo 35% por regulación SBS
    (0, 0.20)   # activo diversificador: máximo 20%
]

restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

def restric_sbs(w):
    return 0.80 - np.sum([wi for wi in w if wi > 0.10])

restricciones.append({'type': 'ineq', 'fun': restric_sbs})
```

## 3. Optimización del Portafolio

### Obtener datos


```python
# Retornos
returns = data.pct_change().dropna()

# Subdividir retornos
ret_activos = returns[activos]
ret_benchmark = returns[benchmark]

# Métricas básicas
mean_returns = ret_activos.mean() * 252
cov_matrix = ret_activos.cov() * 252
```


```python
# Fechas automáticas: de hoy hacia 1 año atrás
end = datetime.today()
start = end - timedelta(days=365)

# Descargar la tasa del Treasury a 5 años (GS5) desde FRED
rf = pdr.DataReader('GS5', 'fred', start, end).mean().values[0] / 100  # Convertido a proporción

print(f"Tasa libre de riesgo promedio (GS5) en el último año: {rf:.4%}")
```

    Tasa libre de riesgo promedio (GS5) en el último año: 4.0436%
    

### Funciones Objetivos


```python
# Ratio de Sharpe
def neg_sharpe(w, mean_returns, cov_matrix, rf):
    port_return = np.dot(w, mean_returns)
    port_vol = np.sqrt(w.T @ cov_matrix @ w)
    return -(port_return - rf) / port_vol

# Mínima Varianza
def varianza(w, cov_matrix):
    return w.T @ cov_matrix @ w

# Ratio Sortino
def neg_sortino(w, ret_activos, rf):
    # Retornos diarios del portafolio
    rp = (ret_activos @ w)
    mean_a = rp.mean() * 252
    downside = rp[rp < 0].std(ddof=1) * np.sqrt(252)
    # Evitar división por cero
    if downside == 0 or np.isnan(downside):
        return np.inf
    return -(mean_a - rf) / downside

# Valor Inicial

w0 = np.array([0.2]*5)
```

### Optimizar los modelos


```python
# Sharpe
res_sharpe = minimize(neg_sharpe, w0, args=(mean_returns, cov_matrix, rf), method='SLSQP', bounds=bounds, constraints=restricciones)

# Mínima Varianza
res_var = minimize(varianza, w0, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=restricciones)

# Sortino
res_sortino = minimize(neg_sortino, w0, args=(ret_activos, rf), method='SLSQP', bounds=bounds, constraints=restricciones)
```

### Consolidar y Comparar Portafolios


```python
# Inicializa DataFrame vacío
resultados = pd.DataFrame(columns=['Modelo', 'Return', 'Volatility', 'Sharpe'])

# Construir tabla resumen
def port_stats(w):
    r = np.dot(w, mean_returns)
    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    s = (r - rf) / v
    return [r, v, s]

# Agrega cada portafolio
nombres = ['Sharpe', 'Mínima Varianza', 'Sortino']
for nombre, res in zip(nombres, [res_sharpe, res_var, res_sortino]):
    stats = port_stats(res.x)  # Tomamos Return, Volatility, Sharpe
    resultados.loc[len(resultados)] = [nombre] + stats

# Agrega el benchmark SPY
spy_ret = ret_benchmark.mean() * 252
spy_vol = ret_benchmark.std() * np.sqrt(252)
spy_sharpe = (spy_ret - rf) / spy_vol

resultados.loc[len(resultados)] = ['SPY (Benchmark)', spy_ret, spy_vol, spy_sharpe]

resultados
```





  <div id="df-7be212e0-3c82-4205-a862-482f90ea2874" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Modelo</th>
      <th>Return</th>
      <th>Volatility</th>
      <th>Sharpe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sharpe</td>
      <td>0.138789</td>
      <td>0.159654</td>
      <td>0.616037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mínima Varianza</td>
      <td>0.046159</td>
      <td>0.050596</td>
      <td>0.113101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sortino</td>
      <td>0.148104</td>
      <td>0.170943</td>
      <td>0.629845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SPY (Benchmark)</td>
      <td>0.153412</td>
      <td>0.208483</td>
      <td>0.541892</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7be212e0-3c82-4205-a862-482f90ea2874')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7be212e0-3c82-4205-a862-482f90ea2874 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7be212e0-3c82-4205-a862-482f90ea2874');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-c696daf8-d3e1-4ea4-aa1e-40af3c9a3737">
      <button class="colab-df-quickchart" onclick="quickchart('df-c696daf8-d3e1-4ea4-aa1e-40af3c9a3737')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-c696daf8-d3e1-4ea4-aa1e-40af3c9a3737 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_c208eb34-dd0a-42cb-8280-612a3f4581d9">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('resultados')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c208eb34-dd0a-42cb-8280-612a3f4581d9 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('resultados');
      }
      })();
    </script>
  </div>

    </div>
  </div>




### Gráfica Comparativa


```python
# Diccionario con portafolios optimizados
portafolios = {
    'Sharpe': res_sharpe.x,
    'Min Varianza': res_var.x,
    'Sortino': res_sortino.x
}

# Calcular precios acumulados de cada portafolio
precios_portafolios = {}

for nombre, pesos in portafolios.items():
    ret_port = ret_activos @ pesos
    precios_portafolios[nombre] = (1 + ret_port).cumprod()

# Agregar benchmark
precios_portafolios['Benchmark'] = (1 + ret_benchmark).cumprod()

# Unir en un DataFrame para graficar
df_comp = pd.DataFrame(precios_portafolios)

# Gráfico comparativo
df_comp.plot(figsize=(10, 6), title='Comparación Histórica: Portafolios vs Benchmark')
plt.ylabel('Crecimiento de $1')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_34_0.png)
    


### Frontera Eficiente


```python
n_portfolios = 100000
results = []
weights_record = []

np.random.seed(42)
count_valid = 0

for _ in range(n_portfolios):
    # Rango de pesos por activo según SBS:
    w1 = np.random.uniform(0, 0.25)   # acción 1
    w2 = np.random.uniform(0, 0.25)   # acción 2
    w3 = np.random.uniform(0, 0.25)   # acción 3
    w4 = np.random.uniform(0.30, 1.00)  # Bono
    w5 = np.random.uniform(0, 0.20)   # diversificador

    w = np.array([w1, w2, w3, w4, w5])
    w = w / np.sum(w)  # normalizar para que suma total = 1

    # Verificar restricción SBS adicional (máx. 80% en activos >10%)
    if np.sum([wi for wi in w if wi > 0.10]) <= 0.80:
        r = np.dot(w, mean_returns) * 100
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * 100
        s = (r / 100 - rf) / (v / 100)

        results.append((r, v, s))
        weights_record.append(w)
        count_valid += 1

print(f"Portafolios válidos generados: {count_valid}")

# Convertir a arrays
results = np.array(results)
returns_sim = results[:, 0]
vol_sim = results[:, 1]
sharpe_sim = results[:, 2]

# Extraer los portafolios optimizados ya calculados
w_sharpe = res_sharpe.x
w_var = res_var.x
w_sortino = res_sortino.x

def get_stats(w):
    r = np.dot(w, mean_returns) * 100
    v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * 100
    return r, v

r_sharpe, v_sharpe = get_stats(w_sharpe)
r_var, v_var = get_stats(w_var)
r_sortino, v_sortino = get_stats(w_sortino)

# SPY
r_spy = spy_ret * 100
v_spy = spy_vol * 100

# Plot
plt.figure(figsize=(10, 7))
sc = plt.scatter(vol_sim, returns_sim, c=sharpe_sim, cmap='viridis', alpha=0.5, label='Portafolios Simulados')
plt.colorbar(sc, label='Sharpe Ratio')

# Marcar portafolios óptimos y benchmark
plt.scatter(v_sharpe, r_sharpe, color='green', marker='*', s=200, label='Máx Sharpe')
plt.scatter(v_var, r_var, color='blue', marker='*', s=200, label='Mín Varianza')
plt.scatter(v_sortino, r_sortino, color='orange', marker='*', s=200, label='Máx Sortino')
plt.scatter(v_spy, r_spy, color='black', marker='X', s=100, label='Benchmark')

plt.title('Frontera Eficiente bajo Restricciones Regulatorias')
plt.xlabel('Volatilidad (%)')
plt.ylabel('Retorno Esperado (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

    Portafolios válidos generados: 6757
    


    
![png](output_36_1.png)
    


## 4. Cálculo de las Métricas de Riesgo

### Parámetros Generales


```python
confianza = 0.95
z = norm.ppf(1 - confianza)
horizonte = 10  # días
monto_invertido = 100000  # USD
n_simulaciones = 100000 # Para Monte Carlo
lambda_=0.94 # Para EWMA
```

### VaR Delta-Normal


```python
# Función para calcular VaR Delta-Normal
def calcular_var_delta_normal(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):

    # Volatilidad diaria por activo
    vol_individual = retornos.std()

    # VaR individual porcentual
    var_ind_pct = -z * vol_individual * np.array(w) * np.sqrt(horizonte)

    # VaR individual absoluto en USD
    var_ind_usd = var_ind_pct * monto_invertido

    # VaR de portafolio total
    cov = retornos.cov()
    port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
    var_port_pct = -z * port_vol * np.sqrt(horizonte)
    var_port_usd = var_port_pct * monto_invertido

    return var_ind_pct, var_ind_usd, var_port_pct, var_port_usd

# Diccionarios para almacenar resultados
var_pct_dict = {}
var_usd_dict = {}
var_portafolio_pct = {}
var_portafolio_usd = {}

# Calcular VaR para cada portafolio
for nombre, pesos in portafolios.items():
    v_pct_ind, v_usd_ind, v_pct_port, v_usd_port = calcular_var_delta_normal(
        pesos, ret_activos)

    var_pct_dict[nombre] = v_pct_ind * 100  # a porcentaje
    var_usd_dict[nombre] = v_usd_ind
    var_portafolio_pct[nombre] = [v_pct_port * 100]
    var_portafolio_usd[nombre] = [v_usd_port]

# DataFrames finales

# En porcentaje (% del monto invertido)
df_var_individual_pct = pd.DataFrame(var_pct_dict, index=activos)
df_var_portafolio_pct = pd.DataFrame(var_portafolio_pct, index=['VaR Portafolio %'])

# En dólares (USD)
df_var_individual_usd = pd.DataFrame(var_usd_dict, index=activos)
df_var_portafolio_usd = pd.DataFrame(var_portafolio_usd, index=['VaR Portafolio USD'])

# Mostrar resultados
print("VaR Individual (%)")
display(df_var_individual_pct)

print("VaR del Portafolio (%)")
display(df_var_portafolio_pct)

print("VaR Individual (USD)")
display(df_var_individual_usd)

print("VaR del Portafolio (USD)")
display(df_var_portafolio_usd)
```

    VaR Individual (%)
    



  <div id="df-b0fc6ec8-c0e7-427c-ae05-7d811413b36d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>3.722461</td>
      <td>5.139760e-16</td>
      <td>3.722461</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1.573794</td>
      <td>1.574370e+00</td>
      <td>1.572009</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.057106</td>
      <td>1.337790e-02</td>
      <td>0.701170</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>0.329269</td>
      <td>4.845817e-01</td>
      <td>0.285313</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>0.249742</td>
      <td>2.459720e-01</td>
      <td>0.249742</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b0fc6ec8-c0e7-427c-ae05-7d811413b36d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b0fc6ec8-c0e7-427c-ae05-7d811413b36d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b0fc6ec8-c0e7-427c-ae05-7d811413b36d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-85a0a1ee-f3fc-4544-8ab7-97045f59e501">
      <button class="colab-df-quickchart" onclick="quickchart('df-85a0a1ee-f3fc-4544-8ab7-97045f59e501')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-85a0a1ee-f3fc-4544-8ab7-97045f59e501 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_fada8d03-6acb-4416-9216-07f84b2b2927">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_individual_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_fada8d03-6acb-4416-9216-07f84b2b2927 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_individual_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR del Portafolio (%)
    



  <div id="df-d82bd873-10e5-42d3-9277-3930014c5e6f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio %</th>
      <td>5.231266</td>
      <td>1.657854</td>
      <td>5.60118</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d82bd873-10e5-42d3-9277-3930014c5e6f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d82bd873-10e5-42d3-9277-3930014c5e6f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d82bd873-10e5-42d3-9277-3930014c5e6f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_f1f9fc55-54f9-4dd6-ae77-b0235c4581bb">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_portafolio_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_f1f9fc55-54f9-4dd6-ae77-b0235c4581bb button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_portafolio_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Individual (USD)
    



  <div id="df-92ff43cb-0e2f-4a93-adaa-5b0f5a9e83e3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>3722.461393</td>
      <td>5.139760e-13</td>
      <td>3722.461299</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1573.793700</td>
      <td>1.574370e+03</td>
      <td>1572.008760</td>
    </tr>
    <tr>
      <th>V</th>
      <td>57.105905</td>
      <td>1.337790e+01</td>
      <td>701.169559</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>329.268783</td>
      <td>4.845817e+02</td>
      <td>285.313362</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>249.741951</td>
      <td>2.459720e+02</td>
      <td>249.742038</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-92ff43cb-0e2f-4a93-adaa-5b0f5a9e83e3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-92ff43cb-0e2f-4a93-adaa-5b0f5a9e83e3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-92ff43cb-0e2f-4a93-adaa-5b0f5a9e83e3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-1169fa92-61b8-4110-baab-9fa2dd776e6f">
      <button class="colab-df-quickchart" onclick="quickchart('df-1169fa92-61b8-4110-baab-9fa2dd776e6f')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-1169fa92-61b8-4110-baab-9fa2dd776e6f button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_ed923392-df0f-4d2c-952b-db58b41dbe74">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_individual_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ed923392-df0f-4d2c-952b-db58b41dbe74 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_individual_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR del Portafolio (USD)
    



  <div id="df-40d7f024-fc33-4f45-adad-45992bb399d2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio USD</th>
      <td>5231.266129</td>
      <td>1657.853568</td>
      <td>5601.180052</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-40d7f024-fc33-4f45-adad-45992bb399d2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-40d7f024-fc33-4f45-adad-45992bb399d2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-40d7f024-fc33-4f45-adad-45992bb399d2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_9f7df389-0953-44ad-92d3-2c6bdfe2c32a">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_portafolio_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_9f7df389-0953-44ad-92d3-2c6bdfe2c32a button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_portafolio_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### VaR Histórico


```python
def calcular_var_historico(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    # Portafolio: pasamos a log-retornos para agregar en el tiempo
    ret_port = retornos @ w
    log_port = np.log1p(ret_port)
    log_T = log_port.rolling(horizonte).sum().dropna()
    ret_T = np.expm1(log_T)  # retorno compuesto a T días

    var_port_pct = -np.percentile(ret_T, (1 - nivel) * 100.0)
    var_port_usd = var_port_pct * monto_invertido

    # VaR "individual ponderado" a T días (mismo criterio que el resto del script)
    var_ind_pct, var_ind_usd = [], []
    for i, col in enumerate(retornos.columns):
        log_i = np.log1p(retornos[col]).rolling(horizonte).sum().dropna()
        ret_i_T = np.expm1(log_i)
        var_i_pct = -np.percentile(ret_i_T, (1 - nivel) * 100.0)
        var_ind_pct.append(var_i_pct * w[i])
        var_ind_usd.append(var_i_pct * w[i] * monto_invertido)

    return np.array(var_ind_pct) * 100.0, np.array(var_ind_usd), var_port_pct * 100.0, var_port_usd

# Diccionarios de resultados
var_hist_pct_dict = {}
var_hist_usd_dict = {}
var_hist_portafolio_pct = {}
var_hist_portafolio_usd = {}

for nombre, pesos in portafolios.items():
    v_pct_ind, v_usd_ind, v_pct_port, v_usd_port = calcular_var_historico(
        pesos, ret_activos)

    var_hist_pct_dict[nombre] = v_pct_ind
    var_hist_usd_dict[nombre] = v_usd_ind
    var_hist_portafolio_pct[nombre] = [v_pct_port]
    var_hist_portafolio_usd[nombre] = [v_usd_port]

# En porcentaje
df_var_hist_ind_pct = pd.DataFrame(var_hist_pct_dict, index=activos)
df_var_hist_port_pct = pd.DataFrame(var_hist_portafolio_pct, index=['VaR Portafolio %'])

# En USD
df_var_hist_ind_usd = pd.DataFrame(var_hist_usd_dict, index=activos)
df_var_hist_port_usd = pd.DataFrame(var_hist_portafolio_usd, index=['VaR Portafolio USD'])

# Mostrar
print("VaR Histórico Individual (%)")
display(df_var_hist_ind_pct)

print("VaR Histórico del Portafolio (%)")
display(df_var_hist_port_pct)

print("VaR Histórico Individual (USD)")
display(df_var_hist_ind_usd)

print("VaR Histórico del Portafolio (USD)")
display(df_var_hist_port_usd)
```

    VaR Histórico Individual (%)
    



  <div id="df-687f70fe-1dc8-405a-97af-3b5f62e6f0eb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>2.731091</td>
      <td>3.770933e-16</td>
      <td>2.731091</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1.258582</td>
      <td>1.259043e+00</td>
      <td>1.257154</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.047119</td>
      <td>1.103842e-02</td>
      <td>0.578551</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>0.270576</td>
      <td>3.982037e-01</td>
      <td>0.234455</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>0.240541</td>
      <td>2.369099e-01</td>
      <td>0.240541</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-687f70fe-1dc8-405a-97af-3b5f62e6f0eb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-687f70fe-1dc8-405a-97af-3b5f62e6f0eb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-687f70fe-1dc8-405a-97af-3b5f62e6f0eb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-88c4e32d-a794-44d6-afaa-e56ad88d3625">
      <button class="colab-df-quickchart" onclick="quickchart('df-88c4e32d-a794-44d6-afaa-e56ad88d3625')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-88c4e32d-a794-44d6-afaa-e56ad88d3625 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_3986490f-ba31-4232-995e-3cf167224e60">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_hist_ind_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_3986490f-ba31-4232-995e-3cf167224e60 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_hist_ind_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Histórico del Portafolio (%)
    



  <div id="df-8ccb71d9-b528-4916-bbbb-b4611f357aff" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio %</th>
      <td>3.982362</td>
      <td>1.235557</td>
      <td>4.263389</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8ccb71d9-b528-4916-bbbb-b4611f357aff')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8ccb71d9-b528-4916-bbbb-b4611f357aff button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8ccb71d9-b528-4916-bbbb-b4611f357aff');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_6073766d-d1c9-4af8-a691-eddd141ab190">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_hist_port_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_6073766d-d1c9-4af8-a691-eddd141ab190 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_hist_port_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Histórico Individual (USD)
    



  <div id="df-d9ec451f-3b2a-496a-ac7f-8fb47ffea67b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>2731.090844</td>
      <td>3.770933e-13</td>
      <td>2731.090774</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1258.581702</td>
      <td>1.259043e+03</td>
      <td>1257.154263</td>
    </tr>
    <tr>
      <th>V</th>
      <td>47.119408</td>
      <td>1.103842e+01</td>
      <td>578.551277</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>270.575670</td>
      <td>3.982037e+02</td>
      <td>234.455430</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>240.540961</td>
      <td>2.369099e+02</td>
      <td>240.541045</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d9ec451f-3b2a-496a-ac7f-8fb47ffea67b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d9ec451f-3b2a-496a-ac7f-8fb47ffea67b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d9ec451f-3b2a-496a-ac7f-8fb47ffea67b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-eb4c4a7a-b216-4c7f-a3d1-6d39c611ea9d">
      <button class="colab-df-quickchart" onclick="quickchart('df-eb4c4a7a-b216-4c7f-a3d1-6d39c611ea9d')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-eb4c4a7a-b216-4c7f-a3d1-6d39c611ea9d button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_22e12c98-d435-4d1f-b713-5855362ec44e">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_hist_ind_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_22e12c98-d435-4d1f-b713-5855362ec44e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_hist_ind_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Histórico del Portafolio (USD)
    



  <div id="df-85c2cedd-9a5e-4e44-8998-4a959f8df0aa" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio USD</th>
      <td>3982.361508</td>
      <td>1235.557098</td>
      <td>4263.38852</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-85c2cedd-9a5e-4e44-8998-4a959f8df0aa')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-85c2cedd-9a5e-4e44-8998-4a959f8df0aa button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-85c2cedd-9a5e-4e44-8998-4a959f8df0aa');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_bb121a4b-3cfe-4d5a-9614-dffdf841a059">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_hist_port_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_bb121a4b-3cfe-4d5a-9614-dffdf841a059 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_hist_port_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### VaR Monte Carlo


```python
def calcular_var_monte_carlo(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    # 1) Estadísticas en log-retornos diarios
    logrets = np.log1p(retornos)
    mu = logrets.mean().values              # vector mu_diario (log)
    sigma = logrets.std(ddof=1).values      # vector sigma_diario (log)
    corr = np.corrcoef(logrets.T)           # correlación entre activos (log)
    n_activos = len(w)

    # 2) Cholesky de correlación para choques correlacionados
    L = np.linalg.cholesky(corr)

    # 3) Simular shocks Z ~ N(0, I) y correlacionarlos
    Z = np.random.randn(n_simulaciones, n_activos) @ L.T   # (n_sim, n_activos)

    # 4) Escalar a horizonte T: log-retorno GBM de cada activo
    T = float(horizonte)   # días
    drift = (mu - 0.5 * sigma**2) * T
    vol_T = sigma * np.sqrt(T)
    logR = drift + Z * vol_T          # log(S_T/S_0)
    R = np.exp(logR) - 1.0            # retorno compuesto a T

    # 5) Retorno del portafolio a T (pesos fijos, aprox. lineal en retornos)
    port_ret_sim = R @ w

    # 6) VaR de portafolio
    var_port_pct = -np.percentile(port_ret_sim, (1 - nivel) * 100.0)
    var_port_usd = var_port_pct * monto_invertido

    # 7) VaR "individual ponderado" (misma lógica que tu código actual)
    var_ind_pct = []
    var_ind_usd = []
    for i in range(n_activos):
        var_i_pct = -np.percentile(R[:, i], (1 - nivel) * 100.0)
        var_ind_pct.append(var_i_pct * w[i])
        var_ind_usd.append(var_i_pct * w[i] * monto_invertido)

    return np.array(var_ind_pct) * 100.0, np.array(var_ind_usd), var_port_pct * 100.0, var_port_usd

  # Diccionarios de resultados
var_mc_pct_dict = {}
var_mc_usd_dict = {}
var_mc_portafolio_pct = {}
var_mc_portafolio_usd = {}

for nombre, pesos in portafolios.items():
    v_pct_ind, v_usd_ind, v_pct_port, v_usd_port = calcular_var_monte_carlo(
        pesos, ret_activos)

    var_mc_pct_dict[nombre] = v_pct_ind
    var_mc_usd_dict[nombre] = v_usd_ind
    var_mc_portafolio_pct[nombre] = [v_pct_port]
    var_mc_portafolio_usd[nombre] = [v_usd_port]

# En porcentaje
df_var_mc_ind_pct = pd.DataFrame(var_mc_pct_dict, index=activos)
df_var_mc_port_pct = pd.DataFrame(var_mc_portafolio_pct, index=['VaR Portafolio %'])

# En USD
df_var_mc_ind_usd = pd.DataFrame(var_mc_usd_dict, index=activos)
df_var_mc_port_usd = pd.DataFrame(var_mc_portafolio_usd, index=['VaR Portafolio USD'])

# Mostrar
print("VaR Monte Carlo Individual (%)")
display(df_var_mc_ind_pct)

print("VaR Monte Carlo del Portafolio (%)")
display(df_var_mc_port_pct)

print("VaR Monte Carlo Individual (USD)")
display(df_var_mc_ind_usd)

print("VaR Monte Carlo del Portafolio (USD)")
display(df_var_mc_port_usd)
```

    VaR Monte Carlo Individual (%)
    



  <div id="df-7431d6c0-6775-4369-ba9c-b5b39ce1184a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>3.292357</td>
      <td>4.571690e-16</td>
      <td>3.301797</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1.421872</td>
      <td>1.428330e+00</td>
      <td>1.424647</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.052989</td>
      <td>1.252310e-02</td>
      <td>0.647863</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>0.295951</td>
      <td>4.375475e-01</td>
      <td>0.253373</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>0.250976</td>
      <td>2.478706e-01</td>
      <td>0.252187</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7431d6c0-6775-4369-ba9c-b5b39ce1184a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7431d6c0-6775-4369-ba9c-b5b39ce1184a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7431d6c0-6775-4369-ba9c-b5b39ce1184a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f718bb2f-9e8f-4d6a-8b10-786147be9485">
      <button class="colab-df-quickchart" onclick="quickchart('df-f718bb2f-9e8f-4d6a-8b10-786147be9485')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f718bb2f-9e8f-4d6a-8b10-786147be9485 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_45f73dcf-7dbc-4f8c-a513-ffa2ef86b173">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_mc_ind_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_45f73dcf-7dbc-4f8c-a513-ffa2ef86b173 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_mc_ind_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Monte Carlo del Portafolio (%)
    



  <div id="df-9b817a97-33ad-4bdd-af33-a520f5d264e5" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio %</th>
      <td>4.621418</td>
      <td>1.475371</td>
      <td>4.986976</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9b817a97-33ad-4bdd-af33-a520f5d264e5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9b817a97-33ad-4bdd-af33-a520f5d264e5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9b817a97-33ad-4bdd-af33-a520f5d264e5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_d2ce2472-719c-4179-8a1a-36a8046fe8a9">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_mc_port_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d2ce2472-719c-4179-8a1a-36a8046fe8a9 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_mc_port_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Monte Carlo Individual (USD)
    



  <div id="df-7dec0734-20c8-4dab-8f70-7a5d6621acac" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>3292.357328</td>
      <td>4.571690e-13</td>
      <td>3301.796909</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1421.872130</td>
      <td>1.428330e+03</td>
      <td>1424.647178</td>
    </tr>
    <tr>
      <th>V</th>
      <td>52.989091</td>
      <td>1.252310e+01</td>
      <td>647.863170</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>295.950974</td>
      <td>4.375475e+02</td>
      <td>253.373402</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>250.975579</td>
      <td>2.478706e+02</td>
      <td>252.186779</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7dec0734-20c8-4dab-8f70-7a5d6621acac')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7dec0734-20c8-4dab-8f70-7a5d6621acac button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7dec0734-20c8-4dab-8f70-7a5d6621acac');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-e5c66bc1-14f8-41f2-a616-c0dd62c4b423">
      <button class="colab-df-quickchart" onclick="quickchart('df-e5c66bc1-14f8-41f2-a616-c0dd62c4b423')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-e5c66bc1-14f8-41f2-a616-c0dd62c4b423 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_1cb11991-831f-40a8-855c-9a737dad2997">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_mc_ind_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_1cb11991-831f-40a8-855c-9a737dad2997 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_mc_ind_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR Monte Carlo del Portafolio (USD)
    



  <div id="df-f907bf9b-a785-457c-a88a-12cf39b23b00" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio USD</th>
      <td>4621.417905</td>
      <td>1475.370744</td>
      <td>4986.975925</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f907bf9b-a785-457c-a88a-12cf39b23b00')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f907bf9b-a785-457c-a88a-12cf39b23b00 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f907bf9b-a785-457c-a88a-12cf39b23b00');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_507d2c9c-f557-4911-9c28-61cf86b5551a">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_mc_port_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_507d2c9c-f557-4911-9c28-61cf86b5551a button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_mc_port_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### VaR EWMA


```python
def calcular_ewma_vol(retornos):
    ewma_var = []
    for serie in retornos.T.values:  # iteramos por columna
        var_t = np.var(serie)  # varianza inicial
        ewma = [var_t]
        for r in serie[1:]:
            var_t = lambda_ * var_t + (1 - lambda_) * r**2
            ewma.append(var_t)
        ewma_var.append(np.sqrt(ewma[-1]))  # última desviación estándar
    return np.array(ewma_var)

def calcular_var_ewma(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    z = norm.ppf(1 - confianza)

    # Volatilidad EWMA individual
    ewma_vol_ind = calcular_ewma_vol(retornos)
    var_ind_pct = -z * ewma_vol_ind * np.array(w) * np.sqrt(horizonte)
    var_ind_usd = var_ind_pct * monto_invertido

    # Varianza EWMA del portafolio
    cov = retornos.cov().values
    ewma_cov = np.outer(ewma_vol_ind, ewma_vol_ind) * np.corrcoef(retornos.T)
    port_vol = np.sqrt(np.dot(w, np.dot(ewma_cov, w)))
    var_port_pct = -z * port_vol * np.sqrt(horizonte)
    var_port_usd = var_port_pct * monto_invertido

    return var_ind_pct * 100, var_ind_usd, var_port_pct * 100, var_port_usd

# Diccionarios para guardar resultados
var_ewma_pct_dict = {}
var_ewma_usd_dict = {}
var_ewma_portafolio_pct = {}
var_ewma_portafolio_usd = {}

for nombre, pesos in portafolios.items():
    v_pct_ind, v_usd_ind, v_pct_port, v_usd_port = calcular_var_ewma(
        pesos, ret_activos)

    var_ewma_pct_dict[nombre] = v_pct_ind
    var_ewma_usd_dict[nombre] = v_usd_ind
    var_ewma_portafolio_pct[nombre] = [v_pct_port]
    var_ewma_portafolio_usd[nombre] = [v_usd_port]

# En porcentaje
df_var_ewma_ind_pct = pd.DataFrame(var_ewma_pct_dict, index=activos)
df_var_ewma_port_pct = pd.DataFrame(var_ewma_portafolio_pct, index=['VaR Portafolio %'])

# En USD
df_var_ewma_ind_usd = pd.DataFrame(var_ewma_usd_dict, index=activos)
df_var_ewma_port_usd = pd.DataFrame(var_ewma_portafolio_usd, index=['VaR Portafolio USD'])

# Mostrar
print("VaR EWMA Individual (%)")
display(df_var_ewma_ind_pct)

print("VaR EWMA del Portafolio (%)")
display(df_var_ewma_port_pct)

print("VaR EWMA Individual (USD)")
display(df_var_ewma_ind_usd)

print("VaR EWMA del Portafolio (USD)")
display(df_var_ewma_port_usd)
```

    VaR EWMA Individual (%)
    



  <div id="df-c8953080-5642-4495-9620-215893c19a7b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>3.589456</td>
      <td>4.956115e-16</td>
      <td>3.589456</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1.320323</td>
      <td>1.320806e+00</td>
      <td>1.318825</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.038414</td>
      <td>8.999156e-03</td>
      <td>0.471669</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>0.314789</td>
      <td>4.632721e-01</td>
      <td>0.272767</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>0.273633</td>
      <td>2.695027e-01</td>
      <td>0.273633</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c8953080-5642-4495-9620-215893c19a7b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c8953080-5642-4495-9620-215893c19a7b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c8953080-5642-4495-9620-215893c19a7b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-724717b6-6e85-4827-99d7-94b453b0b388">
      <button class="colab-df-quickchart" onclick="quickchart('df-724717b6-6e85-4827-99d7-94b453b0b388')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-724717b6-6e85-4827-99d7-94b453b0b388 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_3bb0d97c-ac61-4203-9eda-baf18e1c085e">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_ewma_ind_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_3bb0d97c-ac61-4203-9eda-baf18e1c085e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_ewma_ind_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR EWMA del Portafolio (%)
    



  <div id="df-6349a649-1468-4cf3-b2ff-8c22909a4125" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio %</th>
      <td>4.848696</td>
      <td>1.414907</td>
      <td>5.090973</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6349a649-1468-4cf3-b2ff-8c22909a4125')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6349a649-1468-4cf3-b2ff-8c22909a4125 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6349a649-1468-4cf3-b2ff-8c22909a4125');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_7b87439b-0c11-4f12-8bc8-229e728cefaf">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_ewma_port_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_7b87439b-0c11-4f12-8bc8-229e728cefaf button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_ewma_port_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR EWMA Individual (USD)
    



  <div id="df-6532fa47-895f-4fdd-be51-3c3d58dc1383" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>3589.456441</td>
      <td>4.956115e-13</td>
      <td>3589.456351</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1320.322534</td>
      <td>1.320806e+03</td>
      <td>1318.825071</td>
    </tr>
    <tr>
      <th>V</th>
      <td>38.414474</td>
      <td>8.999156e+00</td>
      <td>471.668553</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>314.789076</td>
      <td>4.632721e+02</td>
      <td>272.766610</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>273.633378</td>
      <td>2.695027e+02</td>
      <td>273.633473</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6532fa47-895f-4fdd-be51-3c3d58dc1383')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6532fa47-895f-4fdd-be51-3c3d58dc1383 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6532fa47-895f-4fdd-be51-3c3d58dc1383');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-ba8324ad-7805-4b8f-b3a7-8d55592d054e">
      <button class="colab-df-quickchart" onclick="quickchart('df-ba8324ad-7805-4b8f-b3a7-8d55592d054e')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-ba8324ad-7805-4b8f-b3a7-8d55592d054e button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_58e8a74a-ad19-4328-9a0b-41e11ee27b5b">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_ewma_ind_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_58e8a74a-ad19-4328-9a0b-41e11ee27b5b button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_ewma_ind_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR EWMA del Portafolio (USD)
    



  <div id="df-f849060e-b7a0-4a15-9f7d-34179f0fedb0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio USD</th>
      <td>4848.696245</td>
      <td>1414.90669</td>
      <td>5090.973213</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f849060e-b7a0-4a15-9f7d-34179f0fedb0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f849060e-b7a0-4a15-9f7d-34179f0fedb0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f849060e-b7a0-4a15-9f7d-34179f0fedb0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_9e335d71-5408-4f9d-8f2e-c571f07b8c84">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_ewma_port_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_9e335d71-5408-4f9d-8f2e-c571f07b8c84 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_ewma_port_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### VaR GARCH


```python
!pip install arch
```

    Requirement already satisfied: arch in /usr/local/lib/python3.11/dist-packages (7.2.0)
    Requirement already satisfied: numpy>=1.22.3 in /usr/local/lib/python3.11/dist-packages (from arch) (2.0.2)
    Requirement already satisfied: scipy>=1.8 in /usr/local/lib/python3.11/dist-packages (from arch) (1.16.1)
    Requirement already satisfied: pandas>=1.4 in /usr/local/lib/python3.11/dist-packages (from arch) (2.2.2)
    Requirement already satisfied: statsmodels>=0.12 in /usr/local/lib/python3.11/dist-packages (from arch) (0.14.5)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.4->arch) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.4->arch) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.4->arch) (2025.2)
    Requirement already satisfied: patsy>=0.5.6 in /usr/local/lib/python3.11/dist-packages (from statsmodels>=0.12->arch) (1.0.1)
    Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.11/dist-packages (from statsmodels>=0.12->arch) (25.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas>=1.4->arch) (1.17.0)
    


```python
from arch import arch_model

def calcular_vol_garch(serie):
    modelo = arch_model(serie * 100, vol='Garch', p=1, q=1, rescale=False)
    resultado = modelo.fit(disp='off')
    sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    return sigma_t

def calcular_var_garch(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    n = len(w)

    # Estimar volatilidad individual con GARCH
    vol_garch_ind = np.array([calcular_vol_garch(retornos[col]) for col in retornos.columns])

    # VaR individual
    var_ind_pct = -z * vol_garch_ind * np.array(w) * np.sqrt(horizonte)
    var_ind_usd = var_ind_pct * monto_invertido

    # Matriz de correlaciones y varianzas individuales
    correlacion = retornos.corr().values
    cov_garch = np.outer(vol_garch_ind, vol_garch_ind) * correlacion

    # VaR de portafolio
    port_vol = np.sqrt(np.dot(w, np.dot(cov_garch, w)))
    var_port_pct = -z * port_vol * np.sqrt(horizonte)
    var_port_usd = var_port_pct * monto_invertido

    return var_ind_pct * 100, var_ind_usd, var_port_pct * 100, var_port_usd

# Diccionarios para resultados
var_garch_pct_dict = {}
var_garch_usd_dict = {}
var_garch_portafolio_pct = {}
var_garch_portafolio_usd = {}

for nombre, pesos in portafolios.items():
    v_pct_ind, v_usd_ind, v_pct_port, v_usd_port = calcular_var_garch(
        pesos, ret_activos)

    var_garch_pct_dict[nombre] = v_pct_ind
    var_garch_usd_dict[nombre] = v_usd_ind
    var_garch_portafolio_pct[nombre] = [v_pct_port]
    var_garch_portafolio_usd[nombre] = [v_usd_port]

# En porcentaje
df_var_garch_ind_pct = pd.DataFrame(var_garch_pct_dict, index=activos)
df_var_garch_port_pct = pd.DataFrame(var_garch_portafolio_pct, index=['VaR Portafolio %'])

# En USD
df_var_garch_ind_usd = pd.DataFrame(var_garch_usd_dict, index=activos)
df_var_garch_port_usd = pd.DataFrame(var_garch_portafolio_usd, index=['VaR Portafolio USD'])

# Mostrar
print("VaR GARCH Individual (%)")
display(df_var_garch_ind_pct)

print("VaR GARCH del Portafolio (%)")
display(df_var_garch_port_pct)

print("VaR GARCH Individual (USD)")
display(df_var_garch_ind_usd)

print("VaR GARCH del Portafolio (USD)")
display(df_var_garch_port_usd)
```

    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    

    VaR GARCH Individual (%)
    

    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    



  <div id="df-d27f5dbb-cf11-4a69-a576-2c0a47ab8b06" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>4.275617</td>
      <td>5.903526e-16</td>
      <td>4.275617</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1.621936</td>
      <td>1.622530e+00</td>
      <td>1.620097</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.041848</td>
      <td>9.803445e-03</td>
      <td>0.513823</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>0.291805</td>
      <td>4.294474e-01</td>
      <td>0.252851</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>0.269014</td>
      <td>2.649535e-01</td>
      <td>0.269015</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d27f5dbb-cf11-4a69-a576-2c0a47ab8b06')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d27f5dbb-cf11-4a69-a576-2c0a47ab8b06 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d27f5dbb-cf11-4a69-a576-2c0a47ab8b06');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d61cac95-cac2-4725-8edd-b2db9a5ee20b">
      <button class="colab-df-quickchart" onclick="quickchart('df-d61cac95-cac2-4725-8edd-b2db9a5ee20b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d61cac95-cac2-4725-8edd-b2db9a5ee20b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_6a6cd44e-d499-4482-b49c-fe031941d5d4">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_garch_ind_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_6a6cd44e-d499-4482-b49c-fe031941d5d4 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_garch_ind_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR GARCH del Portafolio (%)
    



  <div id="df-e6ce0747-330d-4bf2-8996-d9e63c52fb55" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio %</th>
      <td>5.815894</td>
      <td>1.690823</td>
      <td>6.079345</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e6ce0747-330d-4bf2-8996-d9e63c52fb55')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e6ce0747-330d-4bf2-8996-d9e63c52fb55 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e6ce0747-330d-4bf2-8996-d9e63c52fb55');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_90df3001-229b-4d55-b3b8-6435680147af">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_garch_port_pct')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_90df3001-229b-4d55-b3b8-6435680147af button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_garch_port_pct');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR GARCH Individual (USD)
    



  <div id="df-d636d882-b58d-4202-a2b8-974c92e138db" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KLAC</th>
      <td>4275.617144</td>
      <td>5.903526e-13</td>
      <td>4275.617036</td>
    </tr>
    <tr>
      <th>LRCX</th>
      <td>1621.936279</td>
      <td>1.622530e+03</td>
      <td>1620.096737</td>
    </tr>
    <tr>
      <th>V</th>
      <td>41.847720</td>
      <td>9.803445e+00</td>
      <td>513.823347</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>291.805483</td>
      <td>4.294474e+02</td>
      <td>252.851189</td>
    </tr>
    <tr>
      <th>EUR=X</th>
      <td>269.014414</td>
      <td>2.649535e+02</td>
      <td>269.014508</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d636d882-b58d-4202-a2b8-974c92e138db')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d636d882-b58d-4202-a2b8-974c92e138db button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d636d882-b58d-4202-a2b8-974c92e138db');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d5789e92-b6db-48fc-aab7-d2b54ecd6207">
      <button class="colab-df-quickchart" onclick="quickchart('df-d5789e92-b6db-48fc-aab7-d2b54ecd6207')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d5789e92-b6db-48fc-aab7-d2b54ecd6207 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_c6a6bb11-7270-49a6-a8c1-3db8a245807e">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_garch_ind_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_c6a6bb11-7270-49a6-a8c1-3db8a245807e button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_garch_ind_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



    VaR GARCH del Portafolio (USD)
    



  <div id="df-2854f9d8-58f5-4287-83e5-429a582aba00" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Portafolio USD</th>
      <td>5815.89441</td>
      <td>1690.823356</td>
      <td>6079.344778</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2854f9d8-58f5-4287-83e5-429a582aba00')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2854f9d8-58f5-4287-83e5-429a582aba00 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2854f9d8-58f5-4287-83e5-429a582aba00');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_ccdf5d43-0b12-4ab4-b219-db88490f1d51">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_var_garch_port_usd')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ccdf5d43-0b12-4ab4-b219-db88490f1d51 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_var_garch_port_usd');
      }
      })();
    </script>
  </div>

    </div>
  </div>



### CVaR


```python
alpha = 1 - confianza

def cvar_delta_normal(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    # Vol de portafolio con matriz de covarianza histórica
    cov = retornos.cov().values
    port_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    # Fórmula analítica de ES para Normal(0, σ) en cola inferior
    z_alpha = norm.ppf(1 - nivel)            # < 0 cuando nivel=0.95
    cvar_prop = (norm.pdf(z_alpha) / alpha) * port_vol * np.sqrt(horizonte)
    return cvar_prop, cvar_prop * monto_invertido

def cvar_historico(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    ret_port = retornos @ w
    log_T = np.log1p(ret_port).rolling(horizonte).sum().dropna()
    ret_T = np.expm1(log_T)

    var_thr = np.percentile(ret_T, (1 - nivel) * 100.0)
    tail = ret_T[ret_T <= var_thr]
    cvar_prop = -float(tail.mean())
    return cvar_prop, cvar_prop * monto_invertido

def cvar_monte_carlo(w, retornos, nivel=confianza, horizonte=horizonte, n_sim=n_simulaciones, monto_invertido=monto_invertido):
    # Estimar mu/sigma en log-retornos diarios y correlación
    logrets = np.log1p(retornos)
    mu = logrets.mean().values
    sig = logrets.std(ddof=1).values
    corr = np.corrcoef(logrets.T)

    # Corregir numéricamente si corr no es p.s.d.
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eps = 1e-6
        corr = (1 - eps) * corr + eps * np.eye(len(sig))
        L = np.linalg.cholesky(corr)

    # Simular log-retornos a T días (GBM)
    T = float(horizonte)
    Z = np.random.randn(n_sim, len(sig)) @ L.T
    logR = (mu - 0.5 * sig**2) * T + Z * sig * np.sqrt(T)
    R = np.exp(logR) - 1.0
    port_sim = R @ w

    var_thr = np.percentile(port_sim, (1 - nivel) * 100.0)
    tail = port_sim[port_sim <= var_thr]
    cvar_prop = -float(tail.mean())
    return cvar_prop, cvar_prop * monto_invertido

def cvar_ewma(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    # Volatilidades EWMA individuales y matriz de correlación histórica
    ewma_vol_ind = calcular_ewma_vol(retornos)               # vector σ_i (diario)
    corr = np.corrcoef(retornos.T)                           # correlación histórica
    ewma_cov = np.outer(ewma_vol_ind, ewma_vol_ind) * corr   # cov EWMA diaria
    port_vol = float(np.sqrt(np.dot(w, np.dot(ewma_cov, w))))
    z_alpha = norm.ppf(1 - nivel)
    cvar_prop = (norm.pdf(z_alpha) / alpha) * port_vol * np.sqrt(horizonte)
    return cvar_prop, cvar_prop * monto_invertido

def cvar_garch(w, retornos, nivel=confianza, horizonte=horizonte, monto_invertido=monto_invertido):
    # Vol GARCH individual (ya usas arch p=1,q=1) + correlación histórica
    vol_garch_ind = np.array([calcular_vol_garch(retornos[col]) for col in retornos.columns])  # σ_i diarios
    corr = retornos.corr().values
    cov_garch = np.outer(vol_garch_ind, vol_garch_ind) * corr
    port_vol = float(np.sqrt(np.dot(w, np.dot(cov_garch, w))))
    z_alpha = norm.ppf(1 - nivel)
    cvar_prop = (norm.pdf(z_alpha) / alpha) * port_vol * np.sqrt(horizonte)
    return cvar_prop, cvar_prop * monto_invertido

# Ejecutar para los 3 portafolios y 5 métodos
metodos_cvar = {
    'CVaR Delta-Normal': cvar_delta_normal,
    'CVaR Histórico':    cvar_historico,
    'CVaR Monte Carlo':  cvar_monte_carlo,
    'CVaR EWMA':         cvar_ewma,
    'CVaR GARCH':        cvar_garch,
}

# DataFrames resultado (en % y en USD)
df_cvar_port_pct = pd.DataFrame(index=metodos_cvar.keys(), columns=portafolios.keys(), dtype=float)
df_cvar_port_usd = pd.DataFrame(index=metodos_cvar.keys(), columns=portafolios.keys(), dtype=float)

for metodo, fun in metodos_cvar.items():
    for nombre_port, w in portafolios.items():
        cvar_prop, cvar_usd = fun(w, ret_activos)
        df_cvar_port_pct.loc[metodo, nombre_port] = cvar_prop * 100.0
        df_cvar_port_usd.loc[metodo, nombre_port] = cvar_usd

print("CVaR del Portafolio por método (%)")
display(df_cvar_port_pct.round(3))

print("CVaR del Portafolio por método (USD)")
display(df_cvar_port_usd.round(2))
```

    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    

    CVaR del Portafolio por método (%)
    

    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    /tmp/ipython-input-2891345651.py:6: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      sigma_t = resultado.conditional_volatility[-1] / 100  # Devuelve último valor y reescala
    



  <div id="df-b62c172b-0aef-475c-a561-4184764e04fa" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CVaR Delta-Normal</th>
      <td>6.560</td>
      <td>2.079</td>
      <td>7.024</td>
    </tr>
    <tr>
      <th>CVaR Histórico</th>
      <td>5.244</td>
      <td>1.783</td>
      <td>5.645</td>
    </tr>
    <tr>
      <th>CVaR Monte Carlo</th>
      <td>5.759</td>
      <td>1.826</td>
      <td>6.195</td>
    </tr>
    <tr>
      <th>CVaR EWMA</th>
      <td>6.080</td>
      <td>1.774</td>
      <td>6.384</td>
    </tr>
    <tr>
      <th>CVaR GARCH</th>
      <td>7.293</td>
      <td>2.120</td>
      <td>7.624</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b62c172b-0aef-475c-a561-4184764e04fa')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b62c172b-0aef-475c-a561-4184764e04fa button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b62c172b-0aef-475c-a561-4184764e04fa');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-419e8607-efa7-4428-888b-91f1fbeebba5">
      <button class="colab-df-quickchart" onclick="quickchart('df-419e8607-efa7-4428-888b-91f1fbeebba5')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-419e8607-efa7-4428-888b-91f1fbeebba5 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    CVaR del Portafolio por método (USD)
    



  <div id="df-ba8490d1-f910-4a2e-92f9-af6b30b72f78" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CVaR Delta-Normal</th>
      <td>6560.22</td>
      <td>2079.02</td>
      <td>7024.11</td>
    </tr>
    <tr>
      <th>CVaR Histórico</th>
      <td>5243.87</td>
      <td>1782.72</td>
      <td>5645.07</td>
    </tr>
    <tr>
      <th>CVaR Monte Carlo</th>
      <td>5758.99</td>
      <td>1825.81</td>
      <td>6195.30</td>
    </tr>
    <tr>
      <th>CVaR EWMA</th>
      <td>6080.46</td>
      <td>1774.35</td>
      <td>6384.29</td>
    </tr>
    <tr>
      <th>CVaR GARCH</th>
      <td>7293.37</td>
      <td>2120.36</td>
      <td>7623.74</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ba8490d1-f910-4a2e-92f9-af6b30b72f78')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ba8490d1-f910-4a2e-92f9-af6b30b72f78 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ba8490d1-f910-4a2e-92f9-af6b30b72f78');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-5b325761-faf6-4a0b-bf21-37c042c898d1">
      <button class="colab-df-quickchart" onclick="quickchart('df-5b325761-faf6-4a0b-bf21-37c042c898d1')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-5b325761-faf6-4a0b-bf21-37c042c898d1 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



### Cuadro Resumen


```python
# Consolidar todos los VaR y CVaR en porcentaje
resumen_pct = pd.concat([
    df_var_portafolio_pct.rename(index={'VaR Portafolio %': 'VaR Delta-Normal'}),
    df_var_hist_port_pct.rename(index={'VaR Portafolio %': 'VaR Histórico'}),
    df_var_mc_port_pct.rename(index={'VaR Portafolio %': 'VaR Monte Carlo'}),
    df_var_ewma_port_pct.rename(index={'VaR Portafolio %': 'VaR EWMA'}),
    df_var_garch_port_pct.rename(index={'VaR Portafolio %': 'VaR GARCH'}),
    df_cvar_port_pct.rename(index={
        'CVaR Delta-Normal': 'CVaR Delta-Normal',
        'CVaR Histórico': 'CVaR Histórico',
        'CVaR Monte Carlo': 'CVaR Monte Carlo',
        'CVaR EWMA': 'CVaR EWMA',
        'CVaR GARCH': 'CVaR GARCH'
    })
])

# Consolidar todos los VaR y CVaR en dólares
resumen_usd = pd.concat([
    df_var_portafolio_usd.rename(index={'VaR Portafolio USD': 'VaR Delta-Normal'}),
    df_var_hist_port_usd.rename(index={'VaR Portafolio USD': 'VaR Histórico'}),
    df_var_mc_port_usd.rename(index={'VaR Portafolio USD': 'VaR Monte Carlo'}),
    df_var_ewma_port_usd.rename(index={'VaR Portafolio USD': 'VaR EWMA'}),
    df_var_garch_port_usd.rename(index={'VaR Portafolio USD': 'VaR GARCH'}),
    df_cvar_port_usd.rename(index={
        'CVaR Delta-Normal': 'CVaR Delta-Normal',
        'CVaR Histórico': 'CVaR Histórico',
        'CVaR Monte Carlo': 'CVaR Monte Carlo',
        'CVaR EWMA': 'CVaR EWMA',
        'CVaR GARCH': 'CVaR GARCH'
    })
])

# Mostrar resumen final
print("Cuadro Resumen - VaR y CVaR del Portafolio (%)")
display(resumen_pct.round(2))

print("Cuadro Resumen - VaR y CVaR del Portafolio (USD)")
display(resumen_usd.round(2))
```

    Cuadro Resumen - VaR y CVaR del Portafolio (%)
    



  <div id="df-03770e1a-bf9b-4aba-9842-6d8f80268ec2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Delta-Normal</th>
      <td>5.23</td>
      <td>1.66</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>VaR Histórico</th>
      <td>3.98</td>
      <td>1.24</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>VaR Monte Carlo</th>
      <td>4.62</td>
      <td>1.48</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>VaR EWMA</th>
      <td>4.85</td>
      <td>1.41</td>
      <td>5.09</td>
    </tr>
    <tr>
      <th>VaR GARCH</th>
      <td>5.82</td>
      <td>1.69</td>
      <td>6.08</td>
    </tr>
    <tr>
      <th>CVaR Delta-Normal</th>
      <td>6.56</td>
      <td>2.08</td>
      <td>7.02</td>
    </tr>
    <tr>
      <th>CVaR Histórico</th>
      <td>5.24</td>
      <td>1.78</td>
      <td>5.65</td>
    </tr>
    <tr>
      <th>CVaR Monte Carlo</th>
      <td>5.76</td>
      <td>1.83</td>
      <td>6.20</td>
    </tr>
    <tr>
      <th>CVaR EWMA</th>
      <td>6.08</td>
      <td>1.77</td>
      <td>6.38</td>
    </tr>
    <tr>
      <th>CVaR GARCH</th>
      <td>7.29</td>
      <td>2.12</td>
      <td>7.62</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-03770e1a-bf9b-4aba-9842-6d8f80268ec2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-03770e1a-bf9b-4aba-9842-6d8f80268ec2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-03770e1a-bf9b-4aba-9842-6d8f80268ec2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d5fbe3d2-e34d-41d1-a32a-b87c11ca376b">
      <button class="colab-df-quickchart" onclick="quickchart('df-d5fbe3d2-e34d-41d1-a32a-b87c11ca376b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d5fbe3d2-e34d-41d1-a32a-b87c11ca376b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    Cuadro Resumen - VaR y CVaR del Portafolio (USD)
    



  <div id="df-2c3b51eb-0ec2-4cb3-aec6-a8dd1cb5af14" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sharpe</th>
      <th>Min Varianza</th>
      <th>Sortino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>VaR Delta-Normal</th>
      <td>5231.27</td>
      <td>1657.85</td>
      <td>5601.18</td>
    </tr>
    <tr>
      <th>VaR Histórico</th>
      <td>3982.36</td>
      <td>1235.56</td>
      <td>4263.39</td>
    </tr>
    <tr>
      <th>VaR Monte Carlo</th>
      <td>4621.42</td>
      <td>1475.37</td>
      <td>4986.98</td>
    </tr>
    <tr>
      <th>VaR EWMA</th>
      <td>4848.70</td>
      <td>1414.91</td>
      <td>5090.97</td>
    </tr>
    <tr>
      <th>VaR GARCH</th>
      <td>5815.89</td>
      <td>1690.82</td>
      <td>6079.34</td>
    </tr>
    <tr>
      <th>CVaR Delta-Normal</th>
      <td>6560.22</td>
      <td>2079.02</td>
      <td>7024.11</td>
    </tr>
    <tr>
      <th>CVaR Histórico</th>
      <td>5243.87</td>
      <td>1782.72</td>
      <td>5645.07</td>
    </tr>
    <tr>
      <th>CVaR Monte Carlo</th>
      <td>5758.99</td>
      <td>1825.81</td>
      <td>6195.30</td>
    </tr>
    <tr>
      <th>CVaR EWMA</th>
      <td>6080.46</td>
      <td>1774.35</td>
      <td>6384.29</td>
    </tr>
    <tr>
      <th>CVaR GARCH</th>
      <td>7293.37</td>
      <td>2120.36</td>
      <td>7623.74</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2c3b51eb-0ec2-4cb3-aec6-a8dd1cb5af14')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2c3b51eb-0ec2-4cb3-aec6-a8dd1cb5af14 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2c3b51eb-0ec2-4cb3-aec6-a8dd1cb5af14');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-ec36ed68-194b-4b69-af51-0c466ca1a54c">
      <button class="colab-df-quickchart" onclick="quickchart('df-ec36ed68-194b-4b69-af51-0c466ca1a54c')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-ec36ed68-194b-4b69-af51-0c466ca1a54c button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



## 5. Validación Regulatoria

### Validación


```python
# --- LÍMITES REGULATORIOS ---
limite_var_pct           = 6.5   # VaR Portafolio %   (95%, 10 días típico)
limite_cvar_pct          = 9.1   # CVaR Portafolio %  (≈1.4x VaR)
aporte_max_riesgo_pct    = 65.0  # Máx contribución individual para activos de riesgo
aporte_max_defensivo_pct = 10.0  # Máx contribución individual para activos defensivos (p.ej., SHY)

# --- CLASIFICACIÓN DE ACTIVOS ---
# Marca los activos defensivos; el resto se consideran "riesgo".
activos_defensivos = mejor_bono
activos_riesgo = [a for a in activos if a not in activos_defensivos]

# --- Mapeo de métodos a sus DataFrames de VaR (portafolio e individual) ---
mapa_metodos = {
    'Delta-Normal': {'var_port_df': df_var_portafolio_pct, 'var_ind_df': df_var_individual_pct},
    'Histórico':    {'var_port_df': df_var_hist_port_pct,   'var_ind_df': df_var_hist_ind_pct},
    'Monte Carlo':  {'var_port_df': df_var_mc_port_pct,     'var_ind_df': df_var_mc_ind_pct},
    'EWMA':         {'var_port_df': df_var_ewma_port_pct,   'var_ind_df': df_var_ewma_ind_pct},
    'GARCH':        {'var_port_df': df_var_garch_port_pct,  'var_ind_df': df_var_garch_ind_pct},
}

# --- Helper para tomar el CVaR por método desde df_cvar_port_pct ---
def obtener_cvar_port(df_cvar_port_pct, metodo, nombre_port):
    etiqueta = f'CVaR {metodo}'
    if etiqueta in df_cvar_port_pct.index:
        return float(df_cvar_port_pct.loc[etiqueta, nombre_port])
    # compatibilidad por si quedó sólo "CVaR Portafolio %"
    if 'CVaR Portafolio %' in df_cvar_port_pct.index:
        return float(df_cvar_port_pct.loc['CVaR Portafolio %', nombre_port])
    raise KeyError(f"No se encontró CVaR para el método '{metodo}' en df_cvar_port_pct")

# --- Construcción de tabla de validación ---
filas = []
for metodo, dfs in mapa_metodos.items():
    var_port_df = dfs['var_port_df']
    var_ind_df  = dfs['var_ind_df']

    for nombre_port in portafolios.keys():
        # VaR portafolio %
        try:
            var_port_pct = float(var_port_df.loc['VaR Portafolio %', nombre_port])
        except KeyError:
            raise KeyError(f"En {metodo}, falta 'VaR Portafolio %' o la columna '{nombre_port}'")

        # CVaR portafolio %
        cvar_port_pct = obtener_cvar_port(df_cvar_port_pct, metodo, nombre_port)

        # Contribuciones individuales relativas al VaR del portafolio
        if var_port_pct == 0 or np.isnan(var_port_pct):
            # Evita división por cero
            aporte_max_riesgo = np.nan
            aporte_max_def    = np.nan
            cumple_aporte     = False
            detalle_brecha    = "VaR portafolio = 0"
        else:
            serie_var_ind_pct = var_ind_df[nombre_port].reindex(activos)  # asegura orden y presencia
            contrib_pct = (serie_var_ind_pct / var_port_pct) * 100.0

            # Máximo por grupo
            aporte_max_riesgo = float(contrib_pct.loc[activos_riesgo].max()) if len(activos_riesgo) else np.nan
            aporte_max_def    = float(contrib_pct.loc[list(activos_defensivos)].max()) if len(activos_defensivos) else np.nan

            # Chequeo por-asset con topes duales
            mask_riesgo_ok = contrib_pct.loc[activos_riesgo]    <= aporte_max_riesgo_pct if len(activos_riesgo) else pd.Series(dtype=bool)
            mask_def_ok    = contrib_pct.loc[list(activos_defensivos)] <= aporte_max_defensivo_pct if len(activos_defensivos) else pd.Series(dtype=bool)

            # Si hay algún activo fuera de límite
            fuera_riesgo = contrib_pct.loc[activos_riesgo][~mask_riesgo_ok] if len(activos_riesgo) else pd.Series(dtype=float)
            fuera_def    = contrib_pct.loc[list(activos_defensivos)][~mask_def_ok] if len(activos_defensivos) else pd.Series(dtype=float)

            cumple_aporte = (fuera_riesgo.empty and fuera_def.empty)
            detalle_brecha = ""
            if not cumple_aporte:
                partes = []
                if not fuera_riesgo.empty:
                    partes.append(f"Riesgo>{aporte_max_riesgo_pct}%: {', '.join([f'{k}={v:.1f}%' for k, v in fuera_riesgo.items()])}")
                if not fuera_def.empty:
                    partes.append(f"Defensivo>{aporte_max_defensivo_pct}%: {', '.join([f'{k}={v:.1f}%' for k, v in fuera_def.items()])}")
                detalle_brecha = " | ".join(partes)

        # Chequeos VaR/CVaR
        chk_var  = (var_port_pct  <= limite_var_pct)
        chk_cvar = (cvar_port_pct <= limite_cvar_pct)

        filas.append({
            'Portafolio': nombre_port,
            'Método': metodo,
            'VaR Portafolio %': var_port_pct,
            'CVaR Portafolio %': cvar_port_pct,
            'Aporte Máx Riesgo %': aporte_max_riesgo,
            'Aporte Máx Defensivo %': aporte_max_def,
            'Cumple VaR': chk_var,
            'Cumple CVaR': chk_cvar,
            'Cumple Aporte Dual': cumple_aporte,
            'Detalle Aporte (brechas)': detalle_brecha,
            'Cumple TODO': (chk_var and chk_cvar and cumple_aporte),
        })

df_validacion = pd.DataFrame(filas)

# Sólo los modelos que cumplen todo
df_regulatorio_ok = df_validacion[df_validacion['Cumple TODO']].reset_index(drop=True)

print("Validación regulatoria — detalle por modelo (tope dual)")
display(df_validacion.sort_values(['Portafolio','Método']).reset_index(drop=True).round(3))

print("Modelos que PASAN la regulación (tope dual)")
display(df_regulatorio_ok.sort_values(['Portafolio','Método']).round(3))
```

    Validación regulatoria — detalle por modelo (tope dual)
    



  <div id="df-41abd43d-81e1-43e7-899e-ac75adaa4e6d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Portafolio</th>
      <th>Método</th>
      <th>VaR Portafolio %</th>
      <th>CVaR Portafolio %</th>
      <th>Aporte Máx Riesgo %</th>
      <th>Aporte Máx Defensivo %</th>
      <th>Cumple VaR</th>
      <th>Cumple CVaR</th>
      <th>Cumple Aporte Dual</th>
      <th>Detalle Aporte (brechas)</th>
      <th>Cumple TODO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Min Varianza</td>
      <td>Delta-Normal</td>
      <td>1.658</td>
      <td>2.079</td>
      <td>94.964</td>
      <td>29.229</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: LRCX=95.0% | Defensivo&gt;10.0%: SH...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Min Varianza</td>
      <td>EWMA</td>
      <td>1.415</td>
      <td>1.774</td>
      <td>93.349</td>
      <td>32.742</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: LRCX=93.3% | Defensivo&gt;10.0%: SH...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Min Varianza</td>
      <td>GARCH</td>
      <td>1.691</td>
      <td>2.120</td>
      <td>95.961</td>
      <td>25.399</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: LRCX=96.0% | Defensivo&gt;10.0%: SH...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Min Varianza</td>
      <td>Histórico</td>
      <td>1.236</td>
      <td>1.783</td>
      <td>101.901</td>
      <td>32.229</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: LRCX=101.9% | Defensivo&gt;10.0%: S...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Min Varianza</td>
      <td>Monte Carlo</td>
      <td>1.475</td>
      <td>1.826</td>
      <td>96.812</td>
      <td>29.657</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: LRCX=96.8% | Defensivo&gt;10.0%: SH...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sharpe</td>
      <td>Delta-Normal</td>
      <td>5.231</td>
      <td>6.560</td>
      <td>71.158</td>
      <td>6.294</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=71.2%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sharpe</td>
      <td>EWMA</td>
      <td>4.849</td>
      <td>6.080</td>
      <td>74.029</td>
      <td>6.492</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=74.0%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sharpe</td>
      <td>GARCH</td>
      <td>5.816</td>
      <td>7.293</td>
      <td>73.516</td>
      <td>5.017</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=73.5%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sharpe</td>
      <td>Histórico</td>
      <td>3.982</td>
      <td>5.244</td>
      <td>68.580</td>
      <td>6.794</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=68.6%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sharpe</td>
      <td>Monte Carlo</td>
      <td>4.621</td>
      <td>5.759</td>
      <td>71.241</td>
      <td>6.404</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=71.2%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sortino</td>
      <td>Delta-Normal</td>
      <td>5.601</td>
      <td>7.024</td>
      <td>66.459</td>
      <td>5.094</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=66.5%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sortino</td>
      <td>EWMA</td>
      <td>5.091</td>
      <td>6.384</td>
      <td>70.506</td>
      <td>5.358</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=70.5%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sortino</td>
      <td>GARCH</td>
      <td>6.079</td>
      <td>7.624</td>
      <td>70.330</td>
      <td>4.159</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=70.3%</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sortino</td>
      <td>Histórico</td>
      <td>4.263</td>
      <td>5.645</td>
      <td>64.059</td>
      <td>5.499</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td></td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sortino</td>
      <td>Monte Carlo</td>
      <td>4.987</td>
      <td>6.195</td>
      <td>66.208</td>
      <td>5.081</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>Riesgo&gt;65.0%: KLAC=66.2%</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-41abd43d-81e1-43e7-899e-ac75adaa4e6d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-41abd43d-81e1-43e7-899e-ac75adaa4e6d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-41abd43d-81e1-43e7-899e-ac75adaa4e6d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-24170e13-845f-469c-811e-62d9c850f5c9">
      <button class="colab-df-quickchart" onclick="quickchart('df-24170e13-845f-469c-811e-62d9c850f5c9')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-24170e13-845f-469c-811e-62d9c850f5c9 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    Modelos que PASAN la regulación (tope dual)
    



  <div id="df-7f1401c0-58c2-4024-9796-9851482625af" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Portafolio</th>
      <th>Método</th>
      <th>VaR Portafolio %</th>
      <th>CVaR Portafolio %</th>
      <th>Aporte Máx Riesgo %</th>
      <th>Aporte Máx Defensivo %</th>
      <th>Cumple VaR</th>
      <th>Cumple CVaR</th>
      <th>Cumple Aporte Dual</th>
      <th>Detalle Aporte (brechas)</th>
      <th>Cumple TODO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sortino</td>
      <td>Histórico</td>
      <td>4.263</td>
      <td>5.645</td>
      <td>64.059</td>
      <td>5.499</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td></td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7f1401c0-58c2-4024-9796-9851482625af')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7f1401c0-58c2-4024-9796-9851482625af button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7f1401c0-58c2-4024-9796-9851482625af');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>



### Gráfica de Resultado


```python
def graficar_distribucion_filtrada(nombre_port, pesos, retornos,
                                   df_reg_ok,
                                   dfs_var_port,         # dict método -> df_var_<método>_port_pct
                                   df_cvar_port_pct=None,# df_cvar_port_pct con filas 'CVaR <método>'
                                   dibujar_cvar=True):
    # Métodos aprobados para este portafolio
    met_aprob = df_reg_ok.loc[df_reg_ok['Portafolio'] == nombre_port, 'Método'].unique()
    if len(met_aprob) == 0:
        print(f"No hay métodos aprobados para el portafolio {nombre_port}.")
        return

    # Colores por método (consistentes con tu bloque original)
    colores_metodo = {
        'Delta-Normal': 'red',
        'Histórico': 'blue',
        'Monte Carlo': 'green',
        'EWMA': 'orange',
        'GARCH': 'purple',
    }

    # Retornos diarios observados del portafolio
    ret_port = retornos @ pesos

    # Plot distribución
    plt.figure(figsize=(10, 6))
    sns.histplot(ret_port, bins=50, kde=True, color='skyblue', stat='density')
    plt.title(f"Distribución de Retornos Diarios — Portafolio {nombre_port}")
    plt.xlabel("Retorno diario")
    plt.ylabel("Densidad")

    # Líneas de VaR (y CVaR) solo para métodos aprobados
    for metodo in met_aprob:
        # 1) VaR del portafolio para este método
        df_var_port = dfs_var_port.get(metodo, None)
        if df_var_port is not None and (nombre_port in df_var_port.columns):
            # Usa .iloc[0] para acceder al primer (y único) valor de VaR Portafolio %
            try:
                var_pct = float(df_var_port.loc['VaR Portafolio %', nombre_port])
                var_val = -var_pct / 100.0
                plt.axvline(var_val, color=colores_metodo.get(metodo, 'red'),
                            linestyle='--', linewidth=1.8, label=f"VaR {metodo}")
            except Exception:
                pass

        # 2) (Opcional) CVaR del portafolio para este método
        if dibujar_cvar and (df_cvar_port_pct is not None) and (nombre_port in df_cvar_port_pct.columns):
            fila_cvar = f'CVaR {metodo}'
            if fila_cvar in df_cvar_port_pct.index:
                try:
                    cvar_pct = float(df_cvar_port_pct.loc[fila_cvar, nombre_port])
                    cvar_val = -cvar_pct / 100.0
                    plt.axvline(cvar_val, color=colores_metodo.get(metodo, 'red'),
                                linestyle=':', linewidth=1.8, label=f"CVaR {metodo}")
                except Exception:
                    pass

    plt.axvline(0, color='black', linestyle=':')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Mapeo MÉTODO -> DF de VaR (portafolio %) que ya calculaste ---
dfs_var_port = {
    'Delta-Normal': df_var_portafolio_pct,   # índice: 'VaR Portafolio %'
    'Histórico':    df_var_hist_port_pct,
    'Monte Carlo':  df_var_mc_port_pct,
    'EWMA':         df_var_ewma_port_pct,
    'GARCH':        df_var_garch_port_pct,
}

# Dibujar SOLO para portafolios con al menos un método aprobado
# Asegúrate de que df_regulatorio_ok existe y no está vacío
if 'df_regulatorio_ok' not in globals() or df_regulatorio_ok.empty:
    print("No se puede generar la gráfica resumen. No hay modelos que pasen la validación regulatoria.")
else:
    ports_con_aprobados = df_regulatorio_ok['Portafolio'].unique()
    for nombre_port in ports_con_aprobados:
        pesos = portafolios[nombre_port]
        graficar_distribucion_filtrada(
            nombre_port=nombre_port,
            pesos=pesos,
            retornos=ret_activos,
            df_reg_ok=df_regulatorio_ok,
            dfs_var_port=dfs_var_port,
            df_cvar_port_pct=df_cvar_port_pct,  # si no quieres CVaR, pásalo como None o dibujar_cvar=False
            dibujar_cvar=True
        )
```


    
![png](output_59_0.png)
    


## 6. Backtesting


```python
from scipy.stats import chi2

# ---- utilidades de backtest ----
def generar_violaciones(retornos_portafolio, var_series_pos):
    return (retornos_portafolio < -var_series_pos).astype(int)

def test_kupiec(violaciones, alpha):
    n = len(violaciones)
    x = int(violaciones.sum())
    if n == 0:
        return np.nan
    pi = x / n
    pi0 = 1 - alpha
    # manejar casos extremos para evitar log(0)
    if pi == 0 or pi == 1:
        return np.nan
    LR_pof = -2 * (np.log((1 - pi0)**(n - x) * pi0**x) -
                   np.log((1 - pi)**(n - x) * pi**x))
    return 1 - chi2.cdf(LR_pof, df=1)

def test_christoffersen(violaciones):
    v = np.asarray(violaciones, dtype=int)
    if len(v) < 2:
        return np.nan
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(v)):
        prev, curr = v[t - 1], v[t]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1
    # probabilidades de transición (con guardas)
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n00 + n01 + n10 + n11) == 0:
        return np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)
    # evitar log(0)
    eps = 1e-12
    logL_uncond = n00*np.log(max(1-pi,eps)) + (n01+n10+n11)*np.log(max(pi,eps))
    logL_cond   = (
        n00*np.log(max(1-pi01,eps)) + n01*np.log(max(pi01,eps)) +
        n10*np.log(max(1-pi11,eps)) + n11*np.log(max(pi11,eps))
    )
    LR_indep = -2 * (logL_uncond - logL_cond)
    return 1 - chi2.cdf(LR_indep, df=1)

# ---- funciones de VaR rolling por método (horizonte = 1 día para backtesting) ----
def var_rolling_historico(ret_activos, w, alpha, window=250):
    ret_port = ret_activos @ w
    vals = []
    for i in range(window, len(ret_port)):
        roll = ret_port.iloc[i-window:i]
        var_i = -np.percentile(roll, (1 - alpha) * 100)  # magnitud positiva
        vals.append(var_i)
    return ret_port.iloc[window:].values, np.array(vals)

def var_rolling_delta_normal(ret_activos, w, alpha, window=250):
    z = abs(norm.ppf(1 - alpha))
    ret_port = ret_activos @ w
    vals = []
    for i in range(window, len(ret_activos)):
        sub = ret_activos.iloc[i-window:i]
        cov = sub.cov().values
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        var_i = z * port_vol  # magnitud positiva (1 día)
        vals.append(var_i)
    return ret_port.iloc[window:].values, np.array(vals)

def var_rolling_montecarlo(ret_activos, w, alpha, window=250, n_sim=20000):
    ret_port = ret_activos @ w
    vals = []

    for i in range(window, len(ret_activos)):
        sub = ret_activos.iloc[i-window:i]
        logrets = np.log1p(sub)

        mu   = logrets.mean().values
        sig  = logrets.std(ddof=1).values
        corr = np.corrcoef(logrets.T)

        # Guardas por estabilidad
        sig = np.where(sig <= 1e-8, 1e-8, sig)
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # pequeña regularización a la correlación si no es p.s.d.
            eps = 1e-6
            corr = (1 - eps) * corr + eps * np.eye(len(sig))
            L = np.linalg.cholesky(corr)

        # Simulaciones 1D
        Z = np.random.randn(n_sim, len(sig)) @ L.T
        logR = (mu - 0.5 * sig**2) * 1.0 + Z * sig * 1.0
        R = np.exp(logR) - 1.0
        port = R @ w

        var_i = -np.percentile(port, (1 - alpha) * 100.0)  # magnitud positiva
        vals.append(var_i)

    return ret_port.iloc[window:].values, np.array(vals)

def var_rolling_ewma(ret_activos, w, alpha, window=250, lam=lambda_):
    z = abs(norm.ppf(1 - alpha))
    ret_port = ret_activos @ w
    vals = []
    for i in range(window, len(ret_activos)):
        sub = ret_activos.iloc[i-window:i]
        # vol EWMA individual (último punto)
        vols = []
        for col in sub.columns:
            serie = sub[col].values
            var_t = np.var(serie[:5]) if len(serie)>=5 else np.var(serie)  # semilla
            for r in serie[1:]:
                var_t = lam*var_t + (1-lam)*(r**2)
            vols.append(np.sqrt(var_t))
        vols = np.array(vols)
        corr = np.corrcoef(sub.T)
        ewma_cov = np.outer(vols, vols) * corr
        port_vol = np.sqrt(np.dot(w, np.dot(ewma_cov, w)))
        vals.append(z * port_vol)
    return ret_port.iloc[window:].values, np.array(vals)

def var_rolling_garch(ret_activos, w, alpha, window=750):
    from arch import arch_model
    import warnings
    z = abs(norm.ppf(1 - alpha))
    ret_port = ret_activos @ w
    vals = []
    for i in range(window, len(ret_activos)):
        sub = ret_activos.iloc[i-window:i]
        sigmas = []
        for col in sub.columns:
            serie = (sub[col].values * 100.0)  # en %
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    am = arch_model(serie, vol='GARCH', p=1, q=1, rescale=False)
                    res = am.fit(disp='off', show_warning=False)
                # pronóstico 1-step ahead de la varianza (en %^2)
                fcast = res.forecast(horizon=1, reindex=False)
                var_next = float(fcast.variance.values[-1, 0])      # %^2
                sigma_next = np.sqrt(var_next) / 100.0              # a proporción
                # sanity fallback
                if not np.isfinite(sigma_next) or sigma_next <= 0:
                    raise ValueError("sigma_next inválida")
            except Exception:
                sigma_next = float(np.std(serie, ddof=1) / 100.0)   # fallback: vol histórica
            sigmas.append(sigma_next)
        sigmas = np.array(sigmas)
        corr = np.corrcoef(sub.T)
        cov_garch = np.outer(sigmas, sigmas) * corr
        port_vol = np.sqrt(np.dot(w, np.dot(cov_garch, w)))
        vals.append(z * port_vol)
    return ret_port.iloc[window:].values, np.array(vals)

# Mapeo de método -> función rolling
rolling_funcs = {
    'Histórico':    var_rolling_historico,
    'Delta-Normal': var_rolling_delta_normal,
    'Monte Carlo':  var_rolling_montecarlo,
    'EWMA':         var_rolling_ewma,
    'GARCH':        var_rolling_garch,
}

# -------------------------------
# Ejecutar SOLO para df_regulatorio_ok
# -------------------------------
window_hist = 250     # ~1 año
window_dn   = 250
window_mc   = 250
window_ewma = 250
window_gch  = 250     # GARCH requiere más datos

ventanas = {
    'Histórico':    window_hist,
    'Delta-Normal': window_dn,
    'Monte Carlo':  window_mc,
    'EWMA':         window_ewma,
    'GARCH':        window_gch,
}

alpha_bt = confianza  # usar el mismo nivel que definiste

resultados_bt = []
errores_bt = []

# Asegura que tenemos df_regulatorio_ok construido (de tu validación)
if 'df_regulatorio_ok' not in globals():
    raise RuntimeError("df_regulatorio_ok no existe. Ejecuta primero la validación regulatoria.")

for _, fila in df_regulatorio_ok.iterrows():
    nombre_port = fila['Portafolio']
    metodo      = fila['Método']
    w           = portafolios[nombre_port]
    fun_roll    = rolling_funcs.get(metodo, None)
    if fun_roll is None:
        continue

    try:
        win = ventanas[metodo]
        # serie de retornos observados y serie de VaR pronosticado (magnitud positiva)
        ret_eval, var_series_pos = fun_roll(ret_activos, w, alpha_bt, window=win)
        # violaciones
        viol = generar_violaciones(ret_eval, var_series_pos)
        # tests
        p_k = test_kupiec(viol, alpha_bt)
        p_c = test_christoffersen(viol)

        resultados_bt.append({
            'Portafolio': nombre_port,
            'Método': metodo,
            'Observaciones': len(viol),
            'Violaciones': int(viol.sum()),
            'Tasa Observada': float(viol.mean()),
            'p-valor Kupiec': p_k,
            'p-valor Christoffersen': p_c
        })
    except Exception as e:
        errores_bt.append((nombre_port, metodo, str(e)))
        continue

df_backtest_ok = pd.DataFrame(resultados_bt).sort_values(['Portafolio','Método']).reset_index(drop=True)

print("Backtesting (solo modelos que pasaron regulación)")
display(df_backtest_ok)

if errores_bt:
    print("Modelos con error durante el backtesting:")
    for (p,m,err) in errores_bt:
        print(f" - {p} / {m}: {err}")
```

    Backtesting (solo modelos que pasaron regulación)
    



  <div id="df-abffeb82-d556-42f2-a6fb-71922da5202c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Portafolio</th>
      <th>Método</th>
      <th>Observaciones</th>
      <th>Violaciones</th>
      <th>Tasa Observada</th>
      <th>p-valor Kupiec</th>
      <th>p-valor Christoffersen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sortino</td>
      <td>Histórico</td>
      <td>1215</td>
      <td>70</td>
      <td>0.057613</td>
      <td>0.234018</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-abffeb82-d556-42f2-a6fb-71922da5202c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-abffeb82-d556-42f2-a6fb-71922da5202c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-abffeb82-d556-42f2-a6fb-71922da5202c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


  <div id="id_cff8b490-d0f3-4ff2-ace1-6a992c41d1c3">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_backtest_ok')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_cff8b490-d0f3-4ff2-ace1-6a992c41d1c3 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_backtest_ok');
      }
      })();
    </script>
  </div>

    </div>
  </div>



## 7. Stress Testing


```python
# Unir resultados de Kupiec y validación regulatoria
# Asegúrate de que df_backtest_ok existe y no está vacío
if 'df_backtest_ok' not in globals() or df_backtest_ok.empty:
    print("No se puede realizar Stress Testing. No hay modelos que pasen la validación regulatoria o el backtesting falló.")
else:
    df_kupiec_validado = df_backtest_ok.merge(
        df_regulatorio_ok[['Portafolio', 'Método']],
        on=['Portafolio', 'Método'],
        how='inner'
    )

    # Seleccionar el modelo con mayor p-valor
    if df_kupiec_validado.empty:
        raise ValueError("No hay modelos que pasen la validación regulatoria para stress testing.")

    mejor_modelo = df_kupiec_validado.sort_values('p-valor Kupiec', ascending=False).iloc[0]
    port_sel = mejor_modelo['Portafolio']
    metodo_sel = mejor_modelo['Método']

    print(f"Modelo seleccionado para Stress Testing: {port_sel} - {metodo_sel}")

    # Obtener el VaR del modelo seleccionado
    var_seleccionado_pct = resumen_pct.loc[f"VaR {metodo_sel}", port_sel]

    # Escenarios reales de stress testing
    escenarios = [
        {'nombre': 'COVID-19 Marzo 2020',   'ini': '2020-03-01', 'fin': '2020-03-31'},
        {'nombre': 'Invasión Ucrania 2022', 'ini': '2022-02-20', 'fin': '2022-03-10'},
        {'nombre': 'Caída Tech 2022',       'ini': '2022-12-01', 'fin': '2023-01-15'}
    ]

    # Calcular pérdidas y comparar con VaR
    resultados_stress = []
    # Asegúrate de usar ret_activos para el portafolio seleccionado
    ret_port_sel = ret_activos @ portafolios[port_sel]

    for esc in escenarios:
        fecha_ini, fecha_fin = esc['ini'], esc['fin']
        # Filtrar por rango de fechas
        ret_periodo = ret_port_sel.loc[fecha_ini:fecha_fin]

        if len(ret_periodo) == 0:
            print(f"No hay datos para {esc['nombre']} ({fecha_ini} a {fecha_fin}) en la serie.")
            continue

        # Calcular el retorno acumulado en el período
        perdida_pct = (ret_periodo + 1).prod() - 1
        perdida_pct *= 100  # a porcentaje
        perdida_usd = (perdida_pct / 100) * monto_invertido # a USD

        # Comparar con VaR (VaR es pérdida esperada, por eso usamos abs o lo comparamos con el negativo de la pérdida)
        # El VaR seleccionado ya está en magnitud positiva (%)
        supera_var = perdida_pct < -var_seleccionado_pct
        exceso_pp = abs(perdida_pct) - var_seleccionado_pct if supera_var else 0
        exceso_veces = (abs(perdida_pct) / var_seleccionado_pct) if supera_var and var_seleccionado_pct != 0 else 0 # Evitar división por cero

        resultados_stress.append({
            'Escenario': esc['nombre'],
            'Fecha Inicio': fecha_ini,
            'Fecha Fin': fecha_fin,
            'Pérdida %': perdida_pct,
            'Pérdida USD': perdida_usd,
            'VaR % (10D)': var_seleccionado_pct,
            'Supera VaR?': supera_var,
            'Exceso P.P.': exceso_pp,
            'Exceso Veces': exceso_veces
        })

    # Mostrar tabla final
    if resultados_stress:
        df_stress = pd.DataFrame(resultados_stress)
        print("Resultados de Stress Testing con eventos reales y comparación con VaR")
        display(df_stress.round(3))
    else:
        print("No se pudieron calcular resultados de Stress Testing para los escenarios definidos.")
```

    Modelo seleccionado para Stress Testing: Sortino - Histórico
    Resultados de Stress Testing con eventos reales y comparación con VaR
    



  <div id="df-56e49dfd-590b-48ca-9079-33fea0009469" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Escenario</th>
      <th>Fecha Inicio</th>
      <th>Fecha Fin</th>
      <th>Pérdida %</th>
      <th>Pérdida USD</th>
      <th>VaR % (10D)</th>
      <th>Supera VaR?</th>
      <th>Exceso P.P.</th>
      <th>Exceso Veces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COVID-19 Marzo 2020</td>
      <td>2020-03-01</td>
      <td>2020-03-31</td>
      <td>-1.929</td>
      <td>-1928.769</td>
      <td>4.263</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Invasión Ucrania 2022</td>
      <td>2022-02-20</td>
      <td>2022-03-10</td>
      <td>-4.058</td>
      <td>-4058.259</td>
      <td>4.263</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Caída Tech 2022</td>
      <td>2022-12-01</td>
      <td>2023-01-15</td>
      <td>1.980</td>
      <td>1980.301</td>
      <td>4.263</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-56e49dfd-590b-48ca-9079-33fea0009469')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-56e49dfd-590b-48ca-9079-33fea0009469 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-56e49dfd-590b-48ca-9079-33fea0009469');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f9c602a2-348b-43ec-ad9f-97c00440c84f">
      <button class="colab-df-quickchart" onclick="quickchart('df-f9c602a2-348b-43ec-ad9f-97c00440c84f')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f9c602a2-348b-43ec-ad9f-97c00440c84f button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>

