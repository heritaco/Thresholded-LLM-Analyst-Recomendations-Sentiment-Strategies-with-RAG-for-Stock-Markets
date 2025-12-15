import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from IPython.display import display, HTML

pd.set_option('display.max_rows', None)

def extraer_noticias_finviz(tick="TSLA") -> pd.DataFrame:
    """
    Extrae las noticias de un ticker de Finviz.
    Args:
        tick (str): Ticker de la acción a consultar.
    Returns:
        pd.DataFrame: DataFrame con las noticias, con fecha, título y URL.
    """

    url = f"https://finviz.com/quote.ashx?t={tick}&p=d"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error al obtener la página: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")

    noticias = []

    if news_table:
        rows = news_table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            fecha = cols[0].text.strip()
            link_tag = cols[1].find("a")
            titulo = link_tag.text.strip() if link_tag else ""
            url = link_tag['href'] if link_tag and 'href' in link_tag.attrs else ""

            noticias.append({"fecha": fecha, "título": titulo, "url": url})

    df = pd.DataFrame(noticias)

    # Para los links que empiezan con '/news/', agregar el dominio finviz.com
    df['url'] = np.where(
        df['url'].str.startswith('/news/'),
        'https://finviz.com' + df['url'],
        df['url']
    )

    return df









def nombre_bonito(url):
    # Pone bonito el dominio de la URL, es para saber cuantas veces se repite un dominio en el DataFrame de noticias
    if "https://www." in url:
        url = url.replace("https://www.", "")
    elif "https://finance." in url:
        url = url.replace("https://finance.", "")
    elif "https://finviz.com/" in url:
        url = "finviz"
    elif "https://chainstoreage." in url:
        url = url.replace("https://chainstoreage.", "chainstoreage.")

    elif "https://seekingalpha." in url:
        url = url.replace("https://seekingalpha.", "seekingalpha.")

    elif "https://qz." in url:
        url = url.replace("https://qz.", "qz.")

    return url.split(".")[0]








def hiperreferenciar_titulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega hiperenlaces a los títulos de las noticias en el DataFrame.
    Args:
        df (pd.DataFrame): DataFrame con las noticias.
    Returns:
        pd.DataFrame: DataFrame con los títulos convertidos en hiperenlaces.
    """

    if 'url' not in df.columns or 'título' not in df.columns:
        raise ValueError(
            "El DataFrame debe contener las columnas 'url' y 'título'.")

    df['título'] = df.apply(
        lambda row: f'<a href="{row["url"]}" target="_blank">{row["título"]}</a>' if row["url"] else row["título"],
        axis=1
    )

    return df







# Metodo
def web_scrap(tickers: list) -> dict:
    """
    Obtiene los datos de noticias para una lista de tickers de acciones.
    Utiliza la función `extraer_noticias_finviz` para cada ticker y maneja errores de conexión.
    Esta función espera 10 segundos entre cada solicitud para evitar el error 429 (Too Many Requests).
    Args:
        tickers (list): Lista de tickers de acciones a consultar.
    Returns:
        dict: Diccionario donde las claves son los tickers y los valores son DataFrames con las noticias.
    Raises:
        Exception: Si ocurre un error al extraer las noticias para un ticker, se espera 10 segundos y se reintenta.
    """

    dfs = {}
    for tick in tickers:
        try:
            dfs[tick] = extraer_noticias_finviz(tick)
            print(f"{tick} ya!")
        except Exception as e:
            print(f"Tenemos que esperar un poco más para {tick}: {e}")
            time.sleep(10) 
            dfs[tick] = extraer_noticias_finviz(tick)
            print(f"{tick} ya!")

    return dfs





# Metodo
def cuantas_veces(dfs: dict, tick: str, index = False) -> pd.DataFrame:
    """
    Cuenta cuántas veces aparece cada journal en las noticias de un ticker y lo muestra como DataFrame.
    Args:
        dfs (dict): Diccionario de DataFrames de noticias por ticker.
        tick (str): Ticker de la acción a consultar.
    index (bool): Si es True, muestra el índice del DataFrame. Por defecto es False.
    Returns:
        pd.DataFrame: DataFrame con el conteo de cada journal.
    """

    df = dfs[tick]
    conteo = df["url"].apply(nombre_bonito).value_counts().reset_index()
    conteo.columns = ['journal', 'veces']

    display(HTML(f'<h2 style="text-align:left;">Últimas 100 noticias de {tick}</h2>'))

    if index:
        display(conteo.style.set_properties(subset=['journal'], **{'text-align': 'left'})
               .set_properties(subset=['veces'], **{'text-align': 'right'})
               .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]))
    else:
        display(conteo.style.hide(axis="index"))
    
    return None



# Metodo
def cuantas_veces_todos(dfs: dict, index = False) -> None:
    for tick, df in dfs.items():
        cuantas_veces(dfs, tick, index=index)






def agrupar_journals(df: pd.DataFrame, journals: list) -> None:
    """
    Agrega los datos de noticias de diferentes journals al DataFrame total.
    Args:
        dfs (pd.DataFrame): DataFrame con las noticias.
        journals (list): Lista de nombres de journals a filtrar.
    """
    
    # Generalizar para cualquier lista de journals
    dftotal = pd.DataFrame(columns=df.columns)
    fila_vacia = pd.DataFrame([{col: "" for col in df.columns}])

    for journal in journals:
        # Filtrar por journal
        df_journal = df[df['url'].str.contains(journal, case=False, na=False)]
        # Fila con el nombre del journal
        fila_nombre = fila_vacia.copy()
        fila_vacia['fecha'] = "---"
        fila_nombre['título'] = journal
        # Concatenar: nombre, noticias, fila vacía
        dftotal = pd.concat([dftotal, fila_vacia, fila_nombre, df_journal], ignore_index=True)
    
    return dftotal



def highlight_fila_nombre(row, journals):
    if row['título'] in journals:
        return ['font-weight: bold'] * len(row)
    else:
        return [''] * len(row)
    




def quiero_ver(tick: str, journals: list, dfs: dict, index = False) -> None:
    """
    Muestra las noticias de un ticker específico y los journals seleccionados.
    Args:
        tick (str): Ticker de la acción a consultar.
        journals (list): Lista de nombres de journals a filtrar.
        dfs (dict): Diccionario de DataFrames de noticias por ticker.
        index (bool): Si es True, muestra el índice del DataFrame. Por defecto es False.
    Returns:
        None: Muestra las noticias en un formato HTML estilizado.
    Raises:
        ValueError: Si el ticker no se encuentra en el diccionario de DataFrames.
    """
    dftotal = agrupar_journals(dfs[tick], journals)
    dftotal = hiperreferenciar_titulos(dftotal)
    display(HTML(f'<h2 style="text-align:left;">Noticias de {tick}</h2>'))

    if index:
        display(
            dftotal.drop(columns=['url'])
            .style
            .apply(lambda row: highlight_fila_nombre(row, journals), axis=1)
            .set_properties(subset=['fecha'], **{'text-align': 'right'})
            .set_properties(subset=['título'], **{'text-align': 'left'})
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
        )
    else:
        display(
            dftotal.drop(columns=['url'])
            .style
            .hide(axis="index")
            .apply(lambda row: highlight_fila_nombre(row, journals), axis=1)
            .set_properties(subset=['fecha'], **{'text-align': 'right'})
            .set_properties(subset=['título'], **{'text-align': 'left'})
            .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
        )
    
    return None

def quiero_ver_todos(journals, dfs, index = False) -> None:
    """    Muestra las noticias de todos los tickers para los journals seleccionados.
    """
    for tick, df in dfs.items():
        quiero_ver(tick, journals, dfs, index=index)





