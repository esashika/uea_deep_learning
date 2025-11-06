#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baixa imagens a partir da página de resultados do DuckDuckGo Imagens
(simula comportamento de navegador com Selenium).

Uso: apenas execute o script. Configurações no bloco `CONFIG` abaixo.
"""

import os
import re
import time
import pathlib
import requests
from urllib.parse import quote_plus, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# webdriver-manager facilita obter/chamar o driver automaticamente
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options

# -------------------- CONFIGURAÇÃO --------------------
QUERY = "Trout fish images"           # termo de busca
NUM_IMAGES = 100            # quantas imagens tentar baixar
OUTDIR = "img/Trout"     # pasta de saída
HEADLESS = True            # roda sem mostrar janela do navegador?
SCROLL_PAUSE = 0.8         # pausa entre scrolls (s)
MAX_SCROLLS = 60           # limite de scrolls para carregar imagens
TIMEOUT_REQUEST = 30       # timeout para downloads (s)
# -------------------------------------------------------

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-_\. ]", "_", name)
    return name[:120].strip()

def ensure_outdir(path):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def gather_image_urls_ddg(driver, query, num_images, max_scrolls=60, pause=0.8):
    """
    Navega até DuckDuckGo imagens e coleta URLs visíveis das imagens.
    Retorna lista de URLs (não duplicadas) até num_images.
    """
    search = quote_plus(query)
    url = f"https://duckduckgo.com/?q={search}&iax=images&ia=images"
    driver.get(url)
    time.sleep(1.0)

    image_urls = []
    seen = set()
    scrolls = 0

    # A página carrega imagens dinamicamente ao rolar.
    while len(image_urls) < num_images and scrolls < max_scrolls:
        # procura por <img> tags na área de resultados
        imgs = driver.find_elements(By.TAG_NAME, "img")
        for img in imgs:
            try:
                src = img.get_attribute("src") or ""
                data_src = img.get_attribute("data-src") or ""
                # DuckDuckGo às vezes usa srcset; preferimos data-src > src
                candidate = data_src if data_src else src
                if not candidate:
                    continue
                # filtrar data: URIs e imagens muito pequenas (thumbnails inline)
                if candidate.startswith("data:"):
                    continue
                # normalizar e evitar duplicatas
                if candidate in seen:
                    continue
                seen.add(candidate)
                image_urls.append(candidate)
                if len(image_urls) >= num_images:
                    break
            except Exception:
                continue

        # rolar até o fim da página
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        scrolls += 1

        # algumas vezes há um botão "More Results" que precisa ser clicado
        try:
            more = driver.find_element(By.CSS_SELECTOR, "a[data-testid='more-results-button']")
            if more:
                more.click()
                time.sleep(pause)
        except Exception:
            pass

    return image_urls[:num_images]

def guess_extension_from_url(url: str):
    path = urlparse(url).path
    if "." in path:
        ext = path.split(".")[-1].lower()
        if ext in ("jpg","jpeg","png","gif","webp","bmp","tiff","svg"):
            return "." + ext
    # fallback
    return ".jpg"

def download_images(urls, outdir: pathlib.Path):
    headers = {"User-Agent": UA, "Referer": "https://duckduckgo.com/"}
    ok = 0
    fail = 0
    for i, url in enumerate(urls, 1):
        try:
            resp = requests.get(url, stream=True, timeout=TIMEOUT_REQUEST, headers=headers)
            if resp.status_code == 200 and resp.content:
                # determinar extensão
                ctype = resp.headers.get("Content-Type", "")
                ext = None
                if ctype:
                    if "jpeg" in ctype:
                        ext = ".jpg"
                    elif "png" in ctype:
                        ext = ".png"
                    elif "gif" in ctype:
                        ext = ".gif"
                    elif "webp" in ctype:
                        ext = ".webp"
                if not ext:
                    ext = guess_extension_from_url(url)
                outname = f"{i:03d}_{sanitize_filename(urlparse(url).path.split('/')[-1] or 'img')}{ext}"
                outpath = outdir / outname
                with open(outpath, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                print(f"[{i:03d}] OK -> {outpath.name}")
                ok += 1
            else:
                print(f"[{i:03d}] ERRO HTTP {resp.status_code} para {url}")
                fail += 1
        except Exception as e:
            print(f"[{i:03d}] ERRO: {e} ({url})")
            fail += 1
    print(f"Download concluído: {ok} sucesso | {fail} falhas")

def create_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={UA}")
    # Evita imagens sendo bloqueadas por algum bloqueador do driver
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def main():
    outdir = ensure_outdir(OUTDIR)
    print(f"Pesquisando por: {QUERY!r} (modo navegador simulado)")
    driver = create_driver(headless=HEADLESS)
    try:
        urls = gather_image_urls_ddg(driver, QUERY, NUM_IMAGES, max_scrolls=MAX_SCROLLS, pause=SCROLL_PAUSE)
        print(f"Encontradas {len(urls)} URLs (tentando baixar)...")
    finally:
        driver.quit()

    if not urls:
        print("Nenhuma URL encontrada. Tente ajustar o QUERY ou aumentar MAX_SCROLLS/SCROLL_PAUSE.")
        return

    download_images(urls, outdir)
    print(f"Imagens salvas em: {outdir.resolve()}")

if __name__ == "__main__":
    main()
