# app.py
# Streamlit app - Leitor de Comprovantes PIX (Imagem + PDF) usando Google Cloud Vision OCR

import io
import re
import zipfile
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ---- Imports opcionais (para não quebrar o app se faltar) ----
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    pdfplumber = None
    HAS_PDFPLUMBER = False

from google.cloud import vision
from google.oauth2 import service_account


# =========================================================
# 1) OCR - Google Vision client (Streamlit Secrets)
# =========================================================
@st.cache_resource
def get_vision_client() -> vision.ImageAnnotatorClient:
    # No Streamlit Cloud você vai colar isso em Settings -> Secrets
    # [gcp_service_account]
    # type="service_account"
    # ...
    info = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(info)
    return vision.ImageAnnotatorClient(credentials=creds)


def vision_ocr_bytes(image_bytes: bytes) -> str:
    client = get_vision_client()
    img = vision.Image(content=image_bytes)
    resp = client.document_text_detection(image=img)

    if resp.error.message:
        raise RuntimeError(resp.error.message)

    return (resp.full_text_annotation.text or "").strip()


# =========================================================
# 2) Preprocessamento (usa OpenCV se disponível)
# =========================================================
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """Melhora contraste/ruído para OCR em prints/fotos. Se cv2 não existir, retorna original."""
    if not HAS_CV2:
        return pil_img

    img = np.array(pil_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = 1.5 if max(h, w) < 2000 else 1.2
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    out = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(out)


# =========================================================
# 3) Parsing PIX
# =========================================================
def norm_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def br_money_to_float(txt: str) -> Optional[float]:
    if not txt:
        return None
    m = re.search(
        r"(?i)(?:R\$\s*)?([0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})|[0-9]+(?:,[0-9]{2})|[0-9]+(?:\.[0-9]{2}))",
        txt
    )
    if not m:
        return None
    v = m.group(1).strip()
    if "," in v and v.count(",") == 1:
        v = v.replace(".", "").replace(",", ".")
    else:
        v = v.replace(".", "")
    try:
        return float(v)
    except:
        return None


def find_date_time(text: str) -> Tuple[Optional[str], Optional[str]]:
    months = {
        "jan": "01", "janeiro": "01",
        "fev": "02", "fevereiro": "02",
        "mar": "03", "março": "03", "marco": "03",
        "abr": "04", "abril": "04",
        "mai": "05", "maio": "05",
        "jun": "06", "junho": "06",
        "jul": "07", "julho": "07",
        "ago": "08", "agosto": "08",
        "set": "09", "setembro": "09",
        "out": "10", "outubro": "10",
        "nov": "11", "novembro": "11",
        "dez": "12", "dezembro": "12",
    }

    date = None
    time = None

    m1 = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text)
    if m1:
        date = f"{m1.group(1)}/{m1.group(2)}/{m1.group(3)}"
        mt = re.search(r"\b(\d{2}:\d{2}(?::\d{2})?)\b", text[m1.end(): m1.end() + 120])
        if mt:
            time = mt.group(1)

    if not date:
        m2 = re.search(r"\b(\d{2})\s*/\s*([A-Za-zçÇ\.]{3,})\s*/\s*(\d{4})\b", text, flags=re.I)
        if m2:
            d = m2.group(1)
            mon_raw = re.sub(r"\.", "", m2.group(2).lower())
            mon = months.get(mon_raw[:3], None)
            if mon:
                date = f"{d}/{mon}/{m2.group(3)}"
                mt = re.search(r"\b(\d{2}:\d{2}(?::\d{2})?)\b", text[m2.end(): m2.end() + 160])
                if mt:
                    time = mt.group(1)

    if not date:
        m3 = re.search(r"\b(\d{1,2})\s+([A-Za-zçÇ\.]{3,})\s+(\d{4})\b", text, flags=re.I)
        if m3:
            d = f"{int(m3.group(1)):02d}"
            mon_raw = re.sub(r"\.", "", m3.group(2).lower())
            mon = months.get(mon_raw[:3], None)
            if mon:
                date = f"{d}/{mon}/{m3.group(3)}"
                mt = re.search(r"\b(\d{2}:\d{2}(?::\d{2})?)\b", text[m3.end(): m3.end() + 220])
                if mt:
                    time = mt.group(1)

    return date, time


def extract_after_label(text: str, label_patterns: List[str], max_lines: int = 3) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines()]

    def looks_like_new_section(s: str) -> bool:
        return bool(re.search(
            r"^(cnpj|cpf|cpf/cnpj|chave|institui|banco|ag[eê]ncia|agencia|conta|id|valor|data|hora|tipo|autent|controle|c[oó]digo)\b",
            s, flags=re.I
        ))

    for i, ln in enumerate(lines):
        for pat in label_patterns:
            m = re.search(pat, ln, flags=re.I)
            if not m:
                continue

            same = re.search(r"[:\-]\s*(.+)$", ln)
            if same and same.group(1).strip():
                cand = same.group(1).strip()
                if not looks_like_new_section(cand):
                    return cand

            collected = []
            for j in range(i + 1, min(i + 1 + max_lines + 6, len(lines))):
                cand = lines[j].strip()
                if not cand:
                    continue
                if re.fullmatch(r"(?i)nome", cand):
                    continue
                if looks_like_new_section(cand):
                    break
                collected.append(cand)
                if len(collected) >= max_lines:
                    break

            if collected:
                return " ".join(collected).strip()

    return None


def extract_cpf_cnpj_near(text: str, anchor_patterns: List[str]) -> Optional[str]:
    cpfcnpj_pat = r"(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}|\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\*{2,}\.?\*{2,}\.?\*{2,}/?\*{2,}-?\d{2}|\*{2,}\.?\d{3}\.?\d{3}-?\*{2})"
    lines = text.splitlines()

    for i, ln in enumerate(lines):
        for a in anchor_patterns:
            if re.search(a, ln, flags=re.I):
                chunk = "\n".join(lines[i:i+10])

                m0 = re.search(r"(?i)cpf\s*/?\s*cnpj\s*[:\-]?\s*" + cpfcnpj_pat, ln)
                if m0:
                    return m0.group(1).strip()

                m = re.search(r"(?i)cpf\s*/?\s*cnpj\s*[:\-]?\s*" + cpfcnpj_pat, chunk)
                if m:
                    return m.group(1).strip()

                m2 = re.search(cpfcnpj_pat, chunk)
                if m2:
                    return m2.group(1).strip()

    return None


def extract_chave_pix(text: str) -> Optional[str]:
    m = re.search(r"(?i)\bchave\s*(?:pix)?\s*(?:do\s*recebedor)?\s*[:\-]?\s*([A-Za-z0-9@\.\-\_\/]{8,})", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(?i)\bchave\s*[:\-]?\s*([0-9\.\*\/\-]{8,})", text)
    if m2:
        return m2.group(1).strip()
    return None


def extract_id_transacao(text: str) -> Optional[str]:
    patterns = [
        r"(?i)\bid\s*[/\-\s]*transa[cç][aã]o\s*[:\-]?\s*([A-Za-z0-9\-]{10,})",
        r"(?i)\bid\s+da\s+transa[cç][aã]o\s*[:\-]?\s*([A-Za-z0-9\-]{10,})",
        r"(?i)\bc[oó]digo\s+da\s+transa[cç][aã]o\s+(?:pix)?\s*[:\-]?\s*([A-Za-z0-9\-]{10,})",
        r"(?i)\btransa[cç][aã]o\s*[:\-]?\s*([A-Za-z0-9]{20,})"
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None


def extract_bancos(text: str) -> Tuple[Optional[str], Optional[str]]:
    banco_ori = None
    banco_dest = None

    m_ori = re.search(r"(?is)conta\s+de\s+origem.*?\bBanco\s*[:\-]\s*([^\n]+)", text)
    if m_ori:
        banco_ori = m_ori.group(1).strip()

    m_dest = re.search(r"(?is)(?:recebedor|destinat[aá]rio|para).*?\bBanco\s*[:\-]\s*([^\n]+)", text)
    if m_dest:
        banco_dest = m_dest.group(1).strip()

    if not banco_dest:
        banco_dest = extract_after_label(
            text,
            label_patterns=[r"(?i)^\s*institui[cç][aã]o\s*$", r"(?i)\binstitui[cç][aã]o\b"],
            max_lines=1
        )

    if not banco_ori:
        m = re.search(r"(?is)dados\s+do\s+pagador.*?institui[cç][aã]o\s*\n([^\n]+)", text)
        if m:
            banco_ori = m.group(1).strip()

    if not banco_dest:
        m = re.search(r"(?is)dados\s+do\s+recebedor.*?institui[cç][aã]o\s*\n([^\n]+)", text)
        if m:
            banco_dest = m.group(1).strip()

    def clean_bank(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = s.strip()
        s = re.sub(r"^\s*\d+\s*[-–]\s*", "", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s or None

    return clean_bank(banco_ori), clean_bank(banco_dest)


def extract_valor(text: str) -> Optional[float]:
    for pat in [
        r"(?i)\bvalor\s+pago\b[^\n]*",
        r"(?i)\bvalor\s+da\s+transfer[êe]ncia\b[^\n]*",
        r"(?i)\bvalor\b[^\n]*",
    ]:
        m = re.search(pat, text)
        if m:
            v = br_money_to_float(m.group(0))
            if v is not None:
                return v
    m2 = re.search(r"(?i)R\$\s*[0-9\.\,]+", text)
    if m2:
        return br_money_to_float(m2.group(0))
    return None


def extract_pagador_recebedor(text: str) -> Tuple[Optional[str], Optional[str]]:
    pagador = extract_after_label(
        text,
        label_patterns=[
            r"(?i)^\s*pagador\s*$",
            r"(?i)^\s*dados\s+do\s+pagador\s*$",
            r"(?i)^\s*de\s*$",
            r"(?i)\bpagador\b",
        ],
        max_lines=2
    )

    recebedor = extract_after_label(
        text,
        label_patterns=[
            r"(?i)^\s*recebedor\s*$",
            r"(?i)^\s*dados\s+do\s+recebedor\s*$",
            r"(?i)^\s*para\s*$",
            r"(?i)\brecebedor\b",
        ],
        max_lines=2
    )

    if not recebedor:
        m = re.search(r"(?im)^\s*(?:pago\s+para|para)\s*[:\-]?\s*(.+)$", text)
        if m:
            recebedor = m.group(1).strip()

    if not pagador:
        m = re.search(r"(?im)^\s*de\s*[:\-]?\s*(.+)$", text)
        if m:
            pagador = m.group(1).strip()

    def clean(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        x = re.sub(r"(?i)\bnome\b\s*[:\-]?\s*", "", x).strip()
        x = re.sub(r"\s{2,}", " ", x)
        x = re.sub(r"(?i)\bcpf/cnpj\b.*$", "", x).strip()
        x = re.sub(r"(?i)\binstitui[cç][aã]o\b.*$", "", x).strip()
        x = re.sub(r"(?i)\bbanco\b.*$", "", x).strip()
        return x or None

    return clean(pagador), clean(recebedor)


def score_fields(d: Dict) -> int:
    keys = [
        "data_comprovante", "horario_comprovante", "valor",
        "pagador", "recebedor", "banco_origem", "banco_destino",
        "id_transacao", "chave_pix",
        "cpf_cnpj_pagador", "cpf_cnpj_recebedor"
    ]
    s = 0
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        s += 1
    return s


def parse_pix_fields(text: str) -> Dict:
    text = norm_spaces(text)

    data, hora = find_date_time(text)
    valor = extract_valor(text)
    pagador, recebedor = extract_pagador_recebedor(text)
    banco_origem, banco_destino = extract_bancos(text)
    id_transacao = extract_id_transacao(text)
    chave_pix = extract_chave_pix(text)

    cpf_cnpj_pagador = extract_cpf_cnpj_near(
        text,
        anchor_patterns=[r"(?i)\bpagador\b", r"(?i)\bdados\s+do\s+pagador\b", r"(?i)^\s*de\s*$"]
    )
    cpf_cnpj_recebedor = extract_cpf_cnpj_near(
        text,
        anchor_patterns=[r"(?i)\brecebedor\b", r"(?i)\bdados\s+do\s+recebedor\b", r"(?i)^\s*para\s*$"]
    )

    return {
        "data_comprovante": data,
        "horario_comprovante": hora,
        "valor": valor,
        "pagador": pagador,
        "recebedor": recebedor,
        "banco_origem": banco_origem,
        "banco_destino": banco_destino,
        "id_transacao": id_transacao,
        "chave_pix": chave_pix,
        "cpf_cnpj_pagador": cpf_cnpj_pagador,
        "cpf_cnpj_recebedor": cpf_cnpj_recebedor,
        "texto_ocr": text,
    }


# =========================================================
# 4) Leitura de arquivos (imagem/pdf/zip)
# =========================================================
def pdf_to_images_bytes(pdf_bytes: bytes, max_pages: int = 3) -> List[bytes]:
    if not HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber não está instalado. Instale para ler PDFs.")

    images = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            page = pdf.pages[i]
            pil = page.to_image(resolution=240).original.convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            images.append(buf.getvalue())
    return images


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 3) -> str:
    if not HAS_PDFPLUMBER:
        return ""
    chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        n = min(len(pdf.pages), max_pages)
        for i in range(n):
            txt = pdf.pages[i].extract_text() or ""
            txt = txt.strip()
            if txt:
                chunks.append(txt)
    return "\n\n".join(chunks).strip()


def process_one_image_bytes(filename: str, img_bytes: bytes, use_preprocess: bool = True) -> Dict:
    text1 = vision_ocr_bytes(img_bytes)
    d1 = parse_pix_fields(text1)
    d1["arquivo"] = filename
    d1["metodo"] = "opc1_sem_preprocess"
    s1 = score_fields(d1)

    if use_preprocess and HAS_CV2:
        pil = Image.open(io.BytesIO(img_bytes))
        pil2 = preprocess_for_ocr(pil)
        buf = io.BytesIO()
        pil2.save(buf, format="PNG")
        text2 = vision_ocr_bytes(buf.getvalue())
        d2 = parse_pix_fields(text2)
        d2["arquivo"] = filename
        d2["metodo"] = "opc2_com_preprocess"
        s2 = score_fields(d2)
        return d2 if s2 >= s1 else d1

    return d1


def process_uploaded_file(uploaded_file, use_preprocess: bool = True) -> Dict:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith(".pdf"):
        txt = extract_text_from_pdf(raw, max_pages=3)
        best = None

        if txt:
            d_txt = parse_pix_fields(txt)
            d_txt["arquivo"] = uploaded_file.name
            d_txt["metodo"] = "pdf_texto"
            best = d_txt

        imgs = pdf_to_images_bytes(raw, max_pages=3)
        candidates = []
        for idx, img_b in enumerate(imgs, start=1):
            d = process_one_image_bytes(f"{uploaded_file.name}#p{idx}", img_b, use_preprocess=use_preprocess)
            candidates.append(d)

        if candidates:
            best_ocr = max(candidates, key=score_fields)
            if best is None or score_fields(best_ocr) >= score_fields(best):
                best = best_ocr

        if best is None:
            best = {"arquivo": uploaded_file.name, "erro": "Não foi possível extrair texto do PDF."}
        return best

    if name.endswith((".png", ".jpg", ".jpeg")):
        return process_one_image_bytes(uploaded_file.name, raw, use_preprocess=use_preprocess)

    return {"arquivo": uploaded_file.name, "erro": "Formato não suportado (use PDF/JPG/PNG ou ZIP)."}


def expand_zip(uploaded_zip) -> List[Tuple[str, bytes]]:
    raw = uploaded_zip.read()
    out = []
    with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            fname = info.filename
            low = fname.lower()
            if low.endswith((".pdf", ".png", ".jpg", ".jpeg")):
                out.append((fname, z.read(info)))
    return out


def process_zip_files(files: List[Tuple[str, bytes]], use_preprocess: bool = True) -> List[Dict]:
    rows = []
    for fname, b in files:
        if fname.lower().endswith(".pdf"):
            txt = extract_text_from_pdf(b, max_pages=3)
            best = None
            if txt:
                d_txt = parse_pix_fields(txt)
                d_txt["arquivo"] = fname
                d_txt["metodo"] = "pdf_texto"
                best = d_txt

            imgs = pdf_to_images_bytes(b, max_pages=3)
            candidates = []
            for idx, img_b in enumerate(imgs, start=1):
                d = process_one_image_bytes(f"{fname}#p{idx}", img_b, use_preprocess=use_preprocess)
                candidates.append(d)

            if candidates:
                best_ocr = max(candidates, key=score_fields)
                if best is None or score_fields(best_ocr) >= score_fields(best):
                    best = best_ocr

            if best is None:
                best = {"arquivo": fname, "erro": "Não foi possível extrair texto do PDF."}
            rows.append(best)
        else:
            rows.append(process_one_image_bytes(fname, b, use_preprocess=use_preprocess))
    return rows


# =========================================================
# 5) UI Streamlit
# =========================================================
st.set_page_config(page_title="Extrator de comprovantes Pix (Google Vision)", layout="wide")
st.title("Extrator de comprovantes Pix (PDF + imagem) — Google Cloud Vision OCR")

with st.expander("⚙️ Configurações", expanded=True):
    # Não exponha segredo / caminho. Só status.
    secrets_ok = "gcp_service_account" in st.secrets
    st.write(f"**Google Vision Secrets:** {'OK ✅' if secrets_ok else 'NÃO configurado ❌'}")
    show_text = st.checkbox("Mostrar texto OCR (debug)", value=False)
    use_preprocess = st.checkbox("Usar preprocessamento (OpenCV)", value=True, disabled=not HAS_CV2)
    st.caption(f"OpenCV: {'OK' if HAS_CV2 else 'NÃO instalado'} | pdfplumber: {'OK' if HAS_PDFPLUMBER else 'NÃO instalado'}")

uploaded = st.file_uploader(
    "Envie comprovantes (PDF/JPG/PNG) ou um ZIP com vários arquivos",
    type=["pdf", "png", "jpg", "jpeg", "zip"],
    accept_multiple_files=True
)

if uploaded:
    total_files = 0
    zip_items: List[Tuple[str, bytes]] = []
    normal_files = []

    for f in uploaded:
        if f.name.lower().endswith(".zip"):
            items = expand_zip(f)
            zip_items.extend(items)
            total_files += len(items)
        else:
            normal_files.append(f)
            total_files += 1

    st.write(f"Arquivos encontrados: **{total_files}**")

    if st.button("Processar arquivos", type="primary"):
        results: List[Dict] = []
        prog = st.progress(0.0)
        done = 0

        for f in normal_files:
            try:
                results.append(process_uploaded_file(f, use_preprocess=use_preprocess))
            except Exception as e:
                results.append({"arquivo": f.name, "erro": str(e)})
            done += 1
            prog.progress(min(done / max(total_files, 1), 1.0))

        if zip_items:
            try:
                rows = process_zip_files(zip_items, use_preprocess=use_preprocess)
                results.extend(rows)
            except Exception as e:
                results.append({"arquivo": "ZIP", "erro": str(e)})
            prog.progress(1.0)

        df = pd.DataFrame(results)

        if not show_text and "texto_ocr" in df.columns:
            df = df.drop(columns=["texto_ocr"])

        st.subheader("Resultado")
        st.dataframe(df, use_container_width=True, height=520)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV", data=csv_bytes, file_name="comprovantes.csv", mime="text/csv")

        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Base")
        st.download_button(
            "Baixar Excel",
            data=out_xlsx.getvalue(),
            file_name="comprovantes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Envie arquivos (PDF/JPG/PNG) ou um ZIP. Depois clique em **Processar arquivos**.")
