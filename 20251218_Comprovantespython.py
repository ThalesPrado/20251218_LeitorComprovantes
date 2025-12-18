# =========================
# 20251218_Comprovantespython.py
# Extrator de Comprovantes Pix (PDF + Imagem) - Streamlit
# OPÇÃO 1 + OPÇÃO 2 (heurísticas + OCR por rotação)
# =========================

import io
import re
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance

import pytesseract
import pdfplumber


# =========================
# FIXO: CAMINHO DO TESSERACT (WINDOWS)
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================
# MODELO
# =========================
@dataclass
class PixRecord:
    arquivo: str
    data_comprovante: Optional[str] = None
    horario_comprovante: Optional[str] = None
    pagador: Optional[str] = None
    banco_origem: Optional[str] = None
    banco_destino: Optional[str] = None
    valor: Optional[float] = None
    recebedor: Optional[str] = None
    id_transacao: Optional[str] = None
    fonte_texto: Optional[str] = None  # "pdf_text" | "ocr_img" | "pdf_text_vazio"


# =========================
# OCR / TEXTO
# =========================
def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Pré-processamento forte:
    - escala 2x
    - grayscale
    - contraste
    - binarização
    """
    img = img.convert("RGB")
    img = img.resize((img.width * 2, img.height * 2))

    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)

    bw = gray.point(lambda x: 255 if x > 175 else 0, mode="1")
    return bw


def _score_ocr_text(txt: str) -> int:
    """
    Escolhe o melhor OCR:
    Pontua por sinais de comprovante (data/hora/R$/Pix/Instituição/ID etc)
    """
    if not txt:
        return 0
    t = txt.lower()
    score = 0

    # tokens fortes
    for tok, pts in [
        ("pix", 5),
        ("r$", 5),
        ("comprovante", 3),
        ("instituição", 3),
        ("instituicao", 3),
        ("pagador", 3),
        ("recebedor", 3),
        ("destino", 2),
        ("origem", 2),
        ("id", 2),
        ("transa", 2),
        ("cpf", 2),
        ("cnpj", 2),
    ]:
        if tok in t:
            score += pts

    # padrões
    if re.search(r"\d{2}/\d{2}/\d{4}", t):
        score += 6
    if re.search(r"\d{2}:\d{2}:\d{2}", t):
        score += 6
    if re.search(r"r\$\s*[\d\.\s]+,\d{2}", txt):
        score += 8
    if len(txt) > 500:
        score += 2
    if len(txt) > 1500:
        score += 3

    return score


def extract_text_from_image_bytes(img_bytes: bytes) -> str:
    """
    OPÇÃO 2: OCR 4 rotações (0/90/180/270) e escolhe o melhor resultado.
    """
    base = Image.open(io.BytesIO(img_bytes))
    config = r"--oem 3 --psm 6"

    best_txt = ""
    best_score = -1

    for angle in (0, 90, 180, 270):
        img = base.rotate(angle, expand=True)
        img = _preprocess_for_ocr(img)

        try:
            txt = pytesseract.image_to_string(img, lang="por", config=config) or ""
        except Exception:
            txt = pytesseract.image_to_string(img, config=config) or ""

        sc = _score_ocr_text(txt)
        if sc > best_score:
            best_score = sc
            best_txt = txt

    return best_txt or ""


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    return "\n".join(parts).strip()


# =========================
# PARSER (ROBUSTO + OPÇÃO 1)
# =========================
def _clean_line(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s

def _lines(text: str) -> List[str]:
    text = (text or "").replace("\r", "")
    return [_clean_line(x) for x in text.split("\n") if _clean_line(x)]

def _norm_valor(raw: str) -> Optional[float]:
    if not raw:
        return None
    v = raw.replace("R$", "").strip()
    v = v.replace(".", "").replace(",", ".")
    try:
        return float(v)
    except Exception:
        return None


# -------- OPÇÃO 1: limpeza de nomes --------
NAME_CUT_TOKENS = [
    "agencia", "agência", "conta", "cpf", "cnpj", "chave", "chave pix",
    "instituicao", "instituição", "id", "transa", "código", "codigo"
]

def _clean_person_name(s: Optional[str]) -> Optional[str]:
    """
    Limpa casos tipo:
    "TATIANE ... agência 7180 - conta ..."
    """
    if not s:
        return None
    s = _clean_line(s)

    low = s.lower()
    cut_pos = None
    for tok in NAME_CUT_TOKENS:
        p = low.find(tok)
        if p != -1:
            if cut_pos is None or p < cut_pos:
                cut_pos = p

    if cut_pos is not None and cut_pos > 3:
        s = s[:cut_pos].strip(" :-–|")

    # remove traços e sobras
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Se ficou muito curto, descarta
    if len(s) < 3:
        return None

    # Evitar lixo comum
    bad = {"informacoes", "informação", "informaçãoes", "informacoes banco", "comprovante"}
    if s.lower() in bad:
        return None

    return s


# -------- datetime --------
def _find_datetime(text: str, lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"(\d{2}/\d{2}/\d{4})\s*[-–]\s*(\d{2}:\d{2}:\d{2})", text)
    if m:
        return m.group(1), m.group(2)

    m = re.search(r"(\d{2}/\d{2}/\d{4}).*?(?:às|as)\s*(\d{2}:\d{2}:\d{2})", text, re.I)
    if m:
        return m.group(1), m.group(2)

    months = {"jan":"01","fev":"02","mar":"03","abr":"04","mai":"05","jun":"06","jul":"07","ago":"08","set":"09","out":"10","nov":"11","dez":"12"}
    m = re.search(r"(\d{1,2})\s+([a-z]{3})\.?\s+(\d{4}).*?(\d{2}:\d{2}:\d{2})", text.lower())
    if m and m.group(2) in months:
        dd = int(m.group(1))
        return f"{dd:02d}/{months[m.group(2)]}/{m.group(3)}", m.group(4)

    for l in lines:
        mm = re.search(r"(\d{2}/\d{2}/\d{4}).*?(\d{2}:\d{2}:\d{2})", l)
        if mm:
            return mm.group(1), mm.group(2)

    return None, None


def _find_valor(text: str, lines: List[str]) -> Optional[float]:
    for l in lines:
        if "valor" in l.lower():
            m = re.search(r"R\$\s*([\d\.\s]+,\d{2})", l)
            if m:
                return _norm_valor(m.group(1))

    m = re.search(r"R\$\s*([\d\.\s]+,\d{2})", text)
    if m:
        return _norm_valor(m.group(1))

    # fallback: alguns OCRs removem vírgula. Ex: "10000" (nunca é perfeito)
    return None


def _find_id(text: str, lines: List[str]) -> Optional[str]:
    for l in lines:
        m = re.search(r"(ID/Transa[cç][aã]o|ID da transa[cç][aã]o|ID da transacao|C[oó]digo da transa[cç][aã]o Pix)\s*:?\s*([A-Z0-9\-]{15,})",
                      l, re.I)
        if m:
            return m.group(2).strip()

    for i, l in enumerate(lines):
        if "id" in l.lower() and "transa" in l.lower():
            if i + 1 < len(lines):
                cand = lines[i + 1].strip()
                if re.search(r"[A-Z0-9\-]{15,}", cand):
                    return cand
    return None


def _find_after_label(lines: List[str], labels: List[str], max_lines: int = 3) -> Optional[str]:
    labels_low = [x.lower() for x in labels]
    for i, l in enumerate(lines):
        low = l.lower().strip()
        if any(low == lab or low.startswith(lab + ":") for lab in labels_low):
            parts = []
            for k in range(i + 1, min(i + 1 + max_lines, len(lines))):
                cand = _clean_line(lines[k])
                if not cand:
                    break
                # para quando chega em campos
                if any(tok in cand.lower() for tok in ["cpf", "cnpj", "chave", "institu", "agência", "agencia", "conta", "id", "transa", "valor"]):
                    break
                parts.append(cand)
            if parts:
                return " ".join(parts).strip()
    return None


def _find_bank_anywhere(full_text: str) -> List[str]:
    """
    Detecta bancos que aparecem no texto (OCR/ PDF).
    Retorna lista (pode haver mais de um).
    """
    t = full_text.upper()

    bank_map = [
        ("BCO DO BRASIL", "BCO DO BRASIL S.A."),
        ("BANCO DO BRASIL", "BCO DO BRASIL S.A."),
        ("SANTANDER", "BCO SANTANDER (BRASIL) S.A."),
        ("ITAÚ", "ITAÚ"),
        ("ITAU", "ITAÚ"),
        ("BRADESCO", "BRADESCO"),
        ("SICREDI", "SICREDI"),
        ("NUBANK", "NU PAGAMENTOS"),
        ("NU PAGAMENTOS", "NU PAGAMENTOS"),
        ("PICPAY", "PICPAY"),
        ("PAGBANK", "PAGBANK/PAGSEGURO"),
        ("PAGSEGURO", "PAGBANK/PAGSEGURO"),
        ("CAIXA", "CAIXA"),
        ("INTER", "BANCO INTER"),
        ("C6", "C6 BANK"),
    ]

    found = []
    for key, norm in bank_map:
        if key in t:
            found.append(norm)

    # remove duplicatas mantendo ordem
    uniq = []
    for b in found:
        if b not in uniq:
            uniq.append(b)
    return uniq


def parse_pix_text(text: str, filename: str) -> PixRecord:
    rec = PixRecord(arquivo=filename)

    raw_text = text or ""
    lines = _lines(raw_text)
    full = " ".join(lines)

    rec.data_comprovante, rec.horario_comprovante = _find_datetime(full, lines)
    rec.valor = _find_valor(full, lines)
    rec.id_transacao = _find_id(full, lines)

    # -------- EXTRAÇÃO DE NOMES (com limpeza) --------
    # tenta padrões comuns: "Dados do recebedor/pagador"
    # Santander
    if "dados do recebedor" in full.lower():
        r = _find_after_label(lines, ["Para", "Recebedor", "Pago para"])
        rec.recebedor = _clean_person_name(r) if r else rec.recebedor

    if "dados do pagador" in full.lower():
        p = _find_after_label(lines, ["Pagador", "De", "Solicitante"])
        rec.pagador = _clean_person_name(p) if p else rec.pagador

    # Nubank (Destino/Origem com "Nome")
    if not rec.recebedor and "destino" in full.lower():
        r = _find_after_label(lines, ["Nome"])
        rec.recebedor = _clean_person_name(r) if r else rec.recebedor

    if not rec.pagador and "origem" in full.lower():
        p = _find_after_label(lines, ["Nome"])
        rec.pagador = _clean_person_name(p) if p else rec.pagador

    # genérico De/Para
    if not rec.recebedor:
        r = _find_after_label(lines, ["Para", "Nome do destinatário", "Nome do destinatario", "Dados de quem recebeu", "Pago para"])
        rec.recebedor = _clean_person_name(r) if r else rec.recebedor

    if not rec.pagador:
        p = _find_after_label(lines, ["De", "Nome do pagador", "Dados de quem fez a transação", "Solicitante", "Pagador"])
        rec.pagador = _clean_person_name(p) if p else rec.pagador

    # BB (SISBB)
    if not rec.recebedor:
        m = re.search(r"PAGO PARA:\s*(.+?)\s*(?:CNPJ|CHAVE PIX|INSTITUICAO|INSTITUIÇÃO)", raw_text, re.I)
        if m:
            rec.recebedor = _clean_person_name(_clean_line(m.group(1)))

    if not rec.pagador:
        m = re.search(r"CLIENTE:\s*(.+?)\s*(?:AGENCIA|AGÊNCIA|CONTA)", raw_text, re.I)
        if m:
            rec.pagador = _clean_person_name(_clean_line(m.group(1)))

    # -------- OPÇÃO 1: bancos por fallback inteligente --------
    banks_found = _find_bank_anywhere(full)

    # Heurística origem/destino:
    # - Se só 1 banco encontrado, coloca em destino se tiver "para/recebedor/destino", senão origem
    # - Se 2+, tenta: primeiro = origem, último = destino
    if banks_found:
        if len(banks_found) == 1:
            b = banks_found[0]
            if ("para" in full.lower()) or ("destino" in full.lower()) or ("recebedor" in full.lower()):
                rec.banco_destino = rec.banco_destino or b
            else:
                rec.banco_origem = rec.banco_origem or b
        else:
            rec.banco_origem = rec.banco_origem or banks_found[0]
            rec.banco_destino = rec.banco_destino or banks_found[-1]

    # Santander geralmente origem = Santander e destino = BB nos seus comprovantes
    # reforço: se "BCO DO BRASIL" aparecer e destino vazio, preenche destino
    if not rec.banco_destino and "BCO DO BRASIL" in full.upper():
        rec.banco_destino = "BCO DO BRASIL S.A."

    # Limpeza final de nomes
    rec.pagador = _clean_person_name(rec.pagador)
    rec.recebedor = _clean_person_name(rec.recebedor)

    # Ajuste: se recebedor veio cortado tipo "Arlight Importacao e Exportacao, Lo"
    if rec.recebedor and "arligh" in rec.recebedor.lower():
        # tenta achar linha maior contendo ARLIGHET
        for l in lines:
            if "ARLIGHET" in l.upper() and len(l) > len(rec.recebedor):
                rec.recebedor = _clean_person_name(l) or rec.recebedor
                break

    return rec


# =========================
# STREAMLIT APP
# =========================
st.set_page_config(page_title="Extrator de Comprovantes Pix", layout="wide")
st.title("Extrator de Comprovantes Pix (PDF + Imagem) — Opção 1 + 2")

with st.expander("Configurações", expanded=False):
    debug_ocr = st.checkbox("Mostrar DEBUG OCR (texto bruto por arquivo)", value=False)

uploaded = st.file_uploader(
    "Envie comprovantes (PDF/JPG/PNG) ou um ZIP com vários arquivos",
    type=["pdf", "png", "jpg", "jpeg", "zip"],
    accept_multiple_files=True
)

def _iter_files(files) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    for f in files:
        name = f.name
        data = f.read()

        if name.lower().endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(data))
            for info in z.infolist():
                if info.is_dir():
                    continue
                fn = info.filename
                if fn.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                    out.append((fn, z.read(info)))
        else:
            out.append((name, data))
    return out


if uploaded:
    all_files = _iter_files(uploaded)
    st.write(f"Arquivos encontrados: **{len(all_files)}**")

    if st.button("Processar arquivos", type="primary"):
        records: List[PixRecord] = []
        prog = st.progress(0.0)

        for i, (fname, data) in enumerate(all_files, start=1):
            try:
                if fname.lower().endswith(".pdf"):
                    txt = extract_text_from_pdf_bytes(data)
                    fonte = "pdf_text" if txt.strip() else "pdf_text_vazio"
                else:
                    txt = extract_text_from_image_bytes(data)
                    fonte = "ocr_img"

                rec = parse_pix_text(txt, fname)
                rec.fonte_texto = fonte
                records.append(rec)

                if debug_ocr:
                    with st.expander(f"DEBUG OCR - {fname}", expanded=False):
                        st.text((txt or "")[:15000])

            except Exception as e:
                st.warning(f"Falha ao processar {fname}: {e}")
                records.append(PixRecord(arquivo=fname))

            prog.progress(i / len(all_files))

        df = pd.DataFrame([asdict(r) for r in records])

        st.subheader("Resultado")
        st.dataframe(df, use_container_width=True)

        # CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV", data=csv_bytes, file_name="comprovantes_pix.csv", mime="text/csv")

        # Excel
        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Base")

        st.download_button(
            "Baixar Excel",
            data=out_xlsx.getvalue(),
            file_name="comprovantes_pix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Envie PDFs/imagens ou um ZIP para começar.")
