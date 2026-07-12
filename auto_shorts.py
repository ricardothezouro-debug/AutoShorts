"""Corte Automático — gera candidatos a Shorts a partir de um vídeo do YouTube.

Pipeline: baixa o vídeo -> extrai áudio -> detecta picos de áudio -> transcreve com Groq
Whisper -> detecta highlights com Groq LLM (lendo o transcrito) -> monta histórias de
"melhores momentos" -> corta e concatena os trechos em layout vertical 9:16.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# Empacotado com PyInstaller, __file__ aponta pra dentro da pasta temporária de extração
# (_MEIPASS), não pra pasta onde o .exe realmente está — por isso usa sys.executable nesse caso,
# senão config.json/entrada/saida/assets iriam parar num lugar temporário que some a cada execução.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

ENTRADA_DIR = BASE_DIR / "entrada"
SAIDA_DIR = BASE_DIR / "saida"
TEMP_DIR = BASE_DIR / "temp"
CONFIG_PATH = BASE_DIR / "config.json"

# Deixa o ffmpeg/ffprobe que vier junto do .exe (pasta bin/) visível pro subprocess, sem
# depender do usuário ter ffmpeg instalado/no PATH do sistema.
_BIN_DIR = BASE_DIR / "bin"
if _BIN_DIR.is_dir():
    os.environ["PATH"] = str(_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")

DEFAULT_TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SECONDS = 600  # 10 min — mantém cada upload bem abaixo do limite de tamanho da API do Groq

# "Terror" é o perfil original e não muda em nada (mesmo prompt, mesma lógica de sempre) — os
# outros gêneros reaproveitam a mesma arquitetura (picos de áudio/facecam + keywords + LLM lendo
# transcrição -> beats -> histórias curadas), só trocando o "tempero": como o prompt descreve o
# conteúdo, quais categorias de highlight existem, e qual delas vira o fallback de um pico forte
# sem nenhuma outra confirmação (equivalente ao "susto" do terror).
GENRE_PROFILES = {
    "terror": {
        "label": "Terror (padrão)",
        "fallback_categoria": "susto",
        "has_contexto": False,
    },
    "reacao": {
        "label": "Reação",
        "context_desc": "vídeo de reação, onde o criador assiste e comenta algum conteúdo",
        "categorias": {
            "reacao_forte": "reação física ou verbal forte e genuína (surpresa, choque, gargalhada, indignação) diante do que está sendo assistido",
            "comentario_engracado": "comentário ou tirada engraçada do criador, com graça de verdade — não qualquer piada morna",
            "opiniao": "opinião ou análise forte sobre o conteúdo assistido, que funcionaria como citação fora de contexto",
        },
        "fallback_categoria": "reacao_forte",
        "fallback_label": "reação forte",
        "has_contexto": False,
        "priority_text": (
            'IMPORTANTE: priorize incluir o MAIOR número possível dos beats de categoria '
            '"reacao_forte" — não deixe nenhum de fora a menos que seja necessário pra não '
            'estourar a duração. Beats "comentario_engracado" ou "opiniao" só devem entrar se '
            "genuinamente forem fortes o bastante pra segurar um Short sozinhos — prefira uma "
            "compilação mais curta (mas ainda dentro do mínimo) a encher com conteúdo fraco só "
            "pra bater a duração alvo."
        ),
        "storytelling_text": None,
    },
    "generico": {
        "label": "Gameplay Genérico",
        "context_desc": "vídeo de gameplay em geral, sem gênero específico (não é necessariamente terror)",
        "categorias": {
            "destaque": "um momento de destaque no gameplay — jogada impressionante, virada, momento tenso ou marcante, seja qual for o motivo",
            "engracado": "piada com graça de verdade ou reação cômica clara e inesperada — não qualquer comentário casual",
            "frase": "fala isolada que já faria sentido fora de contexto como citação forte",
            "contexto": "o criador explicando do que se trata o vídeo ou um segmento dele (ex: 'hoje eu vou...', 'nesse vídeo...') — importante pro Short fazer sentido sozinho, mesmo sem ser empolgante",
        },
        "fallback_categoria": "destaque",
        "fallback_label": "destaque",
        "has_contexto": True,
        "priority_text": (
            'IMPORTANTE: priorize incluir os beats de categoria "destaque" mais fortes — não '
            "precisa usar todos, prefira qualidade. Beats \"engracado\" ou \"frase\" só devem "
            "entrar se genuinamente forem fortes o bastante pra segurar um Short sozinhos."
        ),
        "storytelling_text": textwrap.dedent("""
            Storytelling importa muito aqui: um Short feito de picos isolados sem conexão não
            funciona tão bem quanto um que conta uma mini-história coerente. Sempre que existir um
            beat de categoria "contexto" (o criador explicando do que se trata o vídeo ou o
            segmento), priorize abrir a compilação com ele, mesmo que cronologicamente distante dos
            outros beats escolhidos — isso ajuda quem não viu o vídeo completo a entender do que se
            trata o Short. Prefira montar sequências de beats que juntos façam sentido como um
            mini-arco (começo, desenvolvimento, desfecho), não só uma lista de momentos aleatórios.
        """).strip(),
    },
    "platina": {
        "label": "Platina (documentário)",
        "context_desc": (
            "documentário longo sobre platinar um jogo — cobre a história do jogo, a experiência "
            "pessoal do criador tentando platinar, e tem edição com humor e narração"
        ),
        "categorias": {
            "conquista": "um marco real de progresso na platina — troféu difícil conquistado, virada de jogo, alívio depois de uma dificuldade",
            "historia": "um trecho interessante sobre a história/lore do jogo, contado ou comentado pelo criador",
            "engracado": "piada, comentário ou reação genuinamente engraçada — não qualquer fala casual",
            "dificuldade": "um momento de frustração/raiva real com a dificuldade de platinar algo — mostra a luta genuína, não uma reclamação qualquer",
            "contexto": "o criador explicando do que se trata o vídeo ou de que jogo/parte da saga está falando — importante pro Short fazer sentido sozinho, mesmo sem ser empolgante",
        },
        "fallback_categoria": "conquista",
        "fallback_label": "conquista",
        "has_contexto": True,
        "priority_text": (
            "IMPORTANTE: aqui NÃO é sobre maximizar quantidade de uma categoria só — o objetivo é "
            "uma mini-história coerente misturando categorias diferentes (ex: um pouco de história "
            "do jogo, a dificuldade enfrentada, e a conquista/alívio no final). Beats fracos ou "
            "repetitivos não devem entrar só pra bater a duração alvo."
        ),
        "storytelling_text": textwrap.dedent("""
            Storytelling é o coração desse gênero: o vídeo original é um documentário narrado, então
            os Shorts precisam fazer sentido como uma mini-história, não uma sequência de cortes
            soltos. Sempre que existir um beat de categoria "contexto" (o criador explicando do que
            se trata o vídeo/segmento, ex: qual jogo, qual parte da saga), priorize abrir a
            compilação com ele, mesmo que cronologicamente distante dos outros beats escolhidos —
            sem esse contexto, quem não viu o vídeo completo não entende do que se trata. Prefira
            uma sequência que tenha um arco (contexto/setup -> desenvolvimento/dificuldade ->
            desfecho/conquista) a só empilhar momentos engraçados sem conexão entre si.
        """).strip(),
    },
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    """Um 'beat': um momento curto de destaque, usado como matéria-prima para montar histórias."""
    start: float
    end: float
    reasons: list = field(default_factory=list)
    label: str = "momento"
    label_priority: int = 0
    categoria: str = "?"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERRO: ffmpeg não encontrado no PATH.\n"
            "Instale o ffmpeg (https://ffmpeg.org/download.html) e garanta que o comando "
            "'ffmpeg' funcione no terminal antes de rodar este script."
        )


def get_groq_client():
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit(
            "ERRO: variável de ambiente GROQ_API_KEY não encontrada.\n"
            "Crie um arquivo .env na pasta do projeto com a linha:\n"
            "GROQ_API_KEY=sua_chave_aqui"
        )
    return Groq(api_key=api_key)


def is_youtube_url(s: str) -> bool:
    return bool(re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", s.strip()))


def download_youtube(url: str, dest_dir: Path, status=None) -> Path:
    import yt_dlp

    def hook(d):
        if status is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                pct = downloaded / total * 100
                status.update("Baixando vídeo...", f"{pct:.0f}%", fraction=0.25 * (pct / 100))
            else:
                status.update("Baixando vídeo...", "", fraction=None)
        elif d.get("status") == "finished":
            status.update("Baixando vídeo...", "processando...", fraction=0.25)

    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        path = Path(filename)
        if path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        if not path.exists():
            raise FileNotFoundError(f"Não foi possível localizar o arquivo baixado: {filename}")
        return path


def extract_audio(video_path: Path, wav_out: Path) -> None:
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(wav_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ERRO ao extrair áudio com ffmpeg:\n{result.stderr[-2000:]}")


def _iter_chunks(total_duration: float, chunk_seconds: float):
    t = 0.0
    while t < total_duration:
        yield t, min(chunk_seconds, total_duration - t)
        t += chunk_seconds


def _get_audio_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def _rolling_delta(values: np.ndarray, window_s: float, baseline_window_s: float) -> np.ndarray:
    """Para cada janela, calcula o quanto ela se destaca da média móvel causal (só passado)."""
    n = len(values)
    baseline_windows = max(1, int(baseline_window_s / window_s))
    deltas = np.empty(n)
    for i in range(n):
        start_idx = max(0, i - baseline_windows)
        baseline = np.mean(values[start_idx:i]) if i > start_idx else values[i]
        deltas[i] = values[i] - baseline
    return deltas


def _pick_peaks_nms(deltas: np.ndarray, window_s: float, threshold: float, min_distance_s: float) -> list:
    """Supressão não-máxima: pega o pico mais forte primeiro, descarta qualquer outro candidato
    a menos de min_distance_s dele, repete com o próximo mais forte restante. Evita que um pico
    fraco "trave" o debounce e mascare um pico real mais forte por perto, e evita que uma região
    densa vire um único cluster gigante."""
    above = [i for i in range(len(deltas)) if deltas[i] >= threshold]
    above_sorted = sorted(above, key=lambda i: deltas[i], reverse=True)
    picked_times = []
    peaks = []
    for i in above_sorted:
        t = i * window_s
        if all(abs(t - pt) >= min_distance_s for pt in picked_times):
            picked_times.append(t)
            peaks.append((t, float(deltas[i])))
    peaks.sort(key=lambda x: x[0])
    return peaks


def _pre_event_dip(deltas: np.ndarray, window_s: float, t: float, lookback_s: float = 3.0) -> float:
    """Mede se houve uma 'calmaria' (queda abaixo do normal) pouco antes do instante t. Um jump
    scare de jogo de terror classicamente vem depois de um instante de quase-silêncio/quietude —
    diferente de um pico isolado dentro de uma região já barulhenta/agitada (fala contínua, efeito
    de UI, gesticulação), que não costuma ter esse "respiro" logo antes. Reaproveita o próprio
    array de deltas (já subtraído da média móvel), então funciona igual pra áudio (dB) e pra
    movimento de facecam (desvios-padrão) sem precisar dos valores brutos de novo."""
    i = int(round(t / window_s))
    lookback_windows = max(1, int(lookback_s / window_s))
    lo = max(0, i - lookback_windows)
    if lo >= i:
        return 0.0
    return -float(np.min(deltas[lo:i]))


def detect_audio_peaks(
    wav_path: Path,
    threshold_db: float,
    min_distance_s: float,
    window_s: float = 0.5,
    baseline_window_s: float = 15.0,
) -> list:
    data, samplerate = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    window_size = max(1, int(window_s * samplerate))
    n_windows = len(data) // window_size
    if n_windows == 0:
        return []

    rms_db = np.empty(n_windows)
    for i in range(n_windows):
        chunk = data[i * window_size:(i + 1) * window_size]
        rms = np.sqrt(np.mean(np.square(chunk)) + 1e-12)
        rms_db[i] = 20 * np.log10(rms + 1e-12)

    deltas = _rolling_delta(rms_db, window_s, baseline_window_s)
    peaks = _pick_peaks_nms(deltas, window_s, threshold_db, min_distance_s)
    return [(t, d, _pre_event_dip(deltas, window_s, t)) for t, d in peaks]


def detect_facecam_motion(
    video_path: Path,
    facecam_rect: dict,
    threshold_sigma: float,
    min_distance_s: float,
    window_s: float = 0.25,
    baseline_window_s: float = 15.0,
) -> list:
    """Detecta picos de movimento/reação na região da facecam (susto, riso, salto) via diferença
    de frame quadro a quadro, num recorte reduzido do vídeo (baixa resolução/fps, sem decodificar
    o vídeo original inteiro em alta qualidade). Serve pra pegar reações fortes que não têm pico
    de áudio junto — susto "mudo", cara de choque sem grito alto no microfone.
    threshold_sigma é em desvios-padrão acima da média móvel (a energia de movimento não tem uma
    escala fixa como dB, então normaliza por desvio-padrão pra o threshold ter sentido em
    qualquer vídeo)."""
    import cv2

    fps = 1.0 / window_s
    crop = f"crop=iw*{facecam_rect['w']}:ih*{facecam_rect['h']}:iw*{facecam_rect['x']}:ih*{facecam_rect['y']}"
    tmp_path = TEMP_DIR / "facecam_motion.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"{crop},scale=96:96,fps={fps},format=gray",
        "-an", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Aviso: falha ao gerar recorte de movimento da facecam ({result.stderr[-500:]}).")
        return []

    cap = cv2.VideoCapture(str(tmp_path))
    prev = None
    energies = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame[:, :, 0].astype(np.float32)
        if prev is not None:
            energies.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    cap.release()
    tmp_path.unlink(missing_ok=True)

    if not energies:
        return []

    deltas = _rolling_delta(np.array(energies), window_s, baseline_window_s)
    std = np.std(deltas) or 1.0
    deltas_sigma = deltas / std
    peaks = _pick_peaks_nms(deltas_sigma, window_s, threshold_sigma, min_distance_s)
    return [(t, d, _pre_event_dip(deltas_sigma, window_s, t)) for t, d in peaks]


def transcribe(wav_path: Path, groq_client, lang: str, chunk_dir: Path, status=None) -> list:
    duration = _get_audio_duration(wav_path)
    chunks = list(_iter_chunks(duration, CHUNK_SECONDS))
    segments = []
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for idx, (chunk_start, chunk_len) in enumerate(chunks):
        if status:
            status.update(
                "Transcrevendo áudio (Groq Whisper)...",
                f"trecho {idx + 1}/{len(chunks)}",
                fraction=0.35 + 0.20 * (idx / len(chunks)),
            )
        chunk_path = chunk_dir / f"chunk_{idx:03d}.mp3"
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", str(chunk_start), "-t", str(chunk_len),
            "-ar", "16000", "-ac", "1", "-b:a", "64k",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"ERRO ao gerar trecho de áudio para transcrição:\n{result.stderr[-2000:]}")

        print(f"  Transcrevendo trecho {idx + 1} ({chunk_start:.0f}s - {chunk_start + chunk_len:.0f}s)...")
        with open(chunk_path, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                file=(chunk_path.name, f.read()),
                model=DEFAULT_TRANSCRIBE_MODEL,
                language=lang,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        raw_segments = getattr(resp, "segments", None)
        if raw_segments is None and isinstance(resp, dict):
            raw_segments = resp.get("segments", [])
        raw_segments = raw_segments or []

        for seg in raw_segments:
            seg_start = seg["start"] if isinstance(seg, dict) else seg.start
            seg_end = seg["end"] if isinstance(seg, dict) else seg.end
            seg_text = seg["text"] if isinstance(seg, dict) else seg.text
            segments.append(Segment(
                start=chunk_start + seg_start,
                end=chunk_start + seg_end,
                text=seg_text.strip(),
            ))

    return segments


def find_keyword_hits(segments: list, keywords: list) -> list:
    hits = []
    lowered_keywords = [k.lower() for k in keywords if k.strip()]
    for seg in segments:
        text_lower = seg.text.lower()
        for kw in lowered_keywords:
            if kw in text_lower:
                hits.append((seg.start, seg.end, f"keyword:{kw}"))
                break
    return hits


def detect_highlights_llm(segments: list, groq_client, llm_model: str, genero: str = "terror", status=None) -> list:
    if not segments:
        return []

    highlights = []
    seg_by_chunk = {}
    for seg in segments:
        chunk_idx = int(seg.start // CHUNK_SECONDS)
        seg_by_chunk.setdefault(chunk_idx, []).append(seg)

    chunk_indices = sorted(seg_by_chunk)
    for pos, chunk_idx in enumerate(chunk_indices):
        if status:
            status.update(
                "Detectando highlights com IA (Groq)...",
                f"trecho {pos + 1}/{len(chunk_indices)}",
                fraction=0.55 + 0.15 * (pos / len(chunk_indices)),
            )
        chunk_segments = seg_by_chunk[chunk_idx]
        transcript_text = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in chunk_segments)

        if genero == "terror":
            prompt = textwrap.dedent(f"""
                Você está analisando a transcrição de um trecho de um vídeo de gameplay de terror
                com facecam. Cada linha tem o formato [inicio-fim] fala, em segundos.

                Aponte momentos de destaque:
                - "susto": QUALQUER menção ou indício de um momento de tensão/susto real — grito,
                  sobressalto, exclamação de medo, comentário logo após um susto, mesmo que a fala em
                  si seja curta. Aqui é melhor marcar demais do que de menos — na dúvida, marque como
                  susto.
                - "engracado": só marque se reconhecer uma piada com graça de verdade ou reação cômica
                  clara e inesperada — NÃO qualquer comentário casual do jogador. Na dúvida, não
                  marque.
                - "frase": só marque se a fala isolada já fizesse sentido fora de contexto como uma
                  citação forte (virada de jogo importante, reflexão com profundidade, conquista
                  clara) — NÃO uma descrição comum do que está acontecendo na tela. Na dúvida, não
                  marque.

                Para cada momento, escolha start/end que cubram a duração NATURAL do momento inteiro,
                não apenas 2-3 segundos soltos: inclua o contexto/buildup e o desfecho/reação. Shorts do
                YouTube podem durar de poucos segundos até 3 minutos (180s) — um susto rápido pode ter
                15-20s, já uma reflexão ou história mais longa pode e deve durar bem mais (30s a 3min) se
                o conteúdo sustentar isso. Não encurte artificialmente um momento que precisa de mais tempo
                para fazer sentido.

                Responda em JSON, com este formato exato:
                {{"highlights": [{{"start": <numero>, "end": <numero>, "categoria": "susto|engracado|frase", "titulo": "<3 a 6 palavras descrevendo a cena, ex: reflexao sobre coxinha>", "motivo": "<explicacao curta>"}}]}}

                Se não houver nenhum momento relevante neste trecho, responda {{"highlights": []}}.

                Transcrição:
                {transcript_text}
            """).strip()
        else:
            profile = GENRE_PROFILES[genero]
            categorias_texto = "\n".join(f'- "{cat}": {desc}' for cat, desc in profile["categorias"].items())
            categorias_opcoes = "|".join(profile["categorias"].keys())
            contexto_note = (
                '\n\nA categoria "contexto" é especialmente importante: mesmo sem ser um momento '
                "empolgante, ela ajuda um corte curto a fazer sentido sozinho pra quem não viu o "
                "vídeo completo. Não deixe de marcar esses momentos."
                if profile["has_contexto"] else ""
            )

            prompt = textwrap.dedent(f"""
                Você está analisando a transcrição de um trecho de um {profile["context_desc"]}.
                Cada linha tem o formato [inicio-fim] fala, em segundos.

                Aponte momentos de destaque, usando estas categorias:
                {categorias_texto}

                Na dúvida entre marcar ou não um momento, não marque — só aponte quando o trecho
                realmente se encaixa bem numa das categorias acima.{contexto_note}

                Para cada momento, escolha start/end que cubram a duração NATURAL do momento inteiro,
                não apenas 2-3 segundos soltos: inclua o contexto/buildup e o desfecho/reação. Shorts do
                YouTube podem durar de poucos segundos até 3 minutos (180s) — um momento rápido pode ter
                15-20s, já uma reflexão ou história mais longa pode e deve durar bem mais (30s a 3min) se
                o conteúdo sustentar isso. Não encurte artificialmente um momento que precisa de mais tempo
                para fazer sentido.

                Responda em JSON, com este formato exato:
                {{"highlights": [{{"start": <numero>, "end": <numero>, "categoria": "{categorias_opcoes}", "titulo": "<3 a 6 palavras descrevendo a cena>", "motivo": "<explicacao curta>"}}]}}

                Se não houver nenhum momento relevante neste trecho, responda {{"highlights": []}}.

                Transcrição:
                {transcript_text}
            """).strip()

        try:
            resp = groq_client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            for h in parsed.get("highlights", []):
                highlights.append({
                    "start": float(h["start"]),
                    "end": float(h["end"]),
                    "categoria": h.get("categoria", "?"),
                    "titulo": h.get("titulo", ""),
                    "motivo": h.get("motivo", ""),
                })
        except Exception as e:
            print(f"  Aviso: falha ao analisar trecho {chunk_idx} com o LLM ({e}). Pulando esse trecho.")

    return highlights


def build_beats(
    audio_peaks: list,
    keyword_hits: list,
    llm_highlights: list,
    pre_pad: float,
    post_pad: float,
    audio_standalone_db: float = 20.0,
    min_beat_len: float = 4.0,
    max_beat_len: float = 40.0,
    intro_skip_s: float = 30.0,
    facecam_motion_peaks: list = None,
    facecam_motion_standalone_sigma: float = 4.0,
    pre_silence_db: float = 8.0,
    pre_stillness_sigma: float = 1.0,
    fallback_categoria: str = "susto",
) -> list:
    """Junta os sinais brutos (picos de áudio, keywords, highlights do LLM) em uma lista de
    'beats' — momentos curtos de destaque que depois servem de matéria-prima para as histórias.
    Não força duração mínima/máxima de short aqui; isso é decidido na montagem de cada história.
    fallback_categoria é a categoria "reativa" desse gênero (susto no terror, reação forte no
    gênero reação, etc) — usada quando um pico forte de áudio/facecam vira beat sozinho, sem
    nenhum highlight do LLM/keyword por perto pra dar um rótulo melhor."""
    # Picos de áudio fracos/ambíguos (música, efeito sonoro do jogo) só reforçam um beat que já
    # existe por keyword/LLM. Já um pico bem forte (>= audio_standalone_db) é provavelmente um
    # grito/reação real mesmo sem fala reconhecível, então ele também pode virar beat sozinho.
    # Esses momentos precisam de mais tempo de build-up antes do ponto de impacto pra criar tensão —
    # cortar direto pro grito, sem nada antes, mata o efeito.
    susto_pre_pad = max(pre_pad, 8.0)

    raw_points = []

    for start, end, reason in keyword_hits:
        kw = reason.split(":", 1)[1] if ":" in reason else reason
        raw_points.append((start - pre_pad, end + post_pad, reason, kw, 2, "?"))

    for h in llm_highlights:
        reason = f"llm:{h['categoria']}"
        label = h.get("titulo") or h.get("categoria") or "momento"
        beat_pre_pad = susto_pre_pad if h.get("categoria") == fallback_categoria else pre_pad
        raw_points.append((h["start"] - beat_pre_pad, h["end"] + post_pad, reason, label, 3, h.get("categoria", "?")))

    # Picos fortes dentro da zona de abertura do vídeo (intro, tela de carregamento, jingle)
    # são um sinal muito pouco confiável de susto real — som alto ali é mais provável ser
    # música/efeito de menu do que um jump scare. Só viram beat sozinho se estiverem fora
    # dessa janela; dentro dela, continuam contando só como reforço fraco (weak_peaks).
    # Além disso, um susto de jogo de terror classicamente vem depois de um instante de
    # quase-silêncio (quieto -> susto). Um pico alto sem essa calmaria antes (fala contínua,
    # som de UI, trilha sonora já agitada) tende a ser falso-positivo — vira só reforço fraco,
    # não um beat "susto" sozinho.
    def is_confident_audio(d, dip):
        return d >= audio_standalone_db and dip >= pre_silence_db

    strong_peaks = [(t, d) for t, d, dip in audio_peaks if is_confident_audio(d, dip) and t >= intro_skip_s]
    weak_peaks = [(t, d) for t, d, dip in audio_peaks if not (is_confident_audio(d, dip) and t >= intro_skip_s)]

    for t, _d in strong_peaks:
        raw_points.append((t - susto_pre_pad, t + post_pad, "audio_peak_forte", fallback_categoria, 1, fallback_categoria))

    # Reação forte na facecam (susto/riso visual) sem precisar de pico de áudio junto — mesma
    # lógica forte/fraca do áudio (incluindo a exigência de "quietude antes"), mas numa escala
    # própria (desvios-padrão, não dB).
    facecam_motion_peaks = facecam_motion_peaks or []

    def is_confident_motion(d, dip):
        return d >= facecam_motion_standalone_sigma and dip >= pre_stillness_sigma

    strong_motion = [(t, d) for t, d, dip in facecam_motion_peaks if is_confident_motion(d, dip) and t >= intro_skip_s]
    weak_motion = [(t, d) for t, d, dip in facecam_motion_peaks if not (is_confident_motion(d, dip) and t >= intro_skip_s)]

    for t, _d in strong_motion:
        raw_points.append((t - susto_pre_pad, t + post_pad, "facecam_motion_forte", fallback_categoria, 1, fallback_categoria))

    raw_points = [(max(0.0, s), e, r, lb, p, cat) for s, e, r, lb, p, cat in raw_points if e > s]
    raw_points.sort(key=lambda x: x[0])

    merged = []
    for start, end, reason, label, priority, categoria in raw_points:
        if merged and start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, end)
            if reason not in merged[-1].reasons:
                merged[-1].reasons.append(reason)
            if priority > merged[-1].label_priority:
                merged[-1].label = label
                merged[-1].label_priority = priority
                merged[-1].categoria = categoria
        else:
            merged.append(Candidate(
                start=start, end=end, reasons=[reason],
                label=label, label_priority=priority, categoria=categoria,
            ))

    # O ponto que a LLM/Whisper marca pra um susto costuma ser a FALA de reação ("nossa",
    # grito articulado, etc), que acontece vários segundos DEPOIS do barulho seco do susto em
    # si — já visto casos com 10-14s de atraso. Por isso a janela de busca pra trás é bem maior
    # que o pre_pad normal, senão o pico de áudio real do susto nunca é encontrado.
    reinforce_lookback_s = 20.0

    for weak_source, reason_tag in ((weak_peaks, "audio_peak"), (weak_motion, "facecam_motion")):
        for t, _d in weak_source:
            for c in merged:
                if (c.start - reinforce_lookback_s) <= t <= (c.end + post_pad) and reason_tag not in c.reasons:
                    c.reasons.append(reason_tag)
                    # Não é só uma tag: o momento real do pico precisa entrar no clipe renderizado,
                    # senão o corte só pega o comentário depois do susto, não o susto em si.
                    c.start = min(c.start, max(0.0, t - pre_pad))
                    c.end = max(c.end, t + post_pad)
                    break

    for c in merged:
        if c.end - c.start < min_beat_len:
            extra = (min_beat_len - (c.end - c.start)) / 2
            c.start = max(0.0, c.start - extra)
            c.end = c.end + extra
        # Um highlight do LLM às vezes vem com um intervalo largo demais (ex: cobrindo vários
        # momentos distintos); sem esse teto, um único beat gigante pode sozinho estourar o
        # limite de duração de uma história inteira.
        elif c.end - c.start > max_beat_len:
            c.end = c.start + max_beat_len

    merged.sort(key=lambda c: c.start)
    return merged


def curate_stories(
    beats: list,
    groq_client,
    llm_model: str,
    num_stories: int,
    target_min: float,
    target_max: float,
    genero: str = "terror",
    status=None,
) -> list:
    """Pede pro LLM organizar os beats detectados em algumas histórias de 'melhores momentos',
    cada uma reunindo vários trechos (não necessariamente contínuos) até fechar a duração alvo."""
    if not beats:
        return []

    if status:
        status.update("Selecionando os melhores momentos (IA)...", fraction=0.72)

    # Sinal de movimento na facecam só existe no modo "completo" — no modo áudio/transcrição
    # (já validado e aprovado) nenhum beat nunca tem essas reasons, então o prompt fica idêntico.
    has_facecam_signal = any(r.startswith("facecam_motion") for b in beats for r in b.reasons)

    fallback_label = "susto" if genero == "terror" else GENRE_PROFILES[genero]["fallback_label"]

    def beat_tag(b):
        tags = []
        if "audio_peak_forte" in b.reasons:
            tags.append(f"GRITO FORTE SEM FALA - alta confianca de {fallback_label} real")
        if "facecam_motion_forte" in b.reasons:
            tags.append(f"REACAO VISUAL FORTE NA FACECAM - possivel {fallback_label} real, mas SEM confirmacao sonora")
        return f" [{'; '.join(tags)}]" if tags else ""

    lines = [f"[{i}] {b.start:.1f}-{b.end:.1f} ({b.categoria}): {b.label}{beat_tag(b)}" for i, b in enumerate(beats)]
    beats_text = "\n".join(lines)

    if genero == "terror":
        prompt_main = textwrap.dedent(f"""
            Aqui está a lista de momentos de destaque (beats) detectados em um vídeo de gameplay de
            terror com facecam. Cada linha tem o formato [indice] inicio-fim (categoria): descrição,
            com os tempos em segundos. Beats marcados "GRITO FORTE SEM FALA" vieram de um pico de
            áudio muito forte (bem acima do normal) mesmo sem fala reconhecível — são fortes
            candidatos a susto real e devem ser priorizados, não descartados.

            Monte ATÉ {num_stories} compilações de "melhores momentos" distintas — {num_stories} é um
            teto, não uma obrigação. Se o material só for bom o suficiente para menos compilações (ou
            até só 1), gere menos: não force conteúdo fraco/repetitivo só pra bater a quantidade.

            Cada compilação reúne vários desses beats (não precisam ser contínuos no vídeo original)
            formando um vídeo editado com duração total entre {target_min:.0f}s e {target_max:.0f}s
            (soma da duração de cada beat escolhido, contando repetições).

            Mantenha a ordem cronológica dos beats dentro de cada compilação.

            IMPORTANTE: priorize incluir o MAIOR número possível dos beats de categoria "susto" — não
            deixe nenhum de fora a menos que seja necessário pra não estourar a duração. Beats
            "engracado" ou "frase" só devem entrar se genuinamente forem fortes o bastante pra segurar
            um Short sozinhos — prefira uma compilação mais curta (mas ainda dentro do mínimo) a encher
            com conteúdo fraco só pra bater a duração alvo. Evite repetir o mesmo tipo de piada/susto
            várias vezes seguidas dentro da MESMA compilação. Não repita o mesmo beat dentro da MESMA
            compilação. Não tem problema repetir o mesmo beat em mais de uma compilação diferente se
            isso fizer sentido pra contar a história — use com moderação.
        """).strip()
    else:
        profile = GENRE_PROFILES[genero]
        storytelling_section = f"\n\n{profile['storytelling_text']}" if profile["storytelling_text"] else ""
        prompt_main = textwrap.dedent(f"""
            Aqui está a lista de momentos de destaque (beats) detectados em um {profile["context_desc"]}.
            Cada linha tem o formato [indice] inicio-fim (categoria): descrição, com os tempos em
            segundos. Beats marcados "GRITO FORTE SEM FALA" vieram de um pico de áudio muito forte
            (bem acima do normal) mesmo sem fala reconhecível.

            Monte ATÉ {num_stories} compilações de "melhores momentos" distintas — {num_stories} é um
            teto, não uma obrigação. Se o material só for bom o suficiente para menos compilações (ou
            até só 1), gere menos: não force conteúdo fraco/repetitivo só pra bater a quantidade.

            Cada compilação reúne vários desses beats (não precisam ser contínuos no vídeo original)
            formando um vídeo editado com duração total entre {target_min:.0f}s e {target_max:.0f}s
            (soma da duração de cada beat escolhido, contando repetições).

            Mantenha a ordem cronológica dos beats dentro de cada compilação.

            {profile["priority_text"]} Evite repetir o mesmo tipo de momento várias vezes seguidas
            dentro da MESMA compilação. Não repita o mesmo beat dentro da MESMA compilação. Não tem
            problema repetir o mesmo beat em mais de uma compilação diferente se isso fizer sentido
            pra contar a história — use com moderação.{storytelling_section}
        """).strip()

    facecam_caveat = textwrap.dedent(f"""
        Este vídeo foi analisado no modo completo, que inclui um sinal extra de movimento na
        facecam (reação visual forte: susto, salto, virada de cabeça). Esse sinal é MENOS confiável
        que áudio forte ou fala reconhecida — pode ser uma reação real, mas também pode ser só a
        pessoa se ajeitando, rindo ou mexendo a câmera, sem nenhuma confirmação de que algo
        relevante aconteceu. Um beat com label genérico "{fallback_label}" e SEM nenhuma outra
        confirmação (sem keyword, sem highlight do LLM, sem áudio forte — só "REACAO VISUAL
        FORTE") é incerto. Ao dar a nota "forca": uma compilação cheia desses beats "cegos" e
        sem confirmação NÃO deve ganhar nota alta só por ter muitos trechos marcados assim —
        dê nota alta só se o conteúdo realmente parece forte (áudio confirmado, ou
        título/descrição real do que aconteceu), não pela quantidade de beats.
    """).strip()

    prompt_tail = textwrap.dedent("""
        Para cada compilação, dê também uma nota "forca" de 1 a 10 avaliando o quão bom esse vídeo
        ficaria sozinho como Short: 8-10 = excelente, prende atenção do início ao fim, forte
        candidato a viralizar; 5-7 = ok, mas não empolgante; 1-4 = fraco, conteúdo repetitivo ou
        sem graça, não deveria ir pro ar. Seja crítico e honesto na nota — não infle o número só
        pra justificar a existência do vídeo. É melhor entregar 1 compilação nota 9 do que 3
        compilações onde a segunda e a terceira são nota 4.

        Responda em JSON. O campo "beats" é um array JSON de números inteiros separados por
        vírgula (não uma string, não uma lista de strings), na ordem cronológica em que devem
        aparecer no vídeo final. Por exemplo: "beats": [0, 2, 5]. Formato exato:
        {"historias": [{"titulo": "<titulo curto e chamativo, 3 a 6 palavras>", "beats": [0, 2, 5], "forca": 8}]}
    """).strip()

    prompt_sections = [prompt_main]
    if has_facecam_signal:
        prompt_sections.append(facecam_caveat)
    prompt_sections.append(prompt_tail)
    prompt_sections.append(f"Beats disponíveis:\n{beats_text}")
    prompt = "\n\n".join(prompt_sections)

    try:
        resp = groq_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  Aviso: falha ao montar histórias com o LLM ({e}).")
        parsed = {"historias": []}

    def to_index(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().lstrip("-").isdigit():
            return int(v.strip())
        return None

    def to_score(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 5.0
        return min(10.0, max(1.0, v))

    stories = []
    for h in parsed.get("historias", []):
        idxs = [to_index(i) for i in h.get("beats", [])]
        idxs = [i for i in idxs if i is not None and 0 <= i < len(beats)]
        if not idxs:
            continue
        # Remove repetição do mesmo beat dentro da MESMA história, mantendo a ordem dada
        seen = set()
        idxs = [i for i in idxs if not (i in seen or seen.add(i))]
        stories.append({
            "titulo": h.get("titulo") or "melhores_momentos",
            "beats": [beats[i] for i in idxs],
            "forca": to_score(h.get("forca")),
        })

    if not stories:
        # fallback: se o LLM falhar, agrupa os beats cronologicamente em partes ~iguais
        print("  Sem resposta útil do LLM para as histórias — agrupando beats cronologicamente.")
        chunk_size = max(1, len(beats) // num_stories)
        for i in range(0, len(beats), chunk_size):
            group = beats[i:i + chunk_size]
            if group:
                stories.append({"titulo": "melhores_momentos", "beats": group, "forca": 5.0})

    # A IA às vezes erra a soma mental da duração de muitos beats e estoura o teto (já visto
    # quase o dobro do limite com 13 trechos) — Shorts tem limite real de duração, então isso
    # é reforçado aqui no código, não só pedido no prompt. Mantém a ordem escolhida pela IA e
    # só para de somar trechos quando o próximo estouraria o teto.
    for s in stories:
        total = 0.0
        kept = []
        for b in s["beats"]:
            dur = b.end - b.start
            if kept and total + dur > target_max:
                break
            kept.append(b)
            total += dur
        if len(kept) < len(s["beats"]):
            print(f"  Cortando \"{s['titulo']}\" de {len(s['beats'])} para {len(kept)} trechos (estourava o limite de {target_max:.0f}s).")
        s["beats"] = kept

    stories = stories[:num_stories]

    # A IA já foi instruída a não forçar quantidade, mas serve de segunda camada de garantia:
    # descarta compilações que ela mesma avaliou como fracas, sempre mantendo pelo menos a melhor.
    MIN_STORY_SCORE = 6.0
    strong = [s for s in stories if s["forca"] >= MIN_STORY_SCORE]
    if not strong and stories:
        strong = [max(stories, key=lambda s: s["forca"])]
    dropped = [s for s in stories if s not in strong]
    for s in dropped:
        print(f"  Descartada (nota IA {s['forca']:.0f}/10, abaixo do minimo): {s['titulo']}")

    return strong


def render_vertical_short(
    video_path: Path,
    start: float,
    end: float,
    facecam_rect: dict,
    gameplay_rect: dict,
    out_path: Path,
    split_ratio: float = 0.5,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    def rect_expr(rect: dict) -> str:
        return f"crop=iw*{rect['w']}:ih*{rect['h']}:iw*{rect['x']}:ih*{rect['y']}"

    top_h = int(1920 * split_ratio)
    top_h -= top_h % 2  # libx264 exige altura par
    top_h = max(2, min(top_h, 1918))
    bottom_h = 1920 - top_h

    filter_complex = (
        f"[0:v]{rect_expr(facecam_rect)},scale=1080:{top_h}[face];"
        f"[0:v]{rect_expr(gameplay_rect)},scale=1080:{bottom_h}[game];"
        f"[face][game]vstack=inputs=2[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", str(video_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERRO ao renderizar clipe {out_path.name}:\n{result.stderr[-2000:]}")
        return False
    return True


def render_story_montage(
    video_path: Path,
    beats: list,
    facecam_rect: dict,
    gameplay_rect: dict,
    split_ratio: float,
    out_path: Path,
    parts_dir: Path,
    on_part=None,
    outro_path: Path = None,
) -> bool:
    """Renderiza cada beat separadamente e concatena os pedaços numa única montagem."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []
    for j, b in enumerate(beats):
        if on_part:
            on_part(j, len(beats))
        part_path = parts_dir / f"part_{j:03d}.mp4"
        if not render_vertical_short(video_path, b.start, b.end, facecam_rect, gameplay_rect, part_path, split_ratio):
            return False
        part_paths.append(part_path)

    if not part_paths:
        return False

    if outro_path and outro_path.exists():
        part_paths.append(outro_path)

    if len(part_paths) == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(part_paths[0], out_path)
        return True

    list_path = parts_dir / "concat_list.txt"
    lines = [f"file '{str(p.resolve()).replace(chr(92), '/')}'" for p in part_paths]
    list_path.write_text("\n".join(lines), encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERRO ao concatenar história {out_path.name}:\n{result.stderr[-2000:]}")
        return False
    return True


def render_outro_video(
    out_path: Path,
    character_path: Path,
    text_line1: str,
    text_line2: str,
    duration_s: float = 4.0,
    output_fps: float = 30.0,
    size: tuple = (1080, 1920),
    status=None,
) -> bool:
    """Gera a tela de encerramento (estética 'future funk'/synthwave) com o personagem do canal
    e a chamada pra assistir o vídeo completo, no mesmo formato/codec dos outros clipes pra poder
    ser concatenada com -c copy."""
    if not character_path.exists():
        print(f"  Aviso: personagem não encontrado em {character_path}, pulando tela de encerramento.")
        return False

    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    W, H = size
    # Renderiza os frames em Python num fps fixo e baixo (a animação é lenta, 30fps já fica
    # suave) e deixa o ffmpeg reamostrar pro fps de saída na hora de codificar — reamostragem no
    # ffmpeg é quase instantânea, enquanto gerar cada frame em PIL (com vários desfoques/glow)
    # é caro. Sem isso, um vídeo de origem em 50/60fps fazia essa etapa demorar minutos e parecer
    # travada (o dobro ou mais de frames pra gerar, sem nenhum ganho visual real).
    render_fps = min(output_fps, 30.0)
    n_frames = int(render_fps * duration_s)

    sky_top, sky_mid, horizon_c = (13, 6, 41), (72, 15, 92), (255, 87, 143)
    sun_top, sun_bottom = (255, 221, 89), (255, 42, 133)
    grid_h_color, grid_v_color = (0, 240, 255), (255, 20, 190)
    horizon_y = int(H * 0.50)
    sun_center = (W // 2, int(H * 0.34))
    sun_r = int(W * 0.24)

    def lerp(a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def make_sky():
        col = Image.new("RGB", (1, H))
        for y in range(H):
            if y <= horizon_y:
                t = y / max(1, horizon_y)
                c = lerp(sky_top, sky_mid, t / 0.6) if t < 0.6 else lerp(sky_mid, horizon_c, (t - 0.6) / 0.4)
            else:
                c = (8, 4, 18)
            col.putpixel((0, y), c)
        return col.resize((W, H)).convert("RGBA")

    def draw_sun(base):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bbox = [sun_center[0] - sun_r, sun_center[1] - sun_r, sun_center[0] + sun_r, sun_center[1] + sun_r]
        steps = 120
        for i in range(steps):
            t = i / steps
            y0 = bbox[1] + (bbox[3] - bbox[1]) * t
            y1 = bbox[1] + (bbox[3] - bbox[1]) * (t + 1 / steps)
            color = lerp(sun_top, sun_bottom, t)
            band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(band).ellipse(bbox, fill=(*color, 255))
            strip_mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(strip_mask).rectangle([0, y0, W, y1], fill=255)
            band.putalpha(Image.composite(band.getchannel("A"), Image.new("L", (W, H), 0), strip_mask))
            layer.alpha_composite(band)

        # faixas retrô cortadas na metade de baixo do sol (visual clássico synthwave)
        y = sun_center[1] + 10
        gap = 22
        while y < bbox[3]:
            eraser = Image.new("L", (W, H), 0)
            ImageDraw.Draw(eraser).rectangle([0, y, W, y + 10], fill=255)
            arr = np.array(layer)
            arr[np.array(eraser) > 0, 3] = 0
            layer = Image.fromarray(arr)
            y += 10 + gap
            gap *= 1.18

        glow = layer.filter(ImageFilter.GaussianBlur(30))
        r, g, b, a = glow.split()
        glow = Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * 0.55))))
        base.alpha_composite(glow)
        base.alpha_composite(layer)
        return base

    def draw_grid(base, phase):
        layer_glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer_sharp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dg, ds = ImageDraw.Draw(layer_glow), ImageDraw.Draw(layer_sharp)

        vp = (W / 2, horizon_y)
        n_fan = 13
        spread = W * 1.4
        for i in range(n_fan + 1):
            x_bottom = (W / 2 - spread / 2) + spread * (i / n_fan)
            dg.line([vp, (x_bottom, H)], fill=(*grid_v_color, 90), width=14)
            ds.line([vp, (x_bottom, H)], fill=(*grid_v_color, 210), width=3)

        n_h = 11
        for i in range(1, n_h + 1):
            tt = ((i + phase) % n_h) / n_h
            y = horizon_y + (H - horizon_y) * (tt ** 2.2)
            alpha = int(60 + 195 * tt)
            dg.line([(0, y), (W, y)], fill=(*grid_h_color, alpha), width=10)
            ds.line([(0, y), (W, y)], fill=(*grid_h_color, alpha), width=2)

        base.alpha_composite(layer_glow.filter(ImageFilter.GaussianBlur(14)))
        base.alpha_composite(layer_sharp)
        return base

    def ease_out_back(t):
        c1, c3 = 1.70158, 2.70158
        t = max(0.0, min(1.0, t))
        return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

    def load_font(name, size_):
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size_)

    def draw_text_glow(base, text, font, center_x, y, fill, glow_color, glow_radius):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        bbox = d.textbbox((0, 0), text, font=font)
        x = center_x - (bbox[2] - bbox[0]) / 2 - bbox[0]
        d.text((x, y), text, font=font, fill=(*glow_color, 255))
        base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(glow_radius)))
        layer2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer2).text((x, y), text, font=font, fill=(*fill, 255))
        base.alpha_composite(layer2)

    character = Image.open(character_path).convert("RGBA")
    char_target_w = int(W * 0.57)
    character = character.resize((char_target_w, int(character.height * char_target_w / character.width)), Image.LANCZOS)

    font1 = load_font("arialbd.ttf", int(W * 0.044))
    font2 = load_font("impact.ttf", int(W * 0.085))

    # Céu e sol são sempre iguais em todo frame (só a grade se move e o personagem/texto entram) —
    # renderizar isso de novo em cada frame era o gargalo real (put_pixel linha a linha + composição
    # de ~120 faixas do sol, repetido 100+ vezes). Faz uma vez só e reaproveita uma cópia por frame.
    static_bg = make_sky()
    static_bg = draw_sun(static_bg)

    frames_dir = TEMP_DIR / "outro_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        if status and i % 10 == 0:
            status.update("Gerando tela de encerramento...", f"frame {i + 1}/{n_frames}", fraction=0.79)
        base = static_bg.copy()
        phase = (i / n_frames) * 11 * 2.2
        base = draw_grid(base, phase)

        t_char = i / (render_fps * 0.9)
        scale = max(0.0, ease_out_back(t_char)) if t_char < 1 else 1.0
        if scale > 0:
            cw, ch = int(character.width * scale), int(character.height * scale)
            if cw > 0 and ch > 0:
                char_scaled = character.resize((cw, ch), Image.LANCZOS)
                cy_final = int(H * 0.60)
                base.alpha_composite(char_scaled, (W // 2 - cw // 2, cy_final - ch // 2))

        t_text = (i - render_fps * 1.1) / (render_fps * 0.6)
        if t_text > 0:
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_text_glow(overlay, text_line1, font1, W // 2, int(H * 0.78), (255, 255, 255), (0, 220, 255), 8)
            draw_text_glow(overlay, text_line2, font2, W // 2, int(H * 0.82), (255, 255, 255), (255, 30, 180), 12)
            r, g, b, a = overlay.split()
            overlay = Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * min(1.0, t_text)))))
            base.alpha_composite(overlay)

        base.convert("RGB").save(frames_dir / f"frame_{i:04d}.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(render_fps), "-i", str(frames_dir / "frame_%04d.png"),
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest",
        "-r", str(output_fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if result.returncode != 0:
        print(f"  Aviso: falha ao renderizar tela de encerramento ({result.stderr[-500:]}).")
        return False
    return True


def make_preview(video_path: Path, facecam_rect: dict, gameplay_rect: dict, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = TEMP_DIR / "_preview_frame.png"
    cmd = ["ffmpeg", "-y", "-ss", "5", "-i", str(video_path), "-frames:v", "1", str(frame_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ERRO ao extrair frame de preview:\n{result.stderr[-2000:]}")

    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    def draw_rect(rect: dict, color: str, label: str):
        x0 = rect["x"] * w
        y0 = rect["y"] * h
        x1 = x0 + rect["w"] * w
        y1 = y0 + rect["h"] * h
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 5, y0 + 5), label, fill=color)

    draw_rect(facecam_rect, "red", "facecam")
    draw_rect(gameplay_rect, "lime", "gameplay")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    frame_path.unlink(missing_ok=True)


def _ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _get_video_fps(path: Path, default: float = 30.0) -> float:
    # A tela de encerramento precisa ser gerada no MESMO fps do vídeo de origem — concatenar
    # com -c copy trechos em fps diferentes (ex: 30fps vs 24fps) gera um arquivo que o ffmpeg
    # não acusa erro, mas fica com timestamps inconsistentes (seek/preview quebrado em vários players).
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        num, den = result.stdout.strip().split("/")
        fps = float(num) / float(den)
        return fps if fps > 0 else default
    except (ValueError, ZeroDivisionError):
        return default


def _get_screen_size() -> tuple:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def select_crop_interactive(video_path: Path, config: dict, status=None) -> tuple:
    """Retorna (facecam_rect, gameplay_rect, split_ratio)."""
    import cv2

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = TEMP_DIR / "_select_frame.png"
    duration = _ffprobe_duration(video_path)
    ts = min(max(5.0, duration * 0.15), max(duration - 1, 5.0)) if duration else 5.0

    cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", str(video_path), "-frames:v", "1", str(frame_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ERRO ao extrair frame para seleção:\n{result.stderr[-2000:]}")

    # cv2.imread falha com paths com acentos no Windows; ler via buffer de bytes contorna isso
    frame_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"ERRO: não consegui abrir o frame extraído em {frame_path}")
    orig_h, orig_w = frame.shape[:2]

    screen_w, screen_h = _get_screen_size()
    available_h = int(screen_h * 0.75)
    available_w = int(screen_w * 0.55)
    scale = min(1.0, available_h / orig_h, available_w / orig_w)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    display_frame = cv2.resize(frame, (disp_w, disp_h))

    portrait_h = disp_h
    portrait_w = int(portrait_h * 9 / 16)
    gap = 20
    header_h = 34
    canvas_w = disp_w + gap + portrait_w
    canvas_h = header_h + max(disp_h, portrait_h)

    PINK = (203, 60, 240)
    CYAN = (255, 255, 0)
    WIN = "Corte Automatico - selecione os recortes"

    HANDLE_R = 9

    state = {
        "facecam": None,
        "gameplay": None,
        "split": float(config.get("split_ratio", 0.5)),  # fração do preview 9:16 dedicada à facecam
        "drag_mode": None,     # None | "corner" | "move" | "split"
        "drag_target": None,   # "facecam" | "gameplay"
        "drag_anchor": None,   # "corner": (ax,ay) ponto fixo oposto | "move": (offx,offy,w,h)
    }

    def crop_from_rect(rect_disp):
        x0, y0, x1, y1 = rect_disp
        ox0, oy0 = int(x0 / scale), int(y0 / scale)
        ox1, oy1 = int(x1 / scale), int(y1 / scale)
        ox0, oy0 = max(0, ox0), max(0, oy0)
        ox1, oy1 = min(orig_w, max(ox1, ox0 + 1)), min(orig_h, max(oy1, oy0 + 1))
        return frame[oy0:oy1, ox0:ox1]

    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    def render():
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

        if state["facecam"] is None:
            msg = "Arraste um retangulo sobre a FACECAM"
        elif state["gameplay"] is None:
            msg = "Arraste um retangulo sobre o GAMEPLAY"
        else:
            msg = "Arraste as bolinhas p/ redimensionar, o centro p/ mover | ENTER continua | F/G refaz | ESC cancela"
        cv2.putText(canvas, msg, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        landscape = display_frame.copy()

        def draw_rect(rect, color):
            if rect is None:
                return
            x0, y0, x1, y1 = rect
            cv2.rectangle(landscape, (x0, y0), (x1, y1), color, 2)
            for cx, cy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                cv2.rectangle(landscape, (cx - 5, cy - 5), (cx + 5, cy + 5), (255, 255, 255), -1)
                cv2.rectangle(landscape, (cx - 5, cy - 5), (cx + 5, cy + 5), color, 1)

        draw_rect(state["facecam"], PINK)
        draw_rect(state["gameplay"], CYAN)

        canvas[header_h:header_h + disp_h, 0:disp_w] = landscape

        portrait = np.full((portrait_h, portrait_w, 3), 15, dtype=np.uint8)
        top_h = int(portrait_h * state["split"])
        top_h = clamp(top_h, 1, portrait_h - 1)
        if state["facecam"]:
            face_img = crop_from_rect(state["facecam"])
            if face_img.size > 0:
                portrait[0:top_h, :] = cv2.resize(face_img, (portrait_w, top_h))
        if state["gameplay"]:
            game_img = crop_from_rect(state["gameplay"])
            if game_img.size > 0:
                portrait[top_h:portrait_h, :] = cv2.resize(game_img, (portrait_w, portrait_h - top_h))
        cv2.rectangle(portrait, (0, 0), (portrait_w - 1, portrait_h - 1), (90, 90, 90), 1)

        grip_w, grip_h = 46, 14
        grip_x0 = portrait_w // 2 - grip_w // 2
        grip_y0 = top_h - grip_h // 2
        cv2.rectangle(portrait, (grip_x0, grip_y0), (grip_x0 + grip_w, grip_y0 + grip_h), (235, 235, 235), -1)
        cv2.rectangle(portrait, (grip_x0, grip_y0), (grip_x0 + grip_w, grip_y0 + grip_h), (110, 110, 110), 1)
        for i in range(4):
            cv2.circle(portrait, (grip_x0 + 11 + i * 8, top_h), 2, (110, 110, 110), -1)

        canvas[header_h:header_h + portrait_h, disp_w + gap:disp_w + gap + portrait_w] = portrait

        cv2.imshow(WIN, canvas)

    def find_handle(px, py):
        for name in ("facecam", "gameplay"):
            rect = state[name]
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            for idx, (cx, cy) in enumerate([(x0, y0), (x1, y0), (x0, y1), (x1, y1)]):
                if abs(px - cx) <= HANDLE_R and abs(py - cy) <= HANDLE_R:
                    return name, idx
        return None, None

    def find_body(px, py):
        for name in ("facecam", "gameplay"):
            rect = state[name]
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            if x0 <= px <= x1 and y0 <= py <= y1:
                return name
        return None

    portrait_x0 = disp_w + gap

    def slot_ratio(name):
        top_h = clamp(int(portrait_h * state["split"]), 1, portrait_h - 1)
        slot_h = top_h if name == "facecam" else (portrait_h - top_h)
        return portrait_w / slot_h

    def constrained_rect(anchor, mouse, ratio):
        ax, ay = anchor
        mx, my = mouse
        sign_x = 1 if mx >= ax else -1
        sign_y = 1 if my >= ay else -1
        max_w = (disp_w - ax) if sign_x > 0 else ax
        max_h = (disp_h - ay) if sign_y > 0 else ay
        raw_w = abs(mx - ax)
        raw_h = abs(my - ay)
        w = max(raw_w, raw_h * ratio, 1)
        h = w / ratio
        if w > max_w > 0:
            w = max_w
            h = w / ratio
        if h > max_h > 0:
            h = max_h
            w = h * ratio
        nx = ax + sign_x * w
        ny = ay + sign_y * h
        x0, x1 = sorted((ax, nx))
        y0, y1 = sorted((ay, ny))
        return (int(x0), int(y0), int(x1), int(y1))

    def refit_rect_to_ratio(rect, ratio):
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max(x1 - x0, 1)
        h = w / ratio
        if h > disp_h:
            h = disp_h
            w = h * ratio
        nx0, nx1 = cx - w / 2, cx + w / 2
        ny0, ny1 = cy - h / 2, cy + h / 2
        if nx0 < 0:
            nx1 -= nx0
            nx0 = 0
        if nx1 > disp_w:
            shift = nx1 - disp_w
            nx0 -= shift
            nx1 -= shift
        if ny0 < 0:
            ny1 -= ny0
            ny0 = 0
        if ny1 > disp_h:
            shift = ny1 - disp_h
            ny0 -= shift
            ny1 -= shift
        return (int(nx0), int(ny0), int(nx1), int(ny1))

    def on_mouse(event, x, y, flags, userdata):
        y -= header_h

        # Enquanto um drag já está em andamento, ele tem prioridade sobre a região do clique
        if state["drag_mode"] == "split":
            y = clamp(y, 0, portrait_h - 1)
            if event == cv2.EVENT_MOUSEMOVE:
                state["split"] = clamp(y / portrait_h, 0.15, 0.85)
                state["facecam"] = refit_rect_to_ratio(state["facecam"], slot_ratio("facecam"))
                state["gameplay"] = refit_rect_to_ratio(state["gameplay"], slot_ratio("gameplay"))
                render()
            elif event == cv2.EVENT_LBUTTONUP:
                state["drag_mode"] = None
                render()
            return

        if state["drag_mode"] in ("corner", "move"):
            x = clamp(x, 0, disp_w - 1)
            y = clamp(y, 0, disp_h - 1)
            target = state["drag_target"]
            if event == cv2.EVENT_MOUSEMOVE:
                if state["drag_mode"] == "corner":
                    ax, ay = state["drag_anchor"]
                    state[target] = constrained_rect((ax, ay), (x, y), slot_ratio(target))
                else:
                    offx, offy, w, h = state["drag_anchor"]
                    w = min(w, disp_w)
                    h = min(h, disp_h)
                    nx0 = clamp(x - offx, 0, disp_w - w)
                    ny0 = clamp(y - offy, 0, disp_h - h)
                    state[target] = (nx0, ny0, nx0 + w, ny0 + h)
                render()
            elif event == cv2.EVENT_LBUTTONUP:
                if state["drag_mode"] == "corner":
                    x0, y0, x1, y1 = state[target]
                    if x1 - x0 < 5 or y1 - y0 < 5:
                        state[target] = None
                state["drag_mode"] = None
                state["drag_target"] = None
                state["drag_anchor"] = None
                render()
            return

        # Nenhum drag ativo: só reagimos a clique novo, decidindo a região
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if 0 <= x < disp_w and 0 <= y < disp_h:
            target, corner_idx = find_handle(x, y)
            if target:
                x0, y0, x1, y1 = state[target]
                corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
                anchor = corners[3 - corner_idx]
                state["drag_mode"] = "corner"
                state["drag_target"] = target
                state["drag_anchor"] = anchor
            else:
                body = find_body(x, y)
                if body:
                    x0, y0, x1, y1 = state[body]
                    state["drag_mode"] = "move"
                    state["drag_target"] = body
                    state["drag_anchor"] = (x - x0, y - y0, x1 - x0, y1 - y0)
                elif state["facecam"] is None:
                    state["drag_mode"] = "corner"
                    state["drag_target"] = "facecam"
                    state["drag_anchor"] = (x, y)
                    state["facecam"] = (x, y, x, y)
                elif state["gameplay"] is None:
                    state["drag_mode"] = "corner"
                    state["drag_target"] = "gameplay"
                    state["drag_anchor"] = (x, y)
                    state["gameplay"] = (x, y, x, y)
            render()
        elif portrait_x0 <= x < portrait_x0 + portrait_w and 0 <= y < portrait_h:
            boundary_y = int(portrait_h * state["split"])
            if abs(y - boundary_y) <= HANDLE_R + 4:
                state["drag_mode"] = "split"

    print("\n=== Seleção de recortes ===")
    print("Arraste a FACECAM e o GAMEPLAY. Depois de criados, arraste as bolinhas dos cantos pra")
    print("redimensionar, o centro do retângulo pra mover, ou a alcinha entre os blocos do preview")
    print("à direita pra ajustar a proporção. ENTER confirma, F/G refazem, ESC cancela.\n")

    cv2.namedWindow(WIN)
    cv2.moveWindow(WIN, 20, 20)
    cv2.setMouseCallback(WIN, on_mouse)
    render()

    frame_count = 0
    while True:
        key = cv2.waitKey(20) & 0xFF
        frame_count += 1
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit("Seleção cancelada. Nada foi salvo.")
        elif key in (ord('f'), ord('F')):
            state["facecam"] = None
            render()
        elif key in (ord('g'), ord('G')):
            state["gameplay"] = None
            render()
        elif key in (13, 32):
            if state["facecam"] and state["gameplay"]:
                break
        # Espera a janela "assentar" antes de confiar na checagem de visibilidade
        # (logo após a criação, WND_PROP_VISIBLE pode falsamente indicar "fechada")
        if frame_count > 15 and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            sys.exit("Seleção cancelada. Nada foi salvo.")

    cv2.destroyAllWindows()

    fx0, fy0, fx1, fy1 = state["facecam"]
    gx0, gy0, gx1, gy1 = state["gameplay"]
    facecam_rect = {
        "x": (fx0 / scale) / orig_w, "y": (fy0 / scale) / orig_h,
        "w": ((fx1 - fx0) / scale) / orig_w, "h": ((fy1 - fy0) / scale) / orig_h,
    }
    gameplay_rect = {
        "x": (gx0 / scale) / orig_w, "y": (gy0 / scale) / orig_h,
        "w": ((gx1 - gx0) / scale) / orig_w, "h": ((gy1 - gy0) / scale) / orig_h,
    }
    split_ratio = state["split"]

    from tkinter import messagebox

    own_root = None
    parent = status.root if status is not None else None
    if parent is None:
        import tkinter as tk
        own_root = tk.Tk()
        own_root.withdraw()
        own_root.attributes("-topmost", True)
        parent = own_root

    save = messagebox.askyesno(
        "Corte Automático",
        "Salvar esses recortes (facecam/gameplay/proporção) no config.json?",
        parent=parent,
    )
    if own_root is not None:
        own_root.destroy()

    if save:
        config["facecam_rect"] = facecam_rect
        config["gameplay_rect"] = gameplay_rect
        config["split_ratio"] = split_ratio
        CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Salvo em {CONFIG_PATH}")
    else:
        print("Recortes não salvos no config.json (valem só para esta execução).")

    return facecam_rect, gameplay_rect, split_ratio


class StatusWindow:
    """Janela simples que mostra o progresso do pipeline (baixando, transcrevendo, gerando clipes...)."""

    def __init__(self):
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("Corte Automático")
        self.root.geometry("420x150")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        self.label = tk.Label(self.root, text="Iniciando...", font=("Segoe UI", 11), wraplength=380, justify="left", anchor="w")
        self.label.pack(pady=(20, 4), padx=20, fill="x")
        self.sub_label = tk.Label(self.root, text="", font=("Segoe UI", 9), fg="gray", wraplength=380, justify="left", anchor="w")
        self.sub_label.pack(padx=20, fill="x")
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=380, mode="determinate")
        self.progress.pack(pady=15, padx=20)
        self.button = None
        self._done = False

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._safe(self.root.update)

    def _on_close(self):
        if self._done:
            self._safe(self.root.destroy)
            return
        print("Janela de progresso fechada — encerrando o processo.")
        os._exit(1)

    def _safe(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    def update(self, text: str, sub: str = "", fraction=None):
        def do():
            self.label.config(text=text)
            self.sub_label.config(text=sub)
            if fraction is None:
                self.progress.config(mode="indeterminate")
                self.progress.step(8)
            else:
                self.progress.config(mode="determinate")
                self.progress["value"] = max(0, min(100, fraction * 100))
            self.root.update()
        self._safe(do)

    def finish(self, text: str):
        self._done = True

        def do():
            self.label.config(text=text)
            self.sub_label.config(text="")
            self.progress.config(mode="determinate")
            self.progress["value"] = 100
            if self.button is None:
                self.button = self._tk.Button(self.root, text="Fechar", command=self.root.destroy)
                self.button.pack(pady=(0, 10))
            self.root.update()
        self._safe(do)

    def wait_close(self):
        self._safe(self.root.mainloop)

    def close(self):
        self._safe(self.root.destroy)


def prompt_for_source_gui() -> tuple:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    result = {"value": None}

    root = tk.Tk()
    root.title("Corte Automático")
    root.geometry("480x430")
    root.resizable(False, False)

    tk.Label(root, text="Cole o link do YouTube:", font=("Segoe UI", 10)).pack(pady=(15, 5))
    url_var = tk.StringVar()
    tk.Entry(root, textvariable=url_var, width=55).pack()

    tk.Label(root, text="— ou —", font=("Segoe UI", 9), fg="gray").pack(pady=8)

    file_var = tk.StringVar()
    file_label = tk.Label(root, text="Nenhum arquivo selecionado", fg="gray")
    file_label.pack()

    def choose_file():
        path = filedialog.askopenfilename(
            title="Selecione o vídeo",
            initialdir=str(ENTRADA_DIR),
            filetypes=[("Vídeos", "*.mp4 *.mkv *.mov *.avi *.webm"), ("Todos os arquivos", "*.*")],
        )
        if path:
            file_var.set(path)
            file_label.config(text=Path(path).name, fg="black")

    tk.Button(root, text="Selecionar arquivo de vídeo...", command=choose_file).pack(pady=5)

    tk.Frame(root, height=1, bg="#ddd").pack(fill="x", padx=20, pady=12)

    tk.Label(root, text="Análise:", font=("Segoe UI", 10, "bold")).pack()
    mode_var = tk.StringVar(value="audio")
    tk.Radiobutton(
        root, text="Áudio/Transcrição (mais rápido)",
        variable=mode_var, value="audio",
    ).pack(anchor="w", padx=60)
    tk.Radiobutton(
        root, text="Completo — inclui reações da facecam (mais lento)",
        variable=mode_var, value="completo",
    ).pack(anchor="w", padx=60)

    tk.Frame(root, height=1, bg="#ddd").pack(fill="x", padx=20, pady=12)

    tk.Label(root, text="Gênero do conteúdo:", font=("Segoe UI", 10, "bold")).pack()
    genero_var = tk.StringVar(value="terror")
    genero_labels = {key: profile["label"] for key, profile in GENRE_PROFILES.items()}
    genero_display = tk.StringVar(value=genero_labels["terror"])

    def on_genero_change(*_args):
        display = genero_display.get()
        for key, label in genero_labels.items():
            if label == display:
                genero_var.set(key)
                break

    genero_menu = tk.OptionMenu(root, genero_display, *genero_labels.values(), command=lambda _v: on_genero_change())
    genero_menu.pack(pady=(4, 0))

    def confirm():
        url = url_var.get().strip()
        file_path = file_var.get().strip()
        if url:
            result["value"] = (url, mode_var.get(), genero_var.get())
        elif file_path:
            result["value"] = (file_path, mode_var.get(), genero_var.get())
        else:
            messagebox.showwarning("Corte Automático", "Cole um link do YouTube ou selecione um arquivo de vídeo.")
            return
        root.destroy()

    def cancel():
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=15)
    tk.Button(btn_frame, text="Cancelar", command=cancel, width=12).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Iniciar", command=confirm, width=12, bg="#4CAF50", fg="white").pack(side="left", padx=5)

    root.mainloop()

    if not result["value"]:
        sys.exit("Cancelado pelo usuário.")
    return result["value"]


def format_hms(t: float) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    secs = int(t % 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:40] or "momento"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baixa um vídeo do YouTube e gera candidatos a Shorts (susto/humor/frases) em layout vertical."
    )
    parser.add_argument(
        "url", nargs="?", default=None,
        help="Link do vídeo no YouTube ou caminho de um arquivo local. Se omitido, abre uma janela pra escolher.",
    )
    parser.add_argument("--output-dir", default=str(SAIDA_DIR), help="Pasta base de saída — cada rodada cria sua própria subpasta (timestamp + vídeo + modo) dentro dela")
    parser.add_argument("--num-stories", type=int, default=3, help="Quantas montagens de 'melhores momentos' gerar (teto, não obrigação)")
    parser.add_argument("--story-min", type=float, default=60.0, help="Duração mínima de cada montagem (s)")
    parser.add_argument("--story-max", type=float, default=180.0, help="Duração máxima de cada montagem (s) — 180s é o limite de Shorts do YouTube")
    parser.add_argument("--audio-threshold-db", type=float, default=12.0, help="dB acima da média móvel para considerar pico (fraco, precisa de keyword/LLM por perto pra virar beat)")
    parser.add_argument("--audio-standalone-db", type=float, default=20.0, help="dB acima da média a partir do qual um pico forte vira beat sozinho (provável grito sem fala)")
    parser.add_argument("--pre-silence-db", type=float, default=8.0, help="Quantos dB de 'calmaria' são exigidos nos ~3s antes de um pico de áudio forte pra confiar nele como susto real (evita falso-positivo de som alto em meio a fala/UI já barulhenta)")
    parser.add_argument("--pre-pad", type=float, default=3.0, help="Segundos de padding antes do ponto detectado")
    parser.add_argument("--post-pad", type=float, default=5.0, help="Segundos de padding depois do ponto detectado")
    parser.add_argument(
        "--mode", choices=["audio", "completo"], default="audio",
        help="'audio': só áudio/transcrição. 'completo': inclui detecção de reações visuais na facecam (mais lento)",
    )
    parser.add_argument(
        "--genero", choices=list(GENRE_PROFILES.keys()), default="terror",
        help="Estilo de conteúdo — muda como a IA categoriza e monta as histórias (terror, reacao, generico, platina)",
    )
    parser.add_argument("--facecam-motion-floor-sigma", type=float, default=2.0, help="Desvios-padrão acima do normal para considerar QUALQUER movimento na facecam (fraco, só reforça beat existente)")
    parser.add_argument("--facecam-motion-threshold", type=float, default=4.0, help="Desvios-padrão acima do normal a partir do qual um movimento na facecam vira beat sozinho (reação forte sem áudio)")
    parser.add_argument("--pre-stillness-sigma", type=float, default=1.0, help="Desvios-padrão de 'quietude' exigidos nos ~3s antes de um movimento forte na facecam pra confiar nele como reação real (evita falso-positivo de gesticulação contínua)")
    parser.add_argument("--lang", default="pt", help="Idioma da transcrição")
    parser.add_argument("--extra-keywords", default="", help="Palavras-chave extras separadas por vírgula")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Modelo Groq usado para detectar highlights e montar histórias")
    parser.add_argument("--keep-temp", action="store_true", help="Não apagar arquivos temporários ao final")
    parser.add_argument("--preview", action="store_true", help="Apenas gerar imagem de preview dos crops e sair")
    parser.add_argument("--select-crop", action="store_true", help="Selecionar interativamente (arrastando o mouse) os retângulos de facecam/gameplay e salvar no config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    used_gui = args.url is None
    if used_gui:
        source, mode, genero = prompt_for_source_gui()
    else:
        source, mode, genero = args.url, args.mode, args.genero

    config = load_config()
    facecam_rect = config.get("facecam_rect", {"x": 0.0, "y": 0.72, "w": 0.28, "h": 0.28})
    gameplay_rect = config.get("gameplay_rect", {"x": 0.15, "y": 0.0, "w": 0.70, "h": 1.0})
    split_ratio = float(config.get("split_ratio", 0.5))

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    win = StatusWindow()

    if is_youtube_url(source):
        win.update("Baixando vídeo...", source)
        print(f"Baixando vídeo: {source}")
        try:
            video_path = download_youtube(source, ENTRADA_DIR, status=win)
        except Exception as e:
            win.close()
            sys.exit(f"ERRO ao baixar vídeo do YouTube: {e}")
        print(f"  Vídeo salvo em: {video_path}")
    else:
        video_path = Path(source)
        if not video_path.exists():
            win.close()
            sys.exit(f"ERRO: arquivo de vídeo não encontrado: {video_path}")
        print(f"Usando vídeo local: {video_path}")
        win.update("Vídeo local carregado", video_path.name, fraction=0.25)

    # Cada rodada vai pra sua própria subpasta (timestamp + nome do vídeo + modo) — sem isso,
    # rodar o mesmo vídeo de novo (ou nos dois modos) sobrescreve/mistura arquivos "01", "02"
    # de rodadas diferentes na mesma pasta, sem jeito de saber qual é qual depois.
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_slug = f"{run_stamp}_{slugify(video_path.stem)}_{mode}"
    output_dir = Path(args.output_dir) / run_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.select_crop:
        win.close()
        select_crop_interactive(video_path, config)
        return

    if args.preview:
        win.close()
        preview_path = TEMP_DIR / "preview.png"
        make_preview(video_path, facecam_rect, gameplay_rect, preview_path)
        print(f"Preview salvo em: {preview_path}")
        return

    if used_gui:
        win.update("Aguardando seleção de recortes...", "veja a janela de seleção", fraction=0.27)
        facecam_rect, gameplay_rect, split_ratio = select_crop_interactive(video_path, config, status=win)

    wav_path = TEMP_DIR / "audio.wav"
    win.update("Extraindo áudio...", fraction=0.29)
    print("Extraindo áudio...")
    extract_audio(video_path, wav_path)

    win.update("Detectando picos de áudio...", fraction=0.32)
    print("Detectando picos de áudio...")
    audio_peaks = detect_audio_peaks(wav_path, args.audio_threshold_db, min_distance_s=15.0)
    print(f"  {len(audio_peaks)} picos de áudio detectados.")

    facecam_motion_peaks = []
    if mode == "completo":
        win.update("Detectando reações na facecam...", fraction=0.34)
        print("Detectando movimento/reações na facecam...")
        facecam_motion_peaks = detect_facecam_motion(
            video_path, facecam_rect, args.facecam_motion_floor_sigma, min_distance_s=15.0,
        )
        print(f"  {len(facecam_motion_peaks)} picos de movimento na facecam detectados.")

    groq_client = get_groq_client()

    print("Transcrevendo áudio (Groq Whisper)...")
    chunk_dir = TEMP_DIR / "chunks"
    segments = transcribe(wav_path, groq_client, args.lang, chunk_dir, status=win)
    print(f"  {len(segments)} segmentos de fala transcritos.")

    keywords = (
        config.get("keywords_susto", [])
        + config.get("keywords_vitoria", [])
        + config.get("keywords_engracado", [])
        + [k.strip() for k in args.extra_keywords.split(",") if k.strip()]
    )
    keyword_hits = find_keyword_hits(segments, keywords)
    print(f"  {len(keyword_hits)} trechos batendo com palavras-chave.")

    print(f"Detectando highlights com LLM (Groq) — gênero: {GENRE_PROFILES[genero]['label']}...")
    llm_highlights = detect_highlights_llm(segments, groq_client, args.llm_model, genero=genero, status=win)
    print(f"  {len(llm_highlights)} highlights sugeridos pelo LLM.")

    beats = build_beats(
        audio_peaks, keyword_hits, llm_highlights,
        args.pre_pad, args.post_pad,
        audio_standalone_db=args.audio_standalone_db,
        facecam_motion_peaks=facecam_motion_peaks,
        facecam_motion_standalone_sigma=args.facecam_motion_threshold,
        pre_silence_db=args.pre_silence_db,
        pre_stillness_sigma=args.pre_stillness_sigma,
        fallback_categoria=GENRE_PROFILES[genero]["fallback_categoria"],
    )

    if not beats:
        print("Nenhum momento de destaque encontrado neste vídeo. Encerrando sem gerar vídeos.")
        win.finish("Nenhum momento de destaque encontrado neste vídeo.")
        win.wait_close()
        if not args.keep_temp:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    print(f"  {len(beats)} beats encontrados:")
    for i, b in enumerate(beats):
        print(f"    [{i}] {format_hms(b.start)}-{format_hms(b.end)} ({b.categoria}): {b.label} — {', '.join(b.reasons)}")

    print("  Montando histórias...")
    stories = curate_stories(
        beats, groq_client, args.llm_model,
        num_stories=args.num_stories, target_min=args.story_min, target_max=args.story_max,
        genero=genero, status=win,
    )

    if not stories:
        print("Não foi possível montar nenhuma história. Encerrando sem gerar vídeos.")
        win.finish("Não foi possível montar nenhuma história.")
        win.wait_close()
        if not args.keep_temp:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    used_beats = {id(b) for story in stories for b in story["beats"]}
    unused_beats = [b for b in beats if id(b) not in used_beats]
    if unused_beats:
        print(f"  {len(unused_beats)} beats não usados em nenhuma história:")
        for b in unused_beats:
            print(f"    {format_hms(b.start)}-{format_hms(b.end)} ({b.categoria}): {b.label} — {', '.join(b.reasons)}")

    print(f"\n{len(stories)} histórias montadas. Renderizando...\n")
    win.update(f"Renderizando {len(stories)} histórias...", fraction=0.78)

    outro_cfg = config.get("outro", {})
    outro_path = None
    if outro_cfg.get("enabled", True):
        win.update("Gerando tela de encerramento...", fraction=0.79)
        outro_path = TEMP_DIR / "outro.mp4"
        character_path = BASE_DIR / outro_cfg.get("character_path", "assets/personagem.png")
        if not render_outro_video(
            outro_path, character_path,
            outro_cfg.get("text_line1", "Assista o vídeo completo em"),
            outro_cfg.get("text_line2", "youtube.com/@Gamoxkun"),
            duration_s=float(outro_cfg.get("duration_s", 4.0)),
            output_fps=_get_video_fps(video_path),
            status=win,
        ):
            outro_path = None

    report = []
    for i, story in enumerate(stories, start=1):
        story_beats = story["beats"]
        slug = slugify(story["titulo"])
        out_path = output_dir / f"{i:02d}_{slug}.mp4"
        parts_dir = TEMP_DIR / f"story_{i:02d}_parts"
        total_duration = sum(b.end - b.start for b in story_beats)

        def on_part(j, total, i=i, n_stories=len(stories), story=story):
            win.update(
                f"Renderizando história {i}/{n_stories}: {story['titulo']}",
                f"trecho {j + 1}/{total}",
                fraction=0.78 + 0.22 * ((i - 1 + (j + 1) / total) / n_stories),
            )

        ok = render_story_montage(
            video_path, story_beats, facecam_rect, gameplay_rect, split_ratio,
            out_path, parts_dir, on_part=on_part, outro_path=outro_path,
        )

        report.append({
            "arquivo": str(out_path) if ok else "FALHOU",
            "titulo": story["titulo"],
            "duracao": f"{total_duration:.0f}s",
            "trechos": len(story_beats),
            "nota_ia": f"{story['forca']:.0f}/10",
            "beats": ", ".join(f"{format_hms(b.start)}-{format_hms(b.end)} ({b.label})" for b in story_beats),
        })
        status_str = "ok" if ok else "falhou"
        print(f"  [{i}/{len(stories)}] {out_path.name} ({status_str}) — {len(story_beats)} trechos, ~{total_duration:.0f}s, nota IA {story['forca']:.0f}/10")

        if not args.keep_temp:
            shutil.rmtree(parts_dir, ignore_errors=True)

    print("\n=== Relatório final ===")
    for r in report:
        print(f"- {r['arquivo']} | {r['titulo']} | {r['trechos']} trechos, {r['duracao']} | nota IA {r['nota_ia']} | {r['beats']}")

    win.finish(f"Concluído! {len(stories)} histórias geradas em saida/")
    win.wait_close()

    if not args.keep_temp:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
