# AutoShorts (Corte Automático)

Pega um vídeo de gameplay de terror com facecam (link do YouTube ou arquivo local), transcreve e
analisa o áudio, detecta os melhores momentos (sustos, reações engraçadas, frases marcantes) e
monta automaticamente compilações verticais (9:16) prontas pra YouTube Shorts — com tela de
encerramento no estilo *future funk* no final de cada vídeo.

## Como funciona

1. **Baixa o vídeo** (yt-dlp) ou usa um arquivo local já na pasta `entrada/`.
2. **Extrai o áudio** e detecta picos (sons altos que rompem um momento de silêncio — o padrão
   clássico de jump scare).
3. **Transcreve** o áudio inteiro com Whisper (via Groq, gratuito).
4. **Detecta highlights** com um LLM (Groq) lendo a transcrição, categorizando cada momento como
   `susto`, `engracado` ou `frase`.
5. No **modo completo**, também analisa a região da facecam procurando reações físicas fortes
   (movimento brusco) que não tiveram um pico de áudio correspondente.
6. Junta tudo em **beats** (momentos candidatos) e pede pro LLM montar algumas **histórias**
   (compilações de 60-180s, dentro do limite de duração de um Short) escolhendo os melhores beats
   — com nota de 1 a 10 avaliando a força de cada compilação, descartando as fracas.
7. Renderiza cada história em layout vertical (facecam em cima, gameplay embaixo) e concatena a
   tela de encerramento no final.

## Requisitos

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) instalado e no PATH (ou use o instalador, que já
  vem com ffmpeg embutido)
- Uma chave de API gratuita da [Groq](https://console.groq.com/keys)

## Instalação (rodando a partir do código-fonte)

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto com sua chave da Groq:

```
GROQ_API_KEY=sua_chave_aqui
```

## Uso

```bash
# Abre uma janela pra colar o link do YouTube (ou escolher um arquivo local) e o modo de análise
python auto_shorts.py

# Ou direto por linha de comando
python auto_shorts.py "https://youtube.com/watch?v=..." --mode completo
```

Na primeira vez, use `--select-crop` pra calibrar visualmente as regiões da facecam e do gameplay
(arrastando retângulos na prévia do vídeo) — isso fica salvo em `config.json` e é reaproveitado
nas próximas rodadas.

```bash
python auto_shorts.py "entrada/meu_video.mp4" --select-crop
```

### Modos de análise

- **`audio`** (padrão): detecção só por áudio + transcrição. Mais rápido.
- **`completo`**: inclui também detecção de reação visual na facecam. Mais lento, pode achar
  sustos "silenciosos" que não têm pico de áudio, mas também é mais suscetível a falsos positivos
  (movimento de câmera, gesticulação).

### Principais argumentos

| Argumento | Descrição |
|---|---|
| `--mode {audio,completo}` | Escolhe o modo de análise |
| `--num-stories N` | Teto de quantas compilações gerar (a IA pode gerar menos se não achar conteúdo forte o bastante) |
| `--story-min` / `--story-max` | Duração mínima/máxima de cada compilação, em segundos |
| `--select-crop` | Abre o seletor visual de recorte (facecam/gameplay) |
| `--preview` | Só gera uma imagem de prévia dos recortes configurados, sem processar o vídeo |
| `--keep-temp` | Não apaga os arquivos temporários ao final (útil pra depurar) |

Rode `python auto_shorts.py --help` pra ver todas as opções.

## Configuração (`config.json`)

- `facecam_rect` / `gameplay_rect` / `split_ratio` — recortes calibrados via `--select-crop`
- `keywords_susto` / `keywords_vitoria` / `keywords_engracado` — palavras-chave que reforçam a
  detecção de highlights na transcrição
- `outro` — tela de encerramento: texto, duração, caminho do personagem (`assets/personagem.png`
  por padrão) e um `enabled: false` pra desativar

## Gerando o executável / instalador

O projeto pode ser empacotado como um `.exe` standalone (PyInstaller) e um instalador Windows
(Inno Setup), pra rodar sem precisar instalar Python:

```bash
pip install pyinstaller
pyinstaller --onefile --name AutoShorts --distpath "." --workpath build --specpath . ^
  --collect-all yt_dlp --collect-all cv2 --collect-all groq --collect-all soundfile ^
  auto_shorts.py
```

Depois, coloque `ffmpeg.exe` e `ffprobe.exe` numa pasta `bin/` ao lado do `.exe` (o script
procura ali automaticamente) e compile `AutoShorts.iss` com o [Inno Setup](https://jrsoftware.org/isinfo.php)
pra gerar o instalador final.

> O `.exe` e o instalador não ficam neste repositório (arquivos grandes demais pro Git normal) —
> baixe a versão mais recente na aba [Releases](https://github.com/ricardothezouro-debug/AutoShorts/releases).

## Estrutura do projeto

```
auto_shorts.py       # script principal (todo o pipeline)
config.json           # configuração (recortes, keywords, encerramento)
assets/personagem.png # personagem usado na tela de encerramento
entrada/               # vídeos baixados/locais
saida/                  # vídeos gerados (uma subpasta por rodada)
```
