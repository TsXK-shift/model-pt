# Tiny Mask MSS v3

Projeto PyTorch para treinar um separador pequeno de fontes musicais no Kaggle T4 x2.

Esta v3 troca o desenho anterior por uma arquitetura de mascara competitiva. O modelo nao gera quatro waveforms livres. Ele prediz mascaras `softmax` em STFT para `vocals`, `drums`, `bass` e `other`; assim, cada bin tempo-frequencia e dividido entre as fontes, reduzindo muito o atalho ruim de copiar a mistura inteira em todos os stems.

## Ideia principal

- STFT stereo com features de fase comprimida, log magnitude e mid/side.
- U-Net 2D leve com blocos depthwise-separable e dilatacao progressiva.
- contexto da waveform bruta por FiLM no gargalo;
- 2 blocos de atencao axial no gargalo;
- mascaras nao negativas com `softmax` entre fontes;
- reconstrucao por ISTFT com fase da mistura;
- loss focada em magnitude multi-escala e `ratio_mask`.

O modelo padrao tem cerca de 8.69M parametros.

## Estrutura

```text
configs/kaggle_t4x2.yaml       Config principal para Kaggle
scripts/train_watch.py         Treino DDP/AMP com log limpo
scripts/kaggle_setup_v3.sh     Baixa projeto + MUSDB18-HQ em /tmp
scripts/kaggle_train_v3.sh     Treina ou continua treino v3
scripts/kaggle_export_checkpoint_v3.sh  Exporta checkpoint para zip
scripts/separate.py            Inferencia com overlap-add
scripts/serve_web.py           Console web local/Colab
src/tinymss/                   Dataset, modelo, losses e utilitarios
webapp/                        Backend FastAPI e interface
```

## Kaggle

Depois de subir `tiny-mask-mss-v3-clean.zip` para o GitHub:

```bash
%%bash
set -e

MSS_ZIP_URL="https://raw.githubusercontent.com/TsXK-shift/model-pt/main/tiny-mask-mss-v3-clean.zip" \
bash <(wget -qO- https://raw.githubusercontent.com/TsXK-shift/model-pt/main/mss/scripts/kaggle_setup_v3.sh)
```

Se preferir colar sem depender do script remoto, use os comandos que estao em `scripts/kaggle_setup_v3.sh`.

Treinar do zero:

```bash
%%bash
set -e

bash /kaggle/working/mss/scripts/kaggle_train_v3.sh
```

Continuar de checkpoint:

```bash
%%bash
set -e

RESUME="/kaggle/working/runs/tiny-mask-v3/checkpoints/last.pt" \
bash /kaggle/working/mss/scripts/kaggle_train_v3.sh
```

Exportar backup:

```bash
%%bash
set -e

bash /kaggle/working/mss/scripts/kaggle_export_checkpoint_v3.sh
```

## Separar audio

```bash
python scripts/separate.py \
  --checkpoint /kaggle/working/runs/tiny-mask-v3/checkpoints/best.pt \
  --input /kaggle/input/minha-musica/audio.wav \
  --out /kaggle/working/separated \
  --segment-seconds 8 \
  --overlap 0.5
```

## Site

```bash
pip install -r requirements-web.txt
python scripts/serve_web.py --host 0.0.0.0 --port 7860
```

O site procura checkpoints em `export_checkpoint-v3/`, `runs/tiny-mask-v3/checkpoints/` e tambem aceita `TINYMSS_CHECKPOINT_DIRS` apontando para uma pasta exata.

## Expectativa realista

Esta arquitetura deve corrigir o erro estrutural de stems quase iguais a mistura. Ainda assim, qualidade profissional com 5-10M parametros e MUSDB18-HQ sozinho nao e garantida no comeco do treino. O sinal bom para acompanhar e: `valid_loss` caindo e `valid_si_sdr` ficando menos negativo a cada bloco de epocas.
