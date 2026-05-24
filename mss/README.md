# Tiny Hybrid MSS

Projeto PyTorch para treinar um separador de fontes musicais pequeno, hibrido e pratico no Kaggle com GPU T4 x2.

O alvo deste projeto nao e competir por "forca bruta" com Demucs v4/BS-RoFormer enormes. A ideia e uma arquitetura pequena, na faixa de 5 a 8M de parametros, que use os pontos que mais importam para reduzir artefatos:

- mascara espectral complexa, para nao depender apenas da fase da mistura;
- atencao axial leve no gargalo espectral;
- condicionamento por contexto temporal da waveform;
- refinamento temporal com convolucoes dilatadas;
- loss combinada no dominio do tempo e em STFT multi-escala;
- augmentations fortes e source remix;
- inferencia por chunks com overlap-add e janela Hann.

## Estrutura

```text
configs/kaggle_t4x2.yaml  Config principal para Kaggle
scripts/train.py          Treino DDP/AMP, validacao, checkpoint e EMA
scripts/separate.py       Inferencia em musica completa com overlap-add
scripts/inspect_model.py  Contagem de parametros e teste rapido
scripts/serve_web.py      Console web local para upload, inferencia e logs
src/tinymss/              Dataset, modelo, losses e utilitarios
webapp/                   Backend FastAPI e interface do separador
```

## Dataset esperado

Use MUSDB18-HQ em WAV com a estrutura padrao:

```text
MUSDB18-HQ/
  train/
    Nome da Musica/
      mixture.wav
      vocals.wav
      drums.wav
      bass.wav
      other.wav
  test/
    Nome da Musica/
      mixture.wav
      vocals.wav
      drums.wav
      bass.wav
      other.wav
```

No Kaggle, adicione um dataset MUSDB18-HQ ao notebook e ajuste `data.root` no YAML ou passe `--data`.

## Rodando no Kaggle T4 x2

No notebook:

```bash
pip install -r requirements-kaggle.txt
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/kaggle_t4x2.yaml \
  --data /kaggle/input/musdb18-hq/MUSDB18-HQ \
  --out /kaggle/working/runs/tiny-hybrid
```

Para a versao v2, depois de subir este projeto para um repo GitHub com a pasta `mss/`, rode:

```bash
REPO_URL="https://github.com/TsXK-shift/model-pt.git" bash scripts/kaggle_setup_v2.sh
bash /kaggle/working/mss/scripts/kaggle_train_v2.sh
```

Para continuar de um checkpoint:

```bash
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/kaggle_t4x2.yaml \
  --data /kaggle/input/musdb18-hq/MUSDB18-HQ \
  --out /kaggle/working/runs/tiny-hybrid \
  --resume /kaggle/working/runs/tiny-hybrid/checkpoints/last.pt
```

## Separando uma musica

```bash
python scripts/separate.py \
  --checkpoint /kaggle/working/runs/tiny-hybrid/checkpoints/best.pt \
  --input /kaggle/input/minha-musica/audio.wav \
  --out /kaggle/working/separated \
  --segment-seconds 8 \
  --overlap 0.5
```

## Usando pelo site

```bash
pip install -r requirements-web.txt
python scripts/serve_web.py --host 0.0.0.0 --port 7860
```

Abra `http://127.0.0.1:7860`. O site procura automaticamente checkpoints em `export_checkpoint/` e `runs/tiny-hybrid/checkpoints/`.

## Expectativa realista

Um modelo de 5 a 8M pode ficar surpreendentemente bom se treinado com disciplina, mas nao existe milagre: MUSDB18-HQ tem so 150 musicas, entao early stopping, augmentations e validacao importam tanto quanto a arquitetura. Para qualidade profissional consistente, o caminho natural depois deste baseline e:

1. treinar por stem ou fazer fine-tuning por stem;
2. adicionar dados extras legalmente licenciados;
3. testar ensembles pequenos;
4. calibrar pesos de loss por fonte.

## Notas v2

Esta versao deve ser treinada do zero. Ela desativa `mix_consistency` durante o treino, corrige a limitacao da mascara complexa, adiciona `ratio_mask` na loss para reduzir copia da mistura em todos os stems, corrige o acumulador de `train_loss` e usa EMA na inferencia quando o checkpoint tiver EMA.
