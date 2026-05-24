# Decisoes de Arquitetura

Este projeto segue uma linha pragmatica: manter o modelo pequeno o bastante para Kaggle T4 x2, mas sem abrir mao dos mecanismos que mais reduzem artefatos em separacao musical.

## Por que hibrido

Separadores puramente espectrais costumam preservar bem regioes harmonicas, mas sofrem com fase e transientes. Separadores puramente temporais conseguem modelar fase, mas precisam de campo receptivo enorme para graves, pratos e reverberacao. Por isso o modelo combina:

- ramo espectral por STFT, estimando mascaras complexas por fonte e canal;
- contexto temporal extraido da waveform e aplicado por FiLM no gargalo espectral;
- refinador temporal TCN com dilatacoes progressivas;
- consistencia de mistura no final, para a soma dos stems voltar para a mistura.

## Modelo

`TinyHybridMSS` tem tres partes:

1. `SpectralUNet`: recebe STFT stereo comprimida, passa por U-Net 2D e prediz mascaras complexas para `vocals`, `drums`, `bass` e `other`.
2. `AxialAttentionBlock`: aplica atencao no tempo e na frequencia no gargalo, evitando o custo de atencao global completa.
3. `TemporalRefiner`: recebe mistura + stems grosseiros e corrige residuos no dominio do tempo com convolucoes depthwise dilatadas.

A configuracao padrao fica perto de 5,8M parametros. Isso e intencional: abaixo disso o modelo perde capacidade rapidamente; acima disso o treino no Kaggle com segmentos longos fica mais apertado.

## Treino

O treino usa quatro protecoes contra artefatos e overfitting:

- `MultiDomainLoss`: L1 no tempo + STFT multi-escala em 512, 1024, 2048 e 4096 pontos.
- `source_remix`: stems de musicas diferentes sao recombinados para multiplicar os cenarios de treino.
- augmentations: ganho por fonte, troca L/R, inversao de polaridade, dropout raro de fonte e perturbacao leve de velocidade/pitch.
- EMA + early stopping em validacao.

## Inferencia

`scripts/separate.py` usa chunks com overlap e janela Hann. Isso evita emendas secas entre blocos e permite processar musicas longas sem estourar VRAM.

## Limite honesto

Com apenas MUSDB18-HQ, um modelo pequeno pode separar bem, mas ainda pode deixar vazamento em mixagens densas. Para chegar mais perto de qualidade comercial, as proximas melhorias sao dados extras licenciados, fine-tuning por stem, ensembles pequenos e calibracao de losses por fonte.

