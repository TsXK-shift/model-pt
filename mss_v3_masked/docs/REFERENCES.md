# Referencias Tecnicas

Estas foram as referencias usadas para guiar as escolhas do baseline:

- Hybrid Transformer Demucs: arquitetura hibrida temporal/espectral com Transformer no gargalo. https://arxiv.org/abs/2211.08553
- Hybrid Demucs: evidencia forte para combinar waveform e espectrograma em MSS. https://arxiv.org/abs/2111.03600
- BS-RoFormer: band-split + Transformer com RoPE, vencedor no SDX23 e referencia moderna para MSS espectral. https://arxiv.org/abs/2309.02612
- Mel-Band RoFormer: uso de bandas mel sobrepostas em vez de band split heuristico. https://arxiv.org/abs/2310.01809
- SDX23 Music Demixing Track: competicao recente focada tambem em robustez contra dados ruidosos. https://arxiv.org/abs/2308.06979
- TFC-TDF-UNet v3: baseline eficiente de alto desempenho usado no contexto do SDX23. https://arxiv.org/abs/2306.09382
- MUSDB18 / MUSDB18-HQ: dataset padrao com 150 musicas e stems `drums`, `bass`, `vocals`, `other`. https://sigsep.github.io/datasets/musdb.html

O modelo deste repositorio nao copia essas arquiteturas grandes. A v3 pega os principios mais importantes para um limite pequeno de parametros: mascara competitiva por fonte, atencao barata no gargalo, contexto de waveform, campo receptivo dilatado e loss multi-dominio.
