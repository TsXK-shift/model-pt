# Tiny Mask MSS v3 Architecture

## Why v3 exists

The v1/v2 design allowed the network to produce four independent waveforms. Early in training it could reduce loss by making every stem sound close to the full mixture, which is exactly the failure mode heard in tests.

v3 changes the constraint: the model predicts a source distribution per stereo STFT bin.

```text
mixture STFT -> neural net -> logits [source, channel, freq, time]
                         -> softmax over source
                         -> masks that sum to 1
                         -> source STFTs -> ISTFT
```

Because masks sum to one across sources, the estimated stems partition the mixture instead of duplicating it.

## Blocks

1. STFT frontend
   - stereo complex STFT;
   - compressed real/imag features;
   - log magnitude for left/right;
   - mid/side log magnitude.

2. Spectral U-Net
   - 2D encoder/decoder over frequency and time;
   - depthwise-separable residual blocks to keep parameters low;
   - progressive dilation for receptive field without a huge model.

3. Wave context
   - a small Conv1D encoder reads the raw waveform;
   - FiLM conditioning injects waveform context into the spectral bottleneck.

4. Axial attention
   - lightweight attention over time and frequency at the bottleneck only;
   - avoids the cost of full attention on the high-resolution spectrogram.

5. Competitive masks
   - logits are reshaped to `[batch, source, channel, freq, time]`;
   - `softmax(dim=source)` creates nonnegative masks;
   - each stem uses mixture phase for ISTFT.

## Training loss

The config weights the losses toward the signal a mask model can learn well:

- lower `time_l1`, because perfect target phase is not fully reachable with mixture phase;
- stronger multi-scale log magnitude;
- strong `ratio_mask`, which directly teaches which source owns each time-frequency region;
- small complex loss, only as a phase/stability guide.

## Parameter target

The default config is about 8.69M trainable parameters, inside the requested 5-10M range.
