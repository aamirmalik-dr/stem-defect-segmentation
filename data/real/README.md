# Running on your own STEM image

The segmenter is trained only on simulation, so on a real micrograph it can
only give a qualitative, ground-truth-free segmentation. This directory is
intentionally empty of image data: no real image is committed, because a
clean, correctly licensed atomic-resolution STEM image with the right
projection is not bundled here. Bring your own instead.

## Bring your own image

Any grayscale PNG, TIFF or JPEG works:

```
stemseg segment path/to/your_image.png --method unet --figure overlay.png
```

The loader (`stemseg.real`) handles the three things a real frame usually
needs:

- **Contrast normalisation** to the 1st-99th percentile, so bit depth and
  detector gain do not matter.
- **Inversion** with `--invert`, for bright-field or contrast-inverted data
  where atomic columns are dark rather than bright.
- **Downsampling** with `--downsample N`, to bring the atomic-column width
  into the range the model was trained on (roughly a 2 to 3 pixel probe).
  If columns in your image are, say, 8 pixels wide, `--downsample 3` brings
  them close to the training regime.

## Honest expectations

- The training imaging model is incoherent Z-contrast with a Gaussian probe.
  Real instruments add aberrations, a detector transfer function, dynamical
  scattering and scan noise this simulator does not model, so expect a domain
  gap. The model card discusses it.
- There is no ground truth for a real image, so the CLI prints no metrics for
  it; it only writes the overlay. Treat the vacancy and dopant classes on real
  data with particular caution, since those are the hardest even in
  simulation.
