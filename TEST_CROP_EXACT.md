# Test Plan: --crop-exact

This document describes the manual test cases for the `--crop-exact` feature.
These tests must be run after a successful build of NVEncC.

## Prerequisites

Build NVEncC on Windows with CUDA support. Place the binary at a known path
and have `ffmpeg` available for verification.

## Test Setup

Create a synthetic 1920x1080 NV12 test video (60 frames, 30 fps):

```
ffmpeg -y -f lavfi -i "testsrc=size=1920x1080:rate=30:duration=2" -pix_fmt nv12 -c:v rawvideo test_input.nut
```

## Test 1 — Regression: existing crop behaviour unchanged

Command:
```
NVEncC64 -i test_input.nut --crop 0,2,0,2 -o test_regression.mp4
```

Expected: encodes without error, output is 1920x1076.

## Test 2 — Exact crop with odd top+bottom

Command:
```
NVEncC64 -i test_input.nut --crop 0,1,0,1 --crop-exact -o test_exact_0111.mp4
```

Expected: encodes without error, output is 1920x1078.

## Test 3 — Exact crop with odd top only

Command:
```
NVEncC64 -i test_input.nut --crop 0,1,0,0 --crop-exact -o test_exact_0100.mp4
```

Expected: encodes without error, output is 1920x1079.

## Test 4 — Exact crop with odd bottom only

Command:
```
NVEncC64 -i test_input.nut --crop 0,0,0,1 --crop-exact -o test_exact_0001.mp4
```

Expected: encodes without error, output is 1920x1079.

## Test 5 — Exact crop with non-trivial odd values

Command:
```
NVEncC64 -i test_input.nut --crop 0,3,0,1 --crop-exact -o test_exact_0301.mp4
```

Expected: encodes without error, output is 1920x1076.

## Test 6 — Negative: odd crop without --crop-exact

Command:
```
NVEncC64 -i test_input.nut --crop 0,1,0,1 -o test_neg_noexact.mp4
```

Expected: exits with error "crop should be divided by 2 (0,1,0,1)".

## Test 7 — Negative: odd left/right with --crop-exact

Command:
```
NVEncC64 -i test_input.nut --crop 1,0,0,0 --crop-exact -o test_neg_oddleft.mp4
```

Expected: exits with error "--crop-exact: left/right crop must be even".

## Test 8 — Negative: odd output height

Command:
```
NVEncC64 -i test_input.nut --crop 0,2,0,1 --crop-exact -o test_neg_oddheight.mp4
```

Expected: exits with error "--crop-exact: output height (1077) must be even".

## Test 9 — Negative: interlaced

Command:
```
NVEncC64 -i test_input.nut --crop 0,1,0,1 --crop-exact --interlace tff -o test_neg_interlace.mp4
```

Expected: exits with error "--crop-exact is not supported with interlaced encoding".

## Test 10 — Visual verification

Compare a frame from test_exact_0111.mp4 against a reference produced by
ffmpeg using `-vf "crop=in_w:in_h-2:0:1"` followed by chroma phase shift:

```
ffmpeg -i test_input.nut -vf "crop=1920:1078:0:1" -frames:v 1 ref_exact.png
ffmpeg -i test_exact_0111.mp4 -frames:v 1 out_exact.png
```

Use an image diff tool (e.g. `compare -metric AE ref_exact.png out_exact.png diff.png`).
Differences should be limited to the chroma sub-sample shift (sub-pixel vertical
displacement of 0.5 chroma rows at the top edge) — this is expected and correct.

## Test 11 — Round-trip verification

For a frame where the top/bottom 1 pixel is the same colour, the output of
`--crop 0,1,0,1 --crop-exact` should be visually identical to a frame where
the top/bottom 2 pixels are cropped with the normal mode:

```
NVEncC64 -i test_input.nut --crop 0,2,0,2 -o test_ref_0202.mp4
```

Then:
```
ffmpeg -i test_exact_0111.mp4 -frames:v 1 out_0111.png
ffmpeg -i test_ref_0202.mp4 -frames:v 1 out_0202.png
ffmpeg -i out_0202.png -vf "crop=in_w:in_h-2:0:1" -y out_0202_top1bot1.png
```

Compare: `compare -metric AE out_0111.png out_0202_top1bot1.png diff.png`
Some difference is expected at the chroma boundary (the exact-crop version
does proper 3:1 chroma interpolation, the simple crop loses the top/bottom
chroma rows). The exact-crop version should look subjectively closer to the
original.

## Test 12 — Command-line reconstruction

Command (verify --crop-exact is round-tripped in the log):
```
NVEncC64 -i test_input.nut --crop 0,1,0,1 --crop-exact -o test_0111.mp4 --log-format json 2>&1 | grep crop
```

Expected: log contains both "--crop 0,1,0,1" and "--crop-exact".
