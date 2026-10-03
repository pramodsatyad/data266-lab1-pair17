# Task 3 Failure Analysis: Monet ↔ Photo CycleGAN

These cases come from `per_image_scores.csv` — the real lowest content-cosine and highest-LPIPS images from the full 7,338-image evaluation, the same ones shown in `outputs/gpu/failures_low_content.png` and `outputs/gpu/failures_high_lpips.png`. I opened the actual input and output images for each one before writing the description below, so these are real observations, not guesses from the numbers alone.

All 5 are in the **B2A** direction (photo → Monet) — this matches the metrics table in `results.md`, where B2A has lower precision and higher LPIPS than A2B.

## Failure Case 1: Checkerboard artifacts on a smooth gradient

- **File:** `70b1293521.jpg` | **Direction:** B2A | **content_cosine:** 0.3985 (lowest in the whole set) | **LPIPS:** 0.6563
- **What goes wrong:** the input is a smooth, almost abstract sunset-over-water photo (soft orange-pink gradient, very little detail). The output has a strong checkerboard/grid pattern stamped over the whole image, and the color shifts from warm orange-pink toward a more yellow-green tone.
- **Why:** smooth, low-detail images give the generator very little texture to work with, so the upsampling artifacts that are normally hidden under real texture become clearly visible here.

## Failure Case 2: Same artifact, another smooth sunset

- **File:** `02ded12bbd.jpg` | **Direction:** B2A | **content_cosine:** 0.4072 | **LPIPS:** 0.7064 (one of the highest)
- **What goes wrong:** another smooth, dark, low-contrast sunset photo. The output again shows a clear checkerboard pattern and a strong color shift — the original's moody dark red/orange becomes a washed-out pale pink/blue.
- **Why:** same root cause as Case 1. Two of the worst-scoring images both being smooth sunset photos is a real pattern, not a coincidence — see "Shortcomings" below.

## Failure Case 3: Fine texture gets blurred

- **File:** `a546e53fbe.jpg` | **Direction:** B2A | **content_cosine:** 0.8425 (fairly high, content mostly kept) | **LPIPS:** 0.9210 (highest in the whole set)
- **What goes wrong:** the input is a photo of a huge flock of white birds on blue water — very fine, repetitive, high-frequency detail. The output keeps the overall scene layout (which is why content cosine is still decent), but the individual birds blur into a soft, blotchy pattern and lose their sharp edges.
- **Why:** this is a good example of why LPIPS and content cosine don't always agree. The scene "shape" survived, but LPIPS (which is sensitive to texture and fine detail) still rates this as the worst-looking translation, because the fine texture that matters for how the image actually looks got destroyed.

## Failure Case 4: Color shift on a high-contrast photo

- **File:** `f265b9c6d2.jpg` | **Direction:** B2A | **content_cosine:** 0.5440 | **LPIPS:** 0.8615
- **What goes wrong:** the input is a dramatic photo of lightning against a dark blue sky. The output keeps a faint trace of the lightning bolt shape, but the strong blue is replaced with a flat pastel pink/purple, and the same checkerboard texture noise from Cases 1–2 shows up again.
- **Why:** this is a combination of the smooth-area artifact (most of the image is a flat sky) and a color shift, likely because dramatic lightning photos are very different from anything in the 300-image Monet set.

## Failure Case 5: Low score, but actually a reasonable result

- **File:** `ee9b29e1b8.jpg` | **Direction:** B2A | **content_cosine:** 0.4513 | **LPIPS:** 0.4913
- **What goes wrong:** the input is a landscape (mountains, water, a small town, big clouds). Looking at the actual output, this one is honestly a fairly convincing painterly translation — the scene layout is clearly preserved and the brushstroke-like texture looks reasonably Monet-like.
- **Why it's still in this list:** this shows a real limitation of the automatic metrics, not of the model — content cosine and LPIPS are computed from deep feature distances, and a successful stylization naturally shifts those features even when the image looks fine to a human. This is a reminder that the human audit (section 11 in `results.md`) matters: it can catch cases like this where the numbers look bad but the image doesn't.

## Training Stability

(Full detail and plots are in `results.md`, section 8. Summary here for context.)

- **0 NaN losses across all 100,000 steps** of training (all 3 stages combined).
- **LR restarts at step 60,000 and step 80,000** (from extending `total_steps` in `gpu.yaml`) each caused a small, temporary bump in cycle loss, identity loss, and discriminator loss — for example cycle loss ticked up from about 0.147 to about 0.177 right after the step-60,000 restart, and from about 0.136 to about 0.165 right after the step-80,000 restart. Each time, the loss recovered back to the pre-restart trend within a few hundred to about 2,000 steps.
- **Grad norms** stayed controlled the whole time: the generator's gradient norm dropped fast from an initial spike near 70 down to the 22–27 range and stayed there; the discriminators' gradient norms stayed under 10 for almost the entire run.
- **No loss spikes or divergence** at any point — the quick score (the local proxy metric, logged every 5,000 steps) went down at every single evaluation point from step 5,000 to step 100,000, with no step where it got worse than the one before.

## Shortcomings and Fixes

1. **Checkerboard / grid artifacts on smooth, low-detail regions** (seen in Failure Cases 1, 2, and partly 4). This is a known issue with transposed-convolution-based upsampling in generators.
   - **Fix to try:** my `ResNetGenerator` (`src/models.py`) uses `nn.ConvTranspose2d` for upsampling, which is a known source of checkerboard artifacts. Swap it for a resize-then-convolution approach, or add a light smoothing/blur term to the loss specifically penalizing high-frequency grid patterns. Test by checking whether these same smooth-sunset-style images score better on LPIPS without hurting the overall FID/MiFID score.

2. **Color shifts on photos very different from the training Monet style** (seen in Failure Cases 1, 2, 4). Since there are only 300 Monet images, the model has a narrow idea of "Monet colors" and can push unusual input colors toward a Monet-typical palette too aggressively.
   - **Fix to try:** add a stronger color-consistency term (for example, comparing color histograms between input and output) alongside the existing identity loss, and check whether content_cosine and LPIPS improve specifically on the worst-scoring images (like the 5 above) without making the B2A images look less like real Monet paintings (watch the FID score to make sure it doesn't get worse).
