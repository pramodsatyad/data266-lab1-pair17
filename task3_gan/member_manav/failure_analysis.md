# Task 3 Failure Analysis: Monet ↔ Photo CycleGAN

These cases come from a fresh per-image scoring pass on the **final model's own predictions**, in `outputs/gpu_final_swa`. The score file is `outputs/gpu_final_swa/per_image_scores.csv`, and the same two figures as before are saved in that folder: `failures_low_content.png` and `failures_high_lpips.png`. I opened the actual input and output images for each case below before writing its description, so these are real observations, not guesses from the numbers alone.

All 5 cases are in the **B2A** direction (photo → Monet), which matches `results.md`, where B2A is the harder direction.

## Failure Case 1: Night sky turned into daytime colors

- **File:** `93bd999d11.jpg` | **Direction:** B2A | **content_cosine:** 0.5194 | **LPIPS:** 0.8257
- **What goes wrong:** the input is a dark night sky full of stars, with a dark tree on the left edge. The output turns the black sky into a light blue and white textured wash, with orange streaks near the top, and the tree becomes a soft green blob.
- **Why:** the 300 Monet paintings almost never show a true night scene, so the model has no real example of "very dark sky" to copy from. It falls back to its normal daytime colors instead of staying dark. A fix to try: add a few extra dark-image augmentations, or a loss term that keeps overall brightness closer to the input.

## Failure Case 2: Checkerboard pattern on a smooth sunset

- **File:** `70b1293521.jpg` | **Direction:** B2A | **content_cosine:** 0.3514 (lowest in the set) | **LPIPS:** 0.7002
- **What goes wrong:** the input is a smooth, almost abstract sunset-over-water photo, soft orange-pink gradient, very little detail. The output has a clear checkerboard/grid pattern stamped over the image, and the color shifts from warm orange-pink toward yellow-green.
- **Why:** a smooth, low-detail input gives the generator very little texture to work with, so the upsampling grid pattern that is normally hidden under real texture becomes clearly visible. This is the same kind of image that scored worst in the earlier model too, so it is a repeated, real weak spot, not a one-off.

## Failure Case 3: Bright yellow field turns into mixed colors

- **File:** `dba404b1bd.jpg` | **Direction:** B2A | **content_cosine:** 0.3778 | **LPIPS:** 0.4557
- **What goes wrong:** the input is a bright yellow flower field under a blue sky with clouds. The output keeps the blue, cloudy sky fairly well, but the solid yellow field becomes a mix of red, orange, and yellow blotches instead of staying one even color.
- **Why:** large, flat, strongly-colored areas like this field are rare and extreme compared to most Monet paintings, so the model "paints" them with a mix of colors it has seen elsewhere instead of keeping the single strong color. A color-consistency loss, comparing color histograms between input and output, could help keep the field's main color closer to the original.

## Failure Case 4: Fine bird texture gets blurred

- **File:** `a546e53fbe.jpg` | **Direction:** B2A | **content_cosine:** 0.6991 (fairly high, content mostly kept) | **LPIPS:** 0.8559 (highest in the whole set)
- **What goes wrong:** the input is a huge flock of white birds on blue water, very fine, repeating detail. The output keeps the overall scene layout (which is why content cosine is still decent), but the individual birds blur into soft, blotchy shapes.
- **Why:** this is a good example of LPIPS and content cosine not fully agreeing. The scene shape survives, but LPIPS is sensitive to fine texture, and that texture is lost here. Resize-based upsampling was already tried (the `arch2` branch in `results.md`) and did not help overall, so a texture-aware loss term is a better next idea than changing the upsampling method again.

## Failure Case 5: Low score, but actually a reasonable result

- **File:** `83f6286883.jpg` | **Direction:** B2A | **content_cosine:** 0.4107 | **LPIPS:** 0.6217
- **What goes wrong:** the input is smooth, calm water with a distant pier line and a soft sunset color band. Looking at the actual output, this one is a fairly convincing painterly translation — the water, the horizon line, and the soft colors are all kept, and it looks reasonably Monet-like.
- **Why it's still in this list:** this shows a real limit of the automatic metrics, not of the model. Content cosine and LPIPS are computed from deep feature distances, and a successful, strong stylization can still shift those features even when the image looks fine to a human. This is exactly why the human audit (section 11 in `results.md`) matters.

## Training Stability

(Full detail and charts are in `results.md`, sections 4 and 8. Summary here for context.)

- **0 NaN losses** across the full training line used to build the final model, from step 0 through step 226,000.
- Each stage change (a new fine-tune branch, or the switch to a constant 5e-5 learning rate) caused a small, temporary bump in the loss curves, which recovered within a few thousand steps each time.
- Grad norms stayed controlled through every stage, including the long constant-learning-rate phase used to build the final model.
- The quick score kept improving across almost the whole line, with normal small ups and downs between checkpoints.

## Shortcomings and Fixes

1. **Dark or night photos come out too bright and too colorful** (Failure Case 1). The 300 Monet paintings rarely show night scenes, so the model has little to learn a "stay dark" style from.
   - **Fix to try:** add a brightness-matching term to the loss, or oversample what few dark-toned Monet paintings exist during training, and check if that reduces this effect without hurting the FID/MiFID score.

2. **Checkerboard / grid artifacts on smooth, low-detail regions** (Failure Case 2). This is a known issue with transposed-convolution upsampling in generators.
   - **Already tried:** swapping to resize-based upsampling (the `arch2` branch, see `results.md` section 4). It did not improve the score and was dropped.
   - **Fix to try instead:** a light smoothing or high-frequency penalty term in the loss, specifically targeting grid patterns, without changing the generator architecture again.

3. **Strong, flat colors shift toward a mixed palette** (Failure Case 3). Since there are only 300 Monet images, the model has a narrow idea of "Monet colors" and pushes unusual, strong input colors toward that narrow style.
   - **Fix to try:** a color-histogram consistency loss alongside the existing identity loss, checked against the worst-scoring images (like the ones above) without making the FID score worse.

4. **Fine repeating texture gets blurred** (Failure Case 4). High-frequency detail, like a flock of birds or fine ripples, is hard for the generator to keep sharp.
   - **Fix to try:** a texture or perceptual loss term, since changing the upsampling method alone (tried in `arch2`) did not fix this kind of problem.
