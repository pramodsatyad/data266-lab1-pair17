# Task 3 — portable CycleGAN

**A = Monet; B = Photo.** A2B means Monet-to-Photo; B2A means Photo-to-Monet.
Both networks are trained from scratch. Baseline: 9-block ResNet generators and
70x70 PatchGAN discriminators. Experimental configuration: resize-convolution
upsampling. Both configurations track raw and EMA generator weights.

## Start in the GPU lab

1. Extract the supplied `task3-code.zip` into a project folder on the lab machine.
   There is no Git dependency. Keep `CODE_BUNDLE_MANIFEST.json` with the extracted code.
2. Use the lab's working CUDA PyTorch kernel. Python >=3.10 is required;
   the trainer supports both PyTorch 2.1 CUDA AMP and newer AMP namespaces. Verify `torch.cuda.is_available()` before full training.
3. Open `src/task3_lab.ipynb` in Jupyter or Colab connected to that lab machine.
   For a local runtime, the Colab browser and Jupyter server run on the lab machine;
   no Google GPU is involved. The notebook includes a setup check and instructions.
4. Set the project root, data root and persistent backup directory in the first cells.
   Put raw images in `data_root/monet_jpg/` and `data_root/photo_jpg/`.
   Use only instructor-permitted development images; do not mix a hidden test set in.
5. Run CPU smoke test, prepare splits, and run a short GPU benchmark. Use its measured
   seconds/step to set a realistic full-run step budget and decay start BEFORE training.
6. Start training. The notebook defaults to a configurable 12-hour overnight budget
   and a four-hour first phase in a detached background job. Use its status cell
   to check logs and checkpoints. After evaluating and preserving the first-phase
   snapshot, set SESSION_HOURS=8.0 and rerun the launch cell to resume.
   TRAIN_HOURS controls the full schedule; SESSION_HOURS only controls this phase.
   Do not run evaluation until training has stopped. Machine sleep, shutdown, or lab
   session cleanup can still stop the job; verify overnight access before leaving.
   Set BACKGROUND_TRAINING=False for foreground training with notebook interrupts.
7. Evaluate immutable checkpoint snapshots, compare raw/EMA on the SAME validation
   set, select a checkpoint with a written reason, and complete the human audit.

For Colab **hosted** runtimes, copy the whole project into the runtime and mount
Drive for backups; opening just the notebook does not install sibling Python files.
With a **local** runtime, `google.colab.drive.mount()` is unavailable. Use a lab
persistent disk, external drive, approved network storage, or explicit downloads.
Saving the notebook does NOT back up model checkpoints or generated files.

## Dependencies and offline preparation

Preserve the lab's matching torch/torchvision CUDA installation. If absent, install
the matching pair using the lab's instructions or https://pytorch.org/get-started/locally/.
Then, using the same Python interpreter as the notebook:

```bash
python -m pip install -r task3_gan/Pramod_Dindukurthi/requirements.txt
```

Full metrics download the torch-fidelity Inception weights and LPIPS AlexNet weights
on first use. Pre-download them during setup if lab network access is limited. Set
`TORCH_HOME` to an approved persistent cache. These weights are evaluation-only;
they never modify submitted images. The run records installed package versions and
`pip freeze`. CUDA mixed precision is optional; CPU smoke tests use float32.

## File layout

```text
task3_gan/data/{monet_jpg,photo_jpg}/     # shared raw data, not in Git
task3_gan/Pramod_Dindukurthi/
  src/task3_lab.ipynb                    # execution notebook; save lab outputs
  src/{models,data,train,infer,metrics,checkpointing}.py
  src/{preflight,smoke_test,audit,report,select_checkpoint}.py
  src/{package_code,record_submission,collect_evidence}.py
  configs/{baseline,resizeconv}.yaml
  requirements.txt
  data_processed/splits.json             # runtime-created split manifest
  checkpoints/latest_checkpoint.pt       # full recovery checkpoint
  checkpoints/periodic/step_*.pt
  checkpoints/best_model.pt              # explicitly selected checkpoint
  outputs/evaluations/<checkpoint-weight>/  # immutable candidate image exports
  outputs/{pred_A2B,pred_B2A}/            # complete selected export
  outputs/plots/
  outputs/human_audit/
  evaluate_local.py
  full_metrics_report.csv                # created from actual evaluation
  metrics_report.csv                     # same source for general lab rubric
  submission.csv                         # only from official evaluator output
  reproducibility/{raw_logs,manifests}/   # self-contained run evidence
  results.md
  failure_analysis.md
```

Default `RUN_DIR` is your member folder, matching the requested structure. For a
second experiment choose a new run directory (e.g. `experiments/resizeconv_seed42`)
and keep both runs. `collect_evidence.py` copies the selected run's logs/manifests
into the team's top-level `reproducibility/` folders without changing originals.

## Terminal equivalents

Run from the member folder; all paths may be overridden. Replace `BACKUP_PATH`
with your actual persistent directory; it is a runtime argument, not a committed path.

```bash
python src/preflight.py --data-root ../data
python src/smoke_test.py
python src/data.py --data-root ../data --output data_processed/splits.json
python src/train.py --config configs/baseline.yaml --data-root ../data --splits data_processed/splits.json --run-dir . --backup-dir BACKUP_PATH
```

Resume the same run with the same config and data:

```bash
python src/train.py --config configs/baseline.yaml --data-root ../data --splits data_processed/splits.json --run-dir . --backup-dir BACKUP_PATH --resume checkpoints/latest_checkpoint.pt
```

The notebook creates a runtime config with benchmark-adjusted step counts. When
using it, resume with that exact saved runtime config, not the original YAML.
`--stop-after N` ends cleanly after N additional steps without restarting/changing
the LR schedule. `max_hours` is the wall-time limit PER INVOCATION, not total lifetime.

## Recovery and evidence

Checkpoints contain four networks, two generator EMA copies, both optimizers,
schedulers, AMP scaler, replay pools, Python/NumPy/CPU/CUDA RNG states, completed
step, dataset/config identity and throughput counters. Data sampling/augmentation
is keyed to sample index so DataLoader prefetch does not change resumed samples.
Exact CPU recovery is tested; exact bitwise reproduction across GPU types, package
versions or nondeterministic CUDA kernels is not promised.

Atomic writes protect against partial files. Every resume creates NEW log segments.
Failed/interrupted runs keep the last complete checkpoint; non-finite events are
printed explicitly. Do not overwrite a run by changing its config and resuming.
If a problem needs a changed objective/precision setting, preserve the failed run
and deliberately create/document a new experiment.

Use an immutable periodic checkpoint for comparisons. `latest_checkpoint.pt` can
change during training. The notebook creates a separate evaluation snapshot before
inference. `best_model.pt` is created only after explicit validation selection;
the script never assumes the latest or lowest-cycle-loss model is best.

## Metrics and interpretation

`evaluate_local.py` computes BOTH directions: local FID, KID, density/coverage,
cycle L1, LPIPS(input, translation), LPIPS(input, cycle), and input/translation
Inception-feature cosine. Inputs and cycles are lossless PNG; translations are
fixed-quality RGB JPG. FID references are the fixed held-out target-domain images.
KID is unscaled unbiased MMD^2 and can be negative. Its subset standard deviation
is NOT a confidence interval. Report real/generated sample counts; small-sample
FID is noisy and biased. Density/coverage uses k=5 and requires >5 real images.

Cycle L1 uses tensors on [0,1] before encoding, not arbitrary unpaired targets.
Training cycle/identity losses in logs use [-1,1] and are unweighted components;
the configured weights are separately recorded. LPIPS input/translation can favor
under-stylization and must be interpreted with style evidence. Inception cosine
is a semantic proxy, not proof of content fidelity. Training throughput counts
BOTH real-domain images per optimizer step, includes data wait and update time,
and excludes checkpoint/preview time. Session wall time is separately recorded.
Parameter totals exclude EMA duplicate storage. Peak GPU memory is allocated
PyTorch GPU memory (not total reserved GPU memory); CPU-only reports use null.

The local metrics use a different Inception implementation from the supplied instructor
evaluator. For Kaggle, use `src/evaluate_official.py` and its confirmed `ID,FID,MiFID`
output. The original instructor notebook is preserved in `reference/`. Do not compare
these FID implementations as if they were interchangeable. Follow any separate instructor
requirements for the source input set, count and direction. Unavailable leaderboard scores
remain pending. No script uploads to Kaggle automatically.

## Blinded human audit

After checkpoint selection, `audit.py prepare` deterministically samples 15 examples
per direction. Give BOTH independent raters only `rater_packet/`, never the private
key, architecture or model score. All three criteria use 1-5, with higher better;
artifact score means freedom from artifacts. After each rater fills their own sheet,
`audit.py summarize` validates all 30 ratings and computes pooled means and exact
percentage agreement per criterion, both directions and overall. This satisfies the
rubric's allowed percentage-agreement alternative. Rerun the report to include it.
Actual people must rate actual generated images; smoke-test ratings are synthetic.

## Incremental commits and transfer

Commit real working milestones on the laptop. Capture `git rev-parse HEAD` before
training and enter it as `CODE_VERSION`; the run additionally hashes Python source.
For no-Git lab transfer: `python src/package_code.py --output task3-code.zip`.
The ZIP deliberately excludes datasets, secrets, checkpoints and generated results.
It is only a CODE bundle, never a training backup. Save completed raw logs unchanged.
Record checkpoint SHA-256 and persistent retrieval instructions; keep large binaries
outside Git. Review executed notebook output for private paths or tokens before push.

## References

- Zhu et al., CycleGAN: https://arxiv.org/abs/1703.10593
- Author reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- Resize-convolution: https://distill.pub/2016/deconv-checkerboard/
- EMA: https://arxiv.org/abs/1806.04498
- KID: https://arxiv.org/abs/1801.01401
- Density/coverage: https://arxiv.org/abs/2002.09797
- LPIPS: https://github.com/richzhang/PerceptualSimilarity
- Inception implementation: https://github.com/toshas/torch-fidelity
- Competition: https://www.kaggle.com/competitions/data-266-fall-2026-gan-image-style-transfer/overview


## Instructor evaluator now available
The supplied original is preserved at `reference/Part3_Evaluation_Script.ipynb`.
`src/evaluate_official.py` preserves its torchvision Inception weights, normalization,
first-300 sorted file selection, matched counts and index-wise cosine-distance formula.
It writes the confirmed `ID,FID,MiFID` schema. Use the local metrics for the broader
rubric and fixed held-out validation; report official-protocol metrics separately.
The official reference folders include training images; this is not held-out evaluation.
Do not reorder or cherry-pick generated images to influence its index-based score.

```bash
python src/evaluate_official.py --data-root ../data --export-dir outputs/evaluations/YOUR_EXPORT --output submission.csv
```
This command does not upload anything. The evaluator itself does not specify which
source input set must be translated; follow instructor input-set instructions. Keep
inference ordering and sample selection fixed before evaluating.
