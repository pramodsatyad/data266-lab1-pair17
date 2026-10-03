# DATA266 Lab 1 — Pair 17

The repository preserves separate Task 1, Task 2, Task 3, report and reproducibility folders.
Task 3 now contains a portable PyTorch CycleGAN implementation for Pramod Dindukurthi.
Task 1 and Task 2 are not implemented by this change.

Start with [Task 3 instructions](task3_gan/Pramod_Dindukurthi/README.md) and
[the execution notebook](task3_gan/Pramod_Dindukurthi/src/task3_lab.ipynb).
The notebook runs in lab Jupyter, VS Code, Google-hosted Colab, or Colab connected
to the GPU-lab machine. Git is needed only on the laptop used for version control.

After installing a compatible PyTorch/torchvision pair and the Task 3 requirements,
the one-command CPU integration smoke test is:

```bash
python task3_gan/Pramod_Dindukurthi/src/smoke_test.py
```

This creates only synthetic temporary test artifacts. It tests data splitting,
both architectures, cycle gradients, training, exact CPU checkpoint resume,
image export, metric mathematics and the blinded audit workflow.
Add `--full-metrics` to also test Inception/LPIPS extraction and report generation;
this downloads pretrained evaluation weights on first use (never generation weights).

Training results, human ratings and Kaggle scores must come from real assignment
runs. The instructor evaluator is retained under Task 3 reference/ and adapted to
`src/evaluate_official.py`; it produces the confirmed ID,FID,MiFID CSV schema.
