# Task 3 run and repository guide

Task 3 evidence includes the 350,704-update main run, both 20,000-update pilot arms and the selected identity=2.5 +5,000 checkpoint. See `task3_gan/Pramod_Dindukurthi/RUN_SUMMARY.md`.

Run from the repository root with the documented compatible PyTorch environment:

```bash
python -m pip install -r task3_gan/Pramod_Dindukurthi/requirements.txt
python task3_gan/Pramod_Dindukurthi/src/smoke_test.py
```

The smoke test uses synthetic data and temporary folders; it does not restart the long training run. Save its actual pass/fail output.

For a new held-out export of the selected raw weights, provide the permitted dataset at `task3_gan/data/{monet_jpg,photo_jpg}` and retrieve `checkpoints/best_model.pt`:

```bash
python task3_gan/Pramod_Dindukurthi/src/infer.py --checkpoint task3_gan/Pramod_Dindukurthi/checkpoints/best_model.pt --data-root task3_gan/data --splits task3_gan/Pramod_Dindukurthi/data_processed/splits.json --output task3_gan/Pramod_Dindukurthi/outputs/selected_validation --weights raw --split val
python task3_gan/Pramod_Dindukurthi/evaluate_local.py --export-dir task3_gan/Pramod_Dindukurthi/outputs/selected_validation --run-dir task3_gan/Pramod_Dindukurthi/experiments/identity_pilot_v1/experiment_id2p5
```

Use a new output directory if an export already exists. The run summary for that pilot arm describes its final +20k state; any runtime totals from it are not timings for the selected +5k checkpoint. Broader local FID uses a different protocol from the official submission score and must be labeled accordingly.

Stage the explicitly packaged files, including selected outputs/splits ignored by the original repository rules:

```bash
bash TASK3_STAGE.sh
git diff --cached --stat
```

To include selected weights and any oversized raw logs using Git LFS (requires installed LFS and sufficient account storage):

```bash
bash TASK3_LFS_STAGE.sh
```

After reviewing staged changes and completing the readiness items:

```bash
git commit -m "Document Task 3 main training, controlled pilot, and selected raw model"
git push -u origin pramod-task3-final
```

The LFS helper uses the exact large-file allowlist in `TASK3_LFS_PATHS.txt`; it stages weights and raw logs without changing their original bytes. Confirm that the LFS objects are uploaded successfully. A commit also includes anything already staged by earlier work, so review the entire index before committing.

The organizer neither trains nor evaluates nor commits nor pushes. Existing code files that differ are backed up outside the repository before replacement. Existing immutable evidence with different bytes causes a preflight stop.
