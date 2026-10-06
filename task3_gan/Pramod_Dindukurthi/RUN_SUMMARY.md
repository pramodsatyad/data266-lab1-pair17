# Task 3: actual training and selected model

A = Monet, B = Photo. A2B is Monet-to-Photo; B2A is Photo-to-Monet.

The main CycleGAN was trained from scratch for **350,704 optimizer updates**, using identity weight 5. Its recorded training time is 10.53 hours of update time; this excludes checkpoint and preview overhead. See `evidence/main_run/training_summary.json` and the unchanged main-run logs and manifests under the repository's `reproducibility/` folder.

Two controlled pilot arms started from the same final main-run raw generators and discriminators. They used fresh Adam optimizers, fresh replay pools, cycle weight 10, initial learning rate 0.00005 and identity weights 5 versus 2.5. Each arm ran 20,000 additional updates and was evaluated every 5,000 updates. Their complete comparison is `experiments/identity_pilot_v1/comparison.csv`.

The selected checkpoint is **experiment_id2p5 after +5,000 updates**, raw weights. Its training ancestry contains **355,704 updates**. The checkpoint's local step is 5,000. The pilot arm's final `training_summary.json` describes +20,000 and must not be presented as the selected checkpoint summary.

The official 300-image-per-direction evaluation gives average FID **100.104527**, average MiFID **0.411176**, and **estimated** displayed score **-50.257851**. No actual Kaggle score is asserted here. The main raw baseline estimate was -51.084785. The identity=2.5 +5k candidate has a small advantage over its control; this single-seed experiment does not establish a consistent identity-weight benefit.

`submission.csv` is an exact copy of the winning CSV. Its images are `outputs/pred_A2B/` and `outputs/pred_B2A/`, with hashes in `outputs/export_manifest.json`. These are the official capped evaluation images, which include training inputs; they are not a held-out validation export. Broader local metrics and the human audit must be reported separately, with their own protocols and checkpoint identity.

The original training folders are retained outside this clone. Raw logs, original plans and evaluation manifests are copied without edits. Paths recorded in those historical files refer to the original execution environment. Runnable entry points accept paths through arguments/configuration. Main-run software copied into `src/` is the current runtime copy; frozen pilot software is separately retained in `experiments/identity_pilot_v1/source/`. Segment manifests remain the authority for software provenance.

See `TASK3_RUN_GUIDE.md` at the repository root for the smoke test, selected-model inference commands and Git staging instructions. Complete `results.md`, `failure_analysis.md`, broader metrics, human ratings and the team report using actual evidence before claiming rubric completion.
