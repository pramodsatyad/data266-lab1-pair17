# Task 3 results — Pramod Dindukurthi

A = Monet and B = Photo. A2B translates Monet to Photo; B2A translates Photo to Monet.

## Model and data

CycleGAN was implemented and trained from scratch: two 9-block, 64-channel ResNet generators with transposed-convolution upsampling, and two 70×70 PatchGAN discriminators. Instance normalization, least-squares adversarial loss, L1 cycle consistency and identity losses were used. The four learned networks contain 28,285,832 parameters; EMA storage and optimizer state are excluded.

The data contains 300 Monet and 7,038 Photo images. The fixed duplicate-filtered split has 255/45 Monet train/validation images and 5,974/1,054 Photo train/validation images. Training uses unpaired sampling, resize to 286, random 256 crops, flips and normalization to [-1,1]. Evaluation uses deterministic RGB resizing to 256. The split and hashes are retained in data_processed/splits.json; raw datasets are supplied separately.

## Training history and selection

The main model completed 350,704 optimizer updates. Its recorded update time was 10.5295 hours (37906.139 seconds), with 18.5038 real source images/s across both domains. This excludes checkpoint/preview overhead. Main settings were batch size 1, Adam learning rate 0.0002, betas (0.5, 0.999), cycle weight 10.0, identity weight 5.0, decay start 175,352, replay pool 50, EMA decay 0.999, AMP=False, seed 42.

Two pilot arms started from the same final raw main generators and discriminators, with fresh Adam optimizers, empty replay pools and matching continued data indices. They compared identity weights 5 and 2.5, with cycle weight 10, learning rate 0.00005, 20,000 additional updates per arm and decay after 10,000. Official raw evaluation was recorded at +5k/+10k/+15k/+20k.

| Arm | Extra updates | Mean official FID | Official MiFID | Estimated score |
|---|---:|---:|---:|---:|
| control_id5 | 5,000 | 100.289777 | 0.411713 | -50.350745 |
| control_id5 | 10,000 | 101.585295 | 0.412205 | -50.998750 |
| control_id5 | 15,000 | 100.841900 | 0.411305 | -50.626603 |
| control_id5 | 20,000 | 101.546524 | 0.411959 | -50.979241 |
| experiment_id2p5 | 5,000 | 100.104527 | 0.411176 | -50.257851 |
| experiment_id2p5 | 10,000 | 101.389827 | 0.411877 | -50.900852 |
| experiment_id2p5 | 15,000 | 101.149669 | 0.411200 | -50.780434 |
| experiment_id2p5 | 20,000 | 101.691790 | 0.411759 | -51.051774 |

The selected checkpoint is experiment_id2p5 at local step 5,000, using raw weights. Its ancestry is 350,704 + 5,000 = 355,704 updates. Both arms' final training_summary.json files describe +20k and are not the selected checkpoint's summaries. The full selected hash is `2c2124442ae6a8665b811ec7c98c1ef29c2f689ea010f26386a9450f32fa2614`; its parent hash is `346f84cbd5cc7ab9a6cc143d85b48f743002be45e2b4a841b5e0c464f11cb94f`.

The selected +5k pilot recorded 572.645 seconds of additional update time, 17.4628 source images/s and 2.5213 GiB maximum recorded allocated GPU memory. Main-plus-selected-pilot update time is 10.6886 hours. Later unselected pilot updates are excluded. Hardware and package versions are recorded in evidence/main_run/training_summary.json: RTX 4090, AMD Ryzen 9 7950X, Python 3.10.13, PyTorch 2.1.2, CUDA 12.1.

## Official submission evaluation

The selected average official FID is 100.104527; MiFID is 0.411176; estimated displayed score is -50.257851. This is an estimate, not a confirmed Kaggle score. The official CSV is submission.csv and remains unchanged. Actual public/private score and rank remain pending.

The official protocol uses 300 images per direction, instructor torchvision Inception features and index-paired cosine MiFID. It includes training inputs as permitted by the assignment. This protocol differs from the held-out local diagnostics below; the local FID values are not Kaggle scores. Selection improvement over the main raw estimate (-51.084785) is small, and the single-seed identity experiment does not establish a consistent identity-weight advantage.

## Held-out local diagnostics

These metrics use the same fixed validation split for baseline and selected raw models, torch-fidelity 2048-D features, KID with 50 subsets of size 45, density/coverage k=5, and LPIPS AlexNet v0.1. KID is unscaled unbiased MMD². Subset standard deviation is not a confidence interval. In A2B only 45 predictions are evaluated; in B2A only 45 real Monet references are available. FID and coverage must be interpreted with these sample counts.

| Direction | Real / generated | FID | KID mean ± subset SD | Density | Coverage | Cycle L1 [0,1] | LPIPS input→translation | LPIPS input→cycle | Content cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2B | 1054 / 45 | 161.246396 | 0.023164 ± 0.003015 | 0.604444 | 0.104364 | 0.067151 | 0.274377 | 0.272125 | 0.837596 |
| B2A | 45 / 1054 | 165.131666 | 0.015323 ± 0.002513 | 0.757116 | 1.000000 | 0.045788 | 0.436377 | 0.174814 | 0.712652 |

| Direction | Baseline local FID | Selected local FID | Change | Baseline KID | Selected KID |
|---|---:|---:|---:|---:|---:|
| A2B | 165.708978 | 161.246396 | -4.462583 | 0.026758 | 0.023164 |
| B2A | 165.861760 | 165.131666 | -0.730094 | 0.015772 | 0.015323 |

Both directional FID and KID values improved numerically relative to the main checkpoint. Content cosine declined slightly in both directions. A2B cycle error improved slightly, while B2A cycle error increased slightly. These measured changes do not by themselves establish better style or perceptual quality. LPIPS(input, translation) measures change and is not a quantity to minimize blindly for domain translation. See failure_analysis.md for quantitative limitations and the pending visual audit.

## Training diagnostics and human audit

Main and selected-pilot G/D, adversarial, cycle/identity, gradient and learning-rate plots are retained in outputs/plots/main_run/ and outputs/plots/selected_pilot/. Curves use deterministic display subsampling; full raw histories remain unchanged. The pilot plots end at +5,000. Last-100 recorded-update summaries, time, counts and finite-event counters are in evidence/selected_training_statistics.json and both metric CSVs. Selected pilot counters: NaN=0, Inf=0. Finite losses and small cycle error are numerical checks, not proof of good translation or convergence.

Human audit: Pending independent ratings. The fixed 30-sample packet exists; blank rater sheets are not completed human-audit results. The packet contains 15 samples per direction, 30 total. Two raters must independently score style, content and freedom from artifacts on integers 1–5, with higher values better for all criteria. Exact agreement is reported per criterion and direction after actual ratings are collected. Synthetic smoke-test ratings are excluded.

## Reproduction and outstanding evidence

See the repository-root TASK3_RUN_GUIDE.md for the smoke test and selected raw inference commands. The smoke test passed using synthetic data with exact CPU resume; it did not run full pretrained metrics and is not assignment quality evidence. Larger checkpoints and raw datasets are supplied separately. Durable retrieval of the selected weights must be documented and the hash verified before claiming that a fresh clone can run the trained model.

Completed metrics: full_metrics_report.csv and metrics_report.csv. Initial baseline diagnostics: evidence/main_run/local_metrics_step_00350704_raw/. Selected export/protocol: `outputs/selected_validation_2c2124442ae6_20261006T071131_193859Z`; hashes and paths: evidence/selected_validation.json. Both pilot arms and comparisons: experiments/identity_pilot_v1/. Original run logs/manifests are under repository reproducibility/. Initial packaging_inventory.json describes the organizer's copy-time snapshot; later report hashes are recorded in evidence/report_update.json.

Independent human ratings, sample-specific visual failure examples, durable checkpoint retrieval, actual Kaggle score/rank and teammate comparison in the final combined report remain to be completed. No missing result is inferred or substituted.
