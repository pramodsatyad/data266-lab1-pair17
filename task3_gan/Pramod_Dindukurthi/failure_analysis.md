# Task 3 failure analysis — selected raw +5k model

Selected checkpoint: `2c2124442ae6a8665b811ec7c98c1ef29c2f689ea010f26386a9450f32fa2614`. A2B is Monet→Photo; B2A is Photo→Monet. The following analysis uses recorded held-out metrics, not inferred visual observations. Sample-specific visual judgments remain pending the independent audit.

| Evidence | Measured limitation | Interpretation and testable follow-up |
|---|---|---|
| A2B: FID 161.246396, KID 0.023164, 45 generated / 1,054 real | The generated and target feature distributions remain different under the local evaluator. | A lower FID than baseline does not establish photorealism. Inspect retained painting texture, lighting and object geometry in the fixed audit pairs; record actual sample IDs before attributing a failure. |
| A2B: coverage 0.104364, density 0.604444 | Coverage is low for this 45-prediction comparison. | The sample-count imbalance limits interpretation. Compare checkpoints under exactly the same split/counts; do not label this alone as mode collapse. |
| B2A: FID 165.131666, KID 0.015323, 1,054 generated / 45 real | Small real Monet reference set limits distributional conclusions. | Coverage=1.0 only means every one of the 45 reference feature neighborhoods was reached. It does not prove all Monet styles are represented or that every prediction is convincing. |
| Content cosine: A2B 0.837596, B2A 0.712652 | Both are slightly below the raw main baseline. | This is a feature proxy, not observed object damage. Have raters inspect whether objects/scene structure are retained; test objective changes only in separate controlled runs. |
| Cycle L1: A2B 0.067151, B2A 0.045788 | Low reconstruction error coexists with substantial distributional distance. | Reconstruction and target-domain appearance are distinct criteria. Review input/translation/cycle examples and the G/D curves together before diagnosing inadequate style transfer. |
| Official identity pilot: both arms best at +5k; later checkpoints did not improve monotonically | More optimization did not ensure a better measured score. | Preserve earlier checkpoints and compare on a fixed protocol. Replicate seeds before treating the small identity=2.5 advantage as reliable. No statistical significance is claimed. |

The official 300-per-direction FID protocol and the local held-out FID protocol are different; their absolute scores must not be compared as a training improvement. KID subset variation is not a confidence interval. Only 45 Monet validation images are available, limiting the precision of local conclusions.

The fixed audit packet is outputs/human_audit/rater_packet/. It contains 30 randomly selected, seeded validation pairs rather than curated attractive results. Two raters score independently. Human-audit status: Pending independent ratings. The fixed 30-sample packet exists; blank rater sheets are not completed human-audit results.

Visual failure examples are not yet claimed. After inspecting the packet, add a table with sample ID, direction, exact observed defect, the relative evidence image path, a hypothesis and a testable fix. Include weak as well as successful examples. Keep hypotheses distinct from measured observations. Do not copy the synthetic smoke-test ratings into this analysis.

Architecture/objective changes are future hypotheses only; no new experiment is reported here. The final selected raw +5k model and its official submission CSV are retained.

## Observed visual examples

Pending inspection. Add sample IDs, evidence links, observed defects and testable hypotheses here.
