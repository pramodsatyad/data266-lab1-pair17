# Task 2 — Error Analysis

Model selected for manual review: **TextCNN**

Twenty test-set errors were manually reviewed:

- 5 confident false positives
- 5 confident false negatives
- 5 near-threshold errors
- 5 slice-specific contrast failures

## Confident False Positives

### Example 29330

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.999994
- **Candidate category:** `confident_false_positive`
- **Error type:** label ambiguity / likely noisy label

**Text:** Wow love the place and everything is very clean and new!\n\nGreat place to come and relax worth a try!\n\nCheers,\n\nEric Van Nguyen\nVisited April 2012

**Analysis:** The review is strongly positive in wording ('love', 'clean', 'great place', 'worth a try'), but the ground-truth label is negative. The model prediction is linguistically reasonable, suggesting possible label noise or context missing from the review.

**Testable fix:** Audit similarly extreme label-text disagreements and test training with label-noise filtering or robust loss.

### Example 37771

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.999992
- **Candidate category:** `confident_false_positive`
- **Error type:** contrast / sentiment reversal

**Text:** I have to say I love cafe rio! Their food is simple yet delicious, the service is great as well as consistent, and my appetite and tummy are always satisfied. However I've been to this location 3 different times and have not experienced any of these.. Even though it may be convenient for many of us here in henderson, please listen to the other reviews and DO NOT EAT AT THIS LOCATION ! New management is highly needed

**Analysis:** The review begins with highly positive statements about Cafe Rio, but then reverses sentiment for this specific location and ends with 'DO NOT EAT AT THIS LOCATION'. The model appears to overweight the positive opening.

**Testable fix:** Add stronger contrast-aware modeling or augment training with reviews containing 'however', 'but', and late sentiment reversals.

### Example 26678

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.999971
- **Candidate category:** `confident_false_positive`
- **Error type:** implicit sentiment / label ambiguity

**Text:** Clubs, Bars and Hostels \nA college kid's night out dream\nWeekends are crazy

**Analysis:** The short review is mostly descriptive and does not contain an explicit negative cue. The negative ground-truth label is difficult to infer from the text alone.

**Testable fix:** Use more contextual representations and audit ambiguous short examples; consider confidence-based handling of weakly expressed sentiment.

### Example 11429

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.999919
- **Candidate category:** `confident_false_positive`
- **Error type:** sarcasm

**Text:** Yay. Cafeteria food. Yay.

**Analysis:** The repeated 'Yay' is sarcastic. A bag-of-local-patterns model can treat 'Yay' as positive even though 'Cafeteria food' and the repetition indicate dissatisfaction.

**Testable fix:** Add sarcasm-oriented augmentation and features that model punctuation, repetition, and incongruity between positive cue words and surrounding context.

### Example 12480

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.999917
- **Candidate category:** `confident_false_positive`
- **Error type:** mixed sentiment / label ambiguity

**Text:** my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!

**Analysis:** The review contains a mild complaint about portion size and price, but also praises the omelette, bacon, and server. The overall wording is substantially positive despite the negative label.

**Testable fix:** Train on more mixed-sentiment examples and audit borderline labels; consider sentence-level sentiment aggregation before document classification.

## Confident False Negatives

### Example 22707

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999984
- **Candidate category:** `confident_false_negative`
- **Error type:** weak positive sentiment / mixed cues

**Text:** Its a nice buffet looking over the Aria pool, but variety was just ok.... Meh..

**Analysis:** The review contains positive language ('nice buffet') but also weak negative cues ('just ok', 'Meh'). The model appears to focus on the negative ending.

**Testable fix:** Add more mixed-review examples and test sentence-level or clause-level pooling so weak positive and negative cues can be balanced.

### Example 30793

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999981
- **Candidate category:** `confident_false_negative`
- **Error type:** temporal sentiment reversal

**Text:** This place is so much better since they changed owners.\n\nMy wife and I went when it was the old owners, it was terrible.  We waited forever and the food never came before we walked out.  People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made more\"".\n\nIt was horrible.\n\nNow its much better.  The staff are very friendly, they treat their customers very well and ...

**Analysis:** Most of the middle of the review describes a terrible experience under the old owners, but the current evaluation is explicitly positive after the ownership change. The model is dominated by the long negative history.

**Testable fix:** Model discourse and temporal transitions explicitly and augment with reviews where the final/current sentiment reverses earlier sentiment.

### Example 16532

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999969
- **Candidate category:** `confident_false_negative`
- **Error type:** contrast / label ambiguity

**Text:** Great gym, but if you ever try to cancel your agreement count on paying 2-3 months longer than expected.  I've been trying to cancel my account because I moved across\nThe country.  They won't cancel it unless I send in a certified letter.  I've done that and am still getting charged months later.

**Analysis:** The review starts with 'Great gym' but the rest describes a severe cancellation and billing problem. The text itself reads predominantly negative even though the provided label is positive.

**Testable fix:** Audit labels for contrast-heavy examples and test clause-weighting that gives more weight to the concluding complaint.

### Example 18956

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999968
- **Candidate category:** `confident_false_negative`
- **Error type:** slang / polarity cue mismatch

**Text:** So good. Pho Van's broth kills it.\nI want to bathe in that shit-

**Analysis:** Phrases such as 'so good' and 'kills it' are strongly positive slang, but informal wording and profanity may be poorly represented by the learned vocabulary.

**Testable fix:** Add informal-language and slang examples to training and preserve useful colloquial tokens during preprocessing.

### Example 6520

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999947
- **Candidate category:** `confident_false_negative`
- **Error type:** mixed sentiment / contrast

**Text:** its an enjoyable atmosphere for all 21+\n(: The beer is Delicious and so is the food - However I unfortunately, can not say the same about the HELP.The service was terrible the waitress were rude not only  to us but to each other....

**Analysis:** The review praises the atmosphere, beer, and food, then strongly criticizes the service after 'However'. The model predicts negative, while the supplied label is positive, making the overall target inherently mixed.

**Testable fix:** Use sentence-level sentiment aggregation or contrast-aware pooling and audit mixed-aspect reviews for label consistency.

## Near-Threshold Errors

### Example 19442

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.500163
- **Candidate category:** `near_threshold_error`
- **Error type:** mixed-aspect long review

**Text:** Meal: Dinner\n\nRationale: After drinking at Reservoir, I was starving. Casa Tapas was the closest place on my list of vegetarian-suitable restaurants in Montreal as gleaned from Chowhound.com. And I just think tapas is neat.\n\nFood: Small plates, kind of pricey, nothing unexpected. In order of appearance: (0) We had red wine. I forget what kind. Probably French or Italian. It was fine. (1) Free bread. Immediately. Lots and lots of free bread...

**Analysis:** The review contains both strong positives and negatives across food, price, service, and atmosphere. The final recommendation is only mildly positive, leaving the example close to the decision boundary.

**Testable fix:** Use hierarchical or sentence-level aggregation for long reviews and test aspect-aware sentiment pooling.

### Example 9863

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.500524
- **Candidate category:** `near_threshold_error`
- **Error type:** ambiguous / mixed descriptive review

**Text:** Fun grungy bar with an interesting crowd. Emphasize the \""GRUNGY!\"" Despite the hard appearance of the crowd, everyone has always been pretty laid back.\n\nThe live music varies nightly but is always loud and lively, played from a crowd-level makeshift stage. \n\nThe \""Ass Juice\"" is good--just don't look at the illustration while you are drinking it. \nThe \""Bacon Martini\"" is, as stated by the bartender, \""Really good bacon soaked in ...

**Analysis:** The review describes the venue positively overall but includes strongly marked negative-sounding descriptors such as 'GRUNGY' and jokes about bad vodka. These mixed lexical cues make the near-threshold prediction understandable.

**Testable fix:** Add more contextual training examples where apparently negative adjectives are used positively or humorously.

### Example 23782

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.500597
- **Candidate category:** `near_threshold_error`
- **Error type:** long-range sentiment reversal

**Text:** I've now made three trips to Lobby's.  The first was an enormous disappointment, the last two have been exceptional.\n\nOn my first trip, I made the mistake of ordering the Italian beef sandwich.  What I received could best be described as a thinly-sliced beef sandwich on a baguette that some jerk spilled soup on.  My sandwich was soggy to the point of disintegration from the moment I received it.  I tend to be a very neat eater, but felt I ne...

**Analysis:** The review opens with a strongly negative first visit, then uses 'However' to describe two later exceptional visits and ultimately recommends the burgers. The model struggles to weight the later positive evidence enough.

**Testable fix:** Test hierarchical sentence pooling or attention over discourse segments so later conclusion/recommendation sentences receive appropriate weight.

### Example 12873

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.500755
- **Candidate category:** `near_threshold_error`
- **Error type:** near-threshold lexical ambiguity

**Text:** Disappointing at best- had fresh spring rolls, came wrapped in crepe not rice paper as advertised on menu- had pad Thai- kinda a big glob of noodles with a a few slices of carrot and green onion and no peanut garnish- nota keeper!!

**Analysis:** The review is clearly negative overall, but food-related words and neutral menu descriptions dilute the negative cues. The probability sits almost exactly at the threshold.

**Testable fix:** Tune the decision threshold on validation data and add more short restaurant complaints with similar neutral food vocabulary.

### Example 20213

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.501321
- **Candidate category:** `near_threshold_error`
- **Error type:** mixed sentiment / contrast

**Text:** Oh my god the food is extremely good, the salad bar is one of the healthiest salad and greens u ever going to have in buffet, even though  its a $$ place, still  they charge you the soft drink......... which Is odd,

**Analysis:** The review strongly praises the food and salad bar but includes a pricing complaint about charging for soft drinks. The small negative clause pulls the prediction across an almost 0.5 boundary.

**Testable fix:** Use clause-level sentiment aggregation and calibrate the classification threshold using validation data.

## Slice-Specific Failures

### Example 37771

- **True label:** 0
- **Predicted label:** 1
- **Confidence:** 0.999992
- **Candidate category:** `slice_specific_failure__has_contrast`
- **Error type:** contrast / sentiment reversal

**Text:** I have to say I love cafe rio! Their food is simple yet delicious, the service is great as well as consistent, and my appetite and tummy are always satisfied. However I've been to this location 3 different times and have not experienced any of these.. Even though it may be convenient for many of us here in henderson, please listen to the other reviews and DO NOT EAT AT THIS LOCATION ! New management is highly needed

**Analysis:** The review begins with generic praise for the chain but explicitly rejects this particular location and ends with a strong negative recommendation. The model overweights the positive beginning.

**Testable fix:** Augment with contrast-heavy examples and test position-aware pooling that preserves late review conclusions.

### Example 22707

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999984
- **Candidate category:** `slice_specific_failure__has_contrast`
- **Error type:** weak polarity under contrast

**Text:** Its a nice buffet looking over the Aria pool, but variety was just ok.... Meh..

**Analysis:** The review begins positively ('nice buffet') but ends with 'just ok' and 'Meh'. The sentiment is weak and mixed, and the supplied positive label conflicts with the negative ending.

**Testable fix:** Add weak/mixed sentiment examples and test calibration or sentence-level aggregation for reviews containing contrast markers.

### Example 30793

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999981
- **Candidate category:** `slice_specific_failure__has_contrast`
- **Error type:** temporal sentiment reversal

**Text:** This place is so much better since they changed owners.\n\nMy wife and I went when it was the old owners, it was terrible.  We waited forever and the food never came before we walked out.  People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made more\"".\n\nIt was horrible.\n\nNow its much better.  The staff are very friendly, they treat their customers very well and ...

**Analysis:** The long negative description concerns the old owners, while the current evaluation after the ownership change is repeatedly positive. The model fails to separate past sentiment from present sentiment.

**Testable fix:** Add temporal/discourse-aware examples and test hierarchical modeling that distinguishes earlier background from the final current assessment.

### Example 16532

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999969
- **Candidate category:** `slice_specific_failure__has_contrast`
- **Error type:** contrast / label ambiguity

**Text:** Great gym, but if you ever try to cancel your agreement count on paying 2-3 months longer than expected.  I've been trying to cancel my account because I moved across\nThe country.  They won't cancel it unless I send in a certified letter.  I've done that and am still getting charged months later.

**Analysis:** The positive phrase 'Great gym' is followed by a much longer complaint about cancellation and continued charges. The predicted negative sentiment is consistent with most of the text despite the positive ground-truth label.

**Testable fix:** Audit contrast-heavy labels and test segment-level weighting so model behavior can be compared against the review's concluding sentiment.

### Example 6520

- **True label:** 1
- **Predicted label:** 0
- **Confidence:** 0.999947
- **Candidate category:** `slice_specific_failure__has_contrast`
- **Error type:** aspect-level mixed sentiment

**Text:** its an enjoyable atmosphere for all 21+\n(: The beer is Delicious and so is the food - However I unfortunately, can not say the same about the HELP.The service was terrible the waitress were rude not only  to us but to each other....

**Analysis:** The atmosphere, beer, and food are praised, while service is strongly criticized after 'However'. The example contains competing aspect sentiments, and the model resolves them differently from the dataset label.

**Testable fix:** Test aspect-aware or sentence-level aggregation and add training examples with positive product quality but negative service.

## Overall Findings

The manual review showed four recurring weaknesses. First, contrast-heavy reviews often changed sentiment after words such as `but` or `however`, and the model sometimes over-weighted an earlier clause. Second, long reviews with temporal reversals were difficult when an earlier negative experience was followed by a positive current assessment. Third, sarcasm, slang, and implicit sentiment were harder than explicit polarity cues. Finally, several highly confident errors appear semantically in tension with the supplied label, suggesting that label ambiguity or noisy examples are also present.

The most useful next experiments would therefore be contrast-aware or hierarchical sentence aggregation, targeted augmentation for sentiment reversals and informal language, validation-set threshold calibration, and a small label-audit study for highly confident text/label disagreements.
