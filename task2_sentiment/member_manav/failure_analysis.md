# Task 2 Failure Analysis: Sentiment Classification on Yelp Polarity

All 20 rows below are copied from `outputs/gpu/error_review.csv` (real model errors on the test set, from the `bilstm_attn` model). Error types were assigned by me after reading each full review. Note: the review text in `error_review.csv` is shortened to 300 characters, so each label is based on the visible (shortened) text plus the top attention words, not necessarily the full original review.

True label: 0 = negative, 1 = positive. Prob = predicted probability of positive.

## Confident False Positives (true label negative, model very sure it's positive)

| id | review (shortened) | label | prob | slice | error type |
|---|---|---|---|---|---|
| 29330 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try! Cheers, Eric Van Nguyen Visited April 2012" | 0 | 1.000 | short | **label noise** — the text itself reads fully positive, with no negative word anywhere |
| 9565 | "Not a very fun place to stay. The place is inexpensive and feels that way... It's a great room if you're just planning to sleep there, and nothing else... The shows at the LVH are a hit or miss..." | 0 | 0.999 | medium, has_negation | **mixed sentiment / contrast** — starts negative, but several positive phrases ("great room", "enjoy the rest of your vacation") follow |
| 408 | "Though I'm a Copper enthusiast... I'd heard that Maharani was a cheaper but tasty option... Copper is definitely still my place, but Maharani was fine enough. First of all, the food came very quickly..." | 0 | 0.999 | medium, has_negation, has_contrast | **mixed sentiment / contrast** — directly compares two restaurants with "though"/"but" |
| 18122 | "20 years for me and sad to see them go. Sadder still to see a great traditional Japanese sushi restaurant end, and not carry on the tradition. We would go here for consistently great fresh fish and great service..." | 0 | 0.999 | medium, has_negation, has_contrast | **mixed sentiment / contrast** — the review is sad about a closing, but uses many positive words to describe how good the place used to be |
| 4488 | "It's good. The rolls are better than the sashimi although one time we had some really nice(and surprising) maguro... It's still enjoyable. Good atmosphere and my girl likes..." | 0 | 0.999 | medium, has_negation, has_contrast | **mixed sentiment / contrast** — visible text reads almost entirely positive |

## Confident False Negatives (true label positive, model very sure it's negative)

| id | review (shortened) | label | prob | slice | error type |
|---|---|---|---|---|---|
| 16148 | "The employees at this Target seemed unusually friendly during my last visit. Two of them asked if they could help me find anything, the cashier was perky and personable... Maybe I've gotten too used t..." | 1 | 0.001 | medium, has_negation, has_contrast | **mixed sentiment / contrast** — true label is positive; "unusually friendly" and "maybe I've gotten too used to..." are hedging words, not sarcasm |
| 22807 | "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server..." | 1 | 0.001 | medium, has_negation, has_contrast | **label noise** — the visible text is strongly negative ("Horrible service", "altercation") for a review labeled positive; possible the original (pre-edit) review was positive and only the edit note is negative |
| 19078 | "I was so looking forward our dinner here... Few hours prior to our dinner, I ended up having the worst stomach ache and I couldn't even try anything, other then a bite of caviar(YUMMO)..." | 1 | 0.001 | medium, has_negation | **mixed sentiment / contrast** — the bad part (stomach ache) is about the reviewer's health, not the restaurant; "YUMMO" signals she liked what she did taste |
| 30793 | "This place is so much better since they changed owners. My wife and I went when it was the old owners, it was terrible. We waited forever and the food never came before we walked out..." | 1 | 0.001 | medium, has_negation, has_contrast | **mixed sentiment / contrast** — classic before/after structure; most of the visible text describes the bad old experience even though the current place is rated positive |
| 11808 | "It was Anniversary time! But we didn't' want to spend a ton of money... Don't get me wrong: This place is TINY! Narrow walk ways and close quarters in general. Even the exhibits are small. I do hop..." | 1 | 0.001 | long, has_negation, has_contrast | **mixed sentiment / contrast** — many hedges and caveats ("don't get me wrong", "TINY", "small") stacked before any positive payoff |

## Near-Threshold Errors (prediction landed right around 0.5)

| id | review (shortened) | label | prob | slice | error type |
|---|---|---|---|---|---|
| 17826 | "Great pizza... The employees were nice. The decor is interesting and fitting. The pasta servings is enough to feed at least two people. Might be much for one..." | 1 | 0.500 | medium, has_negation | **mixed sentiment / contrast** — the top attention words ("minut", "not") point to a complaint later in the review, past the shortened text shown here |
| 9722 | "This place is all about the hotel and the shops. It's a great place to visit and have fun. They are more expensive than a lot of the other casinos. The Bourbon Room (bar) is a total rip off..." | 1 | 0.500 | medium, has_negation | **mixed sentiment / contrast** — positive about the hotel/shops overall, but a specific sharp complaint about the bar |
| 33202 | "For a long time, On the Roxx was the only place to go... Good thing times have changed. I've been here a several times... was never impressed... I was a patron by necessity. It's hidden behind Johnston Road and rightfully so..." | 0 | 0.500 | medium, has_negation, has_contrast | **sarcasm** — "rightfully so" (about being hidden) and "patron by necessity" read as backhanded/ironic |
| 34933 | "AVOID this company. Very slow service. I would give them 0 stars. If you want to get to the airport on time you're better off taking a cab. Must book return trip a day in advance. Not refunded for unused return trip..." | 0 | 0.501 | short, has_negation | **negation** — text is clearly very negative ("AVOID", "0 stars", "Not refunded"), yet the model landed right at the decision boundary; heavy stacked negation may be confusing it |
| 34984 | "I have visited quite a few nail salons... my friend... recommended it... Similar to other experiences we went to the nail bar and I had a gentleman ask what service I wanted and like usual, I asked..." | 0 | 0.501 | long, has_negation | **long review cut at max_len** — visible text is neutral/positive in tone; the attention word "hiccup" hints the real complaint may come later in the review, possibly past where the model's 256-word limit cuts in |

## Slice Errors (short reviews)

| id | review (shortened) | label | prob | slice | error type |
|---|---|---|---|---|---|
| 22962 | "Pros: Broad menu selection. Cons: Unrealistic beer prices... At $6.50 per (imperial) pint, it's the most expensive Guinness in Charlotte." | 0 | 0.593 | short | **mixed sentiment / contrast** — literally written as "Pros" and "Cons" |
| 24888 | "Lovely place in the middle of nowhere. Bottom floor is the bar and the second floor is a game area. Appetizers were good and cold beer." | 1 | 0.333 | short | **other** — review is actually all positive; the word "nowhere" (meant charmingly, as "secluded") may read as negative to the model out of context |
| 36963 | "God love em, they have tried. The casino still is under construction, the rooms still....well...from the 80s and the machines tight as bell but still cheap and a place to lay your head." | 0 | 0.976 | short, has_contrast | **sarcasm** — "God love em, they have tried" is a resigned, backhanded opener, and the rest is faint praise |
| 5329 | "A small coffee is now $2.50 Lol." | 0 | 0.666 | short | **other** — too short, no real sentiment words beyond "Lol" to base a category on |
| 7719 | "Terrible coffee. Like real bad. My friend's tres leches looked solid. And it offered a good view of the cheesy gondoliers at the Venetian." | 0 | 0.528 | short | **mixed sentiment / contrast** — the actual complaint (coffee) is negative, but the surrounding details (dessert, view) are positive/neutral |

## Error Type Counts

| Error type | Count |
|---|---|
| mixed sentiment / contrast | 12 |
| sarcasm | 2 |
| label noise | 2 |
| other | 2 |
| negation | 1 |
| long review cut at max_len | 1 |
| **Total** | **20** |

## Testable Fix

The most common error type is **mixed sentiment / contrast** (12 of 20, well over half the errors). This matches the per-slice metrics in `results.md`: `has_contrast` is the weakest slice for every model (e.g. baseline F1 0.9189, the lowest of any slice).

**Fix to try:** when a review contains a contrast word ("but", "however", "although", "though"), give more weight to the clause **after** the contrast word when building the review's representation (for example, a simple weighted average that upweights words after "but" instead of treating the whole review equally).

**How to test it:** retrain (or re-score) and check that **`has_contrast` macro-F1 improves**, while overall test accuracy does not drop. If `has_contrast` F1 goes up and overall accuracy stays about the same or better, the fix is working. If overall accuracy drops a lot, the extra weight on the "but" clause is probably too strong and is hurting reviews that don't actually flip sentiment after "but".
