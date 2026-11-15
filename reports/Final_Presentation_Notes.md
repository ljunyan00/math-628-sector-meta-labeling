# Final presentation speaker notes

**Math 628 · 27 minutes of presentation, then 3 minutes of questions**

| Slide | Time | Speaking points |
| --- | ---: | --- |
| 1. Learning When to Stay Out | 1 min | Introduce the decision facing a trader who already has a momentum signal. Lead with the main result: the image representation gives a small improvement over the temporal CNN at fixed cutoffs, which disappears when opportunity counts match. |
| 2. Research question | 3 min | Define trade selection and meta-labeling. Explain how a high positive-label rate alone can mislead and why we test both probabilities and dollars. Position the study alongside the chart-CNN and recurrence-plot literature. |
| 3. Data and time splits | 4 min | Show the nine sectors, date ranges and sample counts. Explain why opportunities overlap and how we handled the date boundary, including ending annual validation trades within their year and freezing thresholds before 2022. |
| 4. Trade signal and label | 2 min | Walk from signal close to next-open entry and exit. Explain the profit, loss and time barriers, the 5 bps fee on each side, and the one-position-per-ETF rule. |
| 5. Two return sequences become two maps | 3 min | Point to a pixel on the diagonal and then an off-diagonal pixel. Explain standardized return distance and signed product. Explain the fixed peer universe and why the nine-sector heatmap is descriptive. |
| 6. Model comparison and rules | 2 min | Describe the same tabular context for four classifiers and the sequence input difference between CNNs. Explain the constant probability reference and the decision to target 30% development coverage. |
| 7. Forecasts on the holdout | 3 min | Read the log-loss table and point out that lower is better. Explain that the paired interval for the CNN difference spans zero. Discuss the shift in profitable-label frequency from 57.3% to 50.5%. |
| 8. Trading at frozen thresholds | 3 min | Walk through each CNN's candidate coverage, executed trades, exposure, return and Sharpe. Explain why a threshold can accept more holdout trades than it did during development. |
| 9. Results at matched candidate coverage | 2 min | Compare both CNNs at exactly 323 accepted opportunities and explain why accepted and executed counts differ. Mention the retrospective nature of this match and the uncertainty versus random selection. |
| 10. What the experiment supports | 4 min | Explain why cost alone cannot account for the loss, why reduced market exposure matters, and why the maps' simultaneous-sign invariance and one holdout period limit the conclusion. Summarize the evidence and transition to questions. |
| 11–12. Questions and backup | 3 min | Use the cost grid and factor table if asked. |

## Likely questions

**Does the CNN predict returns better than XGBoost?** The 2D CNN had 0.7119 holdout log loss versus 0.7037 for XGBoost and 0.6983 for the training-prevalence reference. The primary comparison between the CNNs yielded a small 2D improvement and uncertainty spanning zero.

**Did lower exposure explain the higher frozen-threshold Sharpe?** The frozen thresholds accepted 41.1% and 48.3% of opportunities for 2D and 1D. Matching both to 323 candidates yields Sharpes of −0.414 and −0.410. Exposure also depends on signal timing and overlapping sleeves, so these results do not identify a causal effect of exposure.

**Why use a peer basket?** Leaving out the target ETF prevents its return from appearing directly in both series. The fixed two-channel input works for all nine target ETFs. The cost is losing detail about individual peer sectors.

**Could we trade the matched top-323 rule live?** This exercise ranks the full holdout's scores and is retrospective. A live-oriented policy uses the probability cutoffs fixed in development.

**How was leakage controlled?** Inputs end at signal close and VIX is lagged one session. Entries are at the next open. Normalization and estimation use the eligible training information in each fold. An 11-session gap and exit-date checks keep outcomes from crossing the relevant refit boundary.

**What is the next research question?** Whether a sign-preserving encoding or a simple exposure benchmark would give more stable conclusions on held-out dates beyond this one interval.
