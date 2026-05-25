# Win Rate Improvement Plan

- Games analyzed: 1137
- Baseline win rate: 59.6%

## Top Actions
1. [Play Pattern] Increase early board development and reduce clunky opening hands. (signal score: +30.9, games: 723, confidence: Medium, priority: 100.0)
   - Why: Wins average 12.36 more played events per game.
2. [Decklist] Treat Lucky Dime as a core keep and prioritize lines that deploy it on curve. (signal score: +19.7, games: 160, confidence: High, priority: 100.0)
   - Why: When played, this card improves results by 19.7 points vs baseline loss rate.
3. [Decklist] Treat Tamatoa - So Shiny! as a core keep and prioritize lines that deploy it on curve. (signal score: +27.6, games: 110, confidence: High, priority: 100.0)
   - Why: When played, this card improves results by 27.6 points vs baseline loss rate.
4. [Decklist] Run a 10-game experiment cutting 1-2 copies of Piglet - Pooh Pirate Captain. (signal score: +17.2, games: 66, confidence: High, priority: 93.0)
   - Why: Card shows +17.2 loss-rate lift vs baseline.
5. [Decklist] Treat Ruby Chromicon as a core keep and prioritize lines that deploy it on curve. (signal score: +19.2, games: 52, confidence: High, priority: 86.0)
   - Why: When played, this card improves results by 19.2 points vs baseline loss rate.
6. [Play Pattern] Find more lines that keep characters safe to quest repeatedly. (signal score: +8.8, games: 564, confidence: Medium, priority: 84.1)
   - Why: Wins average 3.53 more quest events per game.
7. [Mulligan] Mulligan Vision of the Future more aggressively in your opener unless hand context is premium. (signal score: +18.0, games: 24, confidence: Medium, priority: 72.0)
   - Why: Opening with this card has +18.0 loss-rate lift.
8. [Decklist] Run a 10-game experiment cutting 1-2 copies of Webby Vanderquack - Junior Prospector. (signal score: +11.2, games: 31, confidence: High, priority: 71.7)
   - Why: Card shows +11.2 loss-rate lift vs baseline.
9. [Decklist] Run a 10-game experiment cutting 1-2 copies of Kit Cloudkicker - Tough Guy. (signal score: +12.0, games: 21, confidence: Medium, priority: 70.5)
   - Why: Card shows +12.0 loss-rate lift vs baseline.
10. [Mulligan] Mulligan Tamatoa - Happy as a Clam more aggressively in your opener unless hand context is premium. (signal score: +14.6, games: 20, confidence: Medium, priority: 70.0)
   - Why: Opening with this card has +14.6 loss-rate lift.
11. [Decklist] Run a 10-game experiment cutting 1-2 copies of Lilo - Escape Artist. (signal score: +12.3, games: 19, confidence: Medium, priority: 69.5)
   - Why: Card shows +12.3 loss-rate lift vs baseline.
12. [Mulligan] Mulligan Tramp - Enterprising Dog more aggressively in your opener unless hand context is premium. (signal score: +18.0, games: 12, confidence: Medium, priority: 66.0)
   - Why: Opening with this card has +18.0 loss-rate lift.

## 4-Week Research Loop
1. Pick the top 2 decklist actions and top 1 sequencing action from this report.
2. Run two 10-15 game blocks: Block A with current deck, Block B with one targeted change.
3. Keep mulligan policy fixed during each block to avoid confounding.
4. Re-run analyze.py after each block and compare only the changed metrics.
5. Keep changes that improve win rate and also improve at least one process metric (first quest turn, early ink, or opener loss lift).

## Simulation and Modeling
- Use Simulation Lab for pre-screening candidate cuts/adds before spending games on them.
- Use stress tests on historical losses to spot recurring matchup failures.
- Loss model: not enough stable samples yet for feature-importance guidance.

## Interpretation Guardrails
- Correlation is directional, not proof of causation.
- Prefer high-priority actions with larger sample sizes.
- Revert changes quickly if they fail to improve both win rate and process metrics.