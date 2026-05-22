# Win Rate Improvement Plan

- Games analyzed: 971
- Baseline win rate: 59.5%

## Top Actions
1. [Play Pattern] Increase early board development and reduce clunky opening hands. (signal score: +19.9, games: 563, confidence: Medium, priority: 100.0)
   - Why: Wins average 7.96 more played events per game.
2. [Decklist] Treat Lucky Dime as a core keep and prioritize lines that deploy it on curve. (signal score: +17.5, games: 122, confidence: High, priority: 100.0)
   - Why: When played, this card improves results by 17.5 points vs baseline loss rate.
3. [Decklist] Treat Tamatoa - So Shiny! as a core keep and prioritize lines that deploy it on curve. (signal score: +26.6, games: 79, confidence: High, priority: 99.5)
   - Why: When played, this card improves results by 26.5 points vs baseline loss rate.
4. [Decklist] Run a 10-game experiment cutting 1-2 copies of Piglet - Pooh Pirate Captain. (signal score: +18.8, games: 54, confidence: High, priority: 87.0)
   - Why: Card shows +18.8 loss-rate lift vs baseline.
5. [Decklist] Run a 10-game experiment cutting 1-2 copies of Cinderella - Ballroom Sensation. (signal score: +13.7, games: 48, confidence: High, priority: 84.0)
   - Why: Card shows +13.7 loss-rate lift vs baseline.
6. [Decklist] Run a 10-game experiment cutting 1-2 copies of Lady - Decisive Dog. (signal score: +13.9, games: 46, confidence: High, priority: 83.0)
   - Why: Card shows +13.9 loss-rate lift vs baseline.
7. [Decklist] Treat Ruby Chromicon as a core keep and prioritize lines that deploy it on curve. (signal score: +26.2, games: 35, confidence: High, priority: 77.5)
   - Why: When played, this card improves results by 26.2 points vs baseline loss rate.
8. [Mulligan] Mulligan Vision of the Future more aggressively in your opener unless hand context is premium. (signal score: +17.9, games: 24, confidence: Medium, priority: 72.0)
   - Why: Opening with this card has +17.9 loss-rate lift.
9. [Play Pattern] Find more lines that keep characters safe to quest repeatedly. (signal score: +6.2, games: 443, confidence: Medium, priority: 71.2)
   - Why: Wins average 2.50 more quest events per game.
10. [Decklist] Run a 10-game experiment cutting 1-2 copies of Christopher Robin - Joining the Fun. (signal score: +11.1, games: 31, confidence: High, priority: 71.2)
   - Why: Card shows +11.1 loss-rate lift vs baseline.
11. [Mulligan] Mulligan Tamatoa - Happy as a Clam more aggressively in your opener unless hand context is premium. (signal score: +14.5, games: 20, confidence: Medium, priority: 70.0)
   - Why: Opening with this card has +14.5 loss-rate lift.
12. [Mulligan] Mulligan Tramp - Enterprising Dog more aggressively in your opener unless hand context is premium. (signal score: +17.9, games: 12, confidence: Medium, priority: 66.0)
   - Why: Opening with this card has +17.9 loss-rate lift.

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