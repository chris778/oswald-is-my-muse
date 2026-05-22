# Lorcana Game Knowledge for Analytics

This file provides domain knowledge so that analysis and AI-assisted insights can be interpreted correctly in the context of Disney Lorcana strategy. Update this file as your understanding of the meta evolves.

---

## Core objective and win condition

- First player to accumulate **20 lore** wins.
- Lore is gained by **questing** characters (exerting them during your turn to gain lore equal to their lore value).
- Games also end by **concession** or timeout in digital play.
- This is fundamentally a **lore race**, not a combat game — eliminating opponent characters is a means to slow their lore gain, not a win condition in itself.

---

## Turn structure

Each turn in order:
1. **Ready** — unexert all your characters
2. **Set** — no automatic effects currently trigger here
3. **Draw** — draw one card (OTD player draws an extra card on turn 1)
4. **Main phase** — in any order:
	- **Ink** one card per turn (place it face-down as an ink resource — this uses up that card)
	- **Play** characters/actions/items by spending ink equal to their cost
	- **Quest** with ready characters (exerts them, gains lore)
	- **Challenge** exerted opponent characters with your ready characters

Key constraint: **characters cannot quest or challenge the turn they are played** (summoning sickness, called "drying" in Lorcana — the card must be in play at the start of your turn to act).

---

## OTP vs OTD dynamics

- **OTP (On The Play)** = going first. You act before the opponent every turn. Tempo advantage — your characters dry one turn earlier relative to the opponent's.
- **OTD (On The Draw)** = going second. You draw one extra card on turn 1. Card advantage — starts with 8-card effective opening to OTP's 7.
- In aggressive/tempo decks, OTP is typically stronger. In slower/control decks, OTD's extra card may matter more.
- Most decks in the current meta favor OTP slightly because early board presence and early questing is powerful.

---

## Ink efficiency and the mana curve

- Inking is a **cost** — you sacrifice a card from hand to gain an ink resource.
- **Inkable cards** can be inked; **non-inkable cards** cannot (they must be played, never inked).
- A well-constructed deck inks on curve (turn 1, 2, 3, 4...) so it can play its full-cost cards when they matter.
- **Missing an ink** (no inkable card in hand) is a significant tempo loss.
- **Inking the wrong card** (inking a card you needed to play) is a hidden cost visible in "loss rate when inked" data.
- The early ink rate metric (turns 1–4) measures consistency: high ink rate in wins = you were hitting your curve.

---

## Card types and strategic roles

### Characters
- Have a cost (ink), strength (for challenging), willpower (health), and lore value.
- High-lore characters (2–3 lore) are priority targets to challenge because they generate lore fast.
- Cheap characters (1–2 cost) establish early board presence and enable early questing.
- Expensive characters (5–7 cost) are usually game-ending threats or have powerful abilities.

### Actions
- Instant effects — played and go to the discard pile.
- Can remove characters, generate lore, draw cards, or disrupt the opponent.
- Actions that remove characters ("removal") are critical for slowing opponent lore.

### Items
- Persist on the board like characters but cannot quest or challenge.
- Usually provide ongoing value (draw, ink generation, recurring effects).

---

## Key strategic concepts

### Tempo
Playing cards that impact the board immediately. A "tempo play" advances your board while setting back the opponent's. Going behind on tempo means your characters are being challenged off while theirs quest freely.

### Board control vs lore racing
Two viable strategies:
- **Contest the board**: challenge opponent's questing characters to keep their lore low while yours quest safely.
- **Race**: ignore the opponent's board and quest as fast as possible, accepting challenges.

Most games involve both, with the balance shifting based on board state.

### Challenge timing
- You can only challenge **exerted** characters.
- If the opponent quested with a character, it's exerted — you can challenge it on your next turn.
- Challenging off high-lore characters before they quest again is high-value. Challenging low-lore characters is often tempo-negative.

### Mulligan strategy
- Players can mulligan their opening hand once (return any number, draw back to 7).
- A good opening hand has: an inkable 1- or 2-cost card to ink turn 1, a low-cost play for turn 2–3, ideally a "curve" setup.
- High-cost cards in an opener (5+ cost) are usually bad unless the deck is specifically built for them.

---

## Archetypes in this dataset

### Amethyst/Steel
- **Style**: Tempo / board control. Amethyst provides card advantage and disruption; Steel provides cheap efficient characters and removal.
- **Key win condition**: Early board with cheap characters, challenge off threats, quest safely once ahead.
- **Key cards seen in data**: Madam Mim - Snake (cheap body + ability), Pete - Games Referee (board threat), John Smith - Skillful Explorer, Will o' the Wisp - Forest Spirit, Diablo - Obedient Raven, Merlin - Goat (high lore), Doc - Bold Knight, Fire the Cannons!, Friends on the Other Side.
- **Critical turns**: Turn 2–3 board presence is essential. Missing your turn-3 play is a strong predictor of loss.
- **Weakness**: Can run out of gas in long games; losing board early is hard to recover from.

### Ruby/Sapphire
- **Style**: Midrange / engine. Ruby provides aggression and challenge tools; Sapphire provides card draw and value.
- **Key win condition**: Build an engine through draw (Sapphire tools like Fishbone Quill, Scrooge McDuck) and use Ruby threats to close.
- **Key cards seen in data**: Sisu - Daring Visitor / Sisu - Empowered Sibling, Belle - Apprentice Inventor, Oswald - The Lucky Rabbit, Pawpsicle, Maurice's Workshop, Sapphire Coil, Brawl, Unconventional Tool.
- **Critical turns**: Establishing draw engine by turns 3–4. Sisu as a tempo swing.

### Amber/Emerald
- **Style**: Aggro / lore rush. Amber has sticky low-cost characters; Emerald has evasive (Rush/Evasive) threats.
- **Key win condition**: Go fast. Play multiple low-cost characters, quest every turn, challenge anything that blocks the race.
- **Key cards seen in data**: Lady - Decisive Dog, Lady - Miss Park Avenue, Tramp - Enterprising Dog, Go Go Tomago, Nani - Stage Manager, Ursula - Deceiver, Rapunzel - Gifted with Healing.
- **Critical turns**: Must be questing by turn 2–3. Slow starts are nearly unrecoverable.

### Emerald/Sapphire
- **Style**: Control / value. Both colors provide card draw and board manipulation.
- **Key win condition**: Out-resource opponent over long game. Use removal to stay ahead, then close with high-lore threats.
- **Key cards seen in data**: Kit Cloudkicker, Belle - Apprentice Inventor, Scrooge McDuck, Tamatoa, Maurice's Workshop, Under the Sea, Ursula - Deceiver.

### Amber/Steel
- **Style**: Songs / combo. Amber provides song synergies; Steel provides cheap singers and value.
- **Key win condition**: Chain songs (actions) using characters as "singers" to get massive value, then out-tempo on lore.
- **Key cards seen in data**: Ariel variants, Cinderella - Ballroom Sensation, Pete - Games Referee, Rapunzel - Gifted with Healing, A Whole New World, Let the Storm Rage On.

---

## How replay events map to strategy

| Event type | What it means | Strategic signal |
|---|---|---|
| `CARD_PLAYED` | Played a card from hand | Board development, curve execution |
| `CARD_INKED` | Used a card as ink resource | Curve consistency; high ink rate = good curve |
| `CARD_QUEST` | Quested a character for lore | Lore generation speed; earlier = better |
| `CARD_ATTACK` | Challenged an opponent character | Board control; too many in losses = getting into unfavorable fights |
| `ABILITY_ACTIVATED` | Used a character or item ability | Varies by ability |
| `INITIAL_HAND` | Opening hand cards | Starting conditions |
| `MULLIGAN` | Mulligan count | Hand quality signal |

---

## Interpreting the analytics correctly

### Card-loss correlation
A card with high "loss lift" appearing in your games is associated with a higher loss rate. This can mean:
- The card itself is weak or doesn't fit the deck
- **Or** the card only shows up late in games (because you couldn't play it earlier), and losing games run longer — so you see the card more in losses by accident
- Always cross-reference with "avg turn played" and "play-through rate" before cutting a card

### "Loss lift when played" vs "loss lift when drawn"
- High **loss rate when drawn** but low **loss rate when played**: The card is fine when you play it, but seeing it in certain games (board states) correlates with losing. Probably not the card's fault.
- High **loss rate when played**: Playing this card specifically correlates with losing. Stronger cut signal.
- High **loss rate when inked**: You're being forced to ink it when you shouldn't, or the games where you ink it are losing ones.

### Card timing (avg turn played in wins vs losses)
- If you play a card **earlier in wins**: It's a high-value early play. Build your deck and sequencing around enabling it early.
- If you play it **later in losses**: Either you're drawing it late in losing games (long games) or you're not prioritizing it correctly.
- Cards played at the same average turn in wins and losses: timing-neutral, win rate depends on other factors.

### First quest turn
- Questing on turn 2 is the gold standard for most decks.
- Questing on turn 3 is acceptable but slightly behind.
- Turn 4+ first quest is a significant tempo deficit — likely means you had no board or were challenged off early characters.

### Mulligan data
- Taking 0 mulligans in a winning game means you kept a strong hand.
- High mulligan counts in losses could mean you kept bad hands, or the deck has consistency problems.
- Compare OTP vs OTD mulligan rates — going first may incentivize keeping lower-curve hands.

---

## Meta context (as of May 2026 — update as needed)

- Set 12 ("Infinity") is the current set.
- Quick Play uses a BO1 format; Infinity BO3 uses best-of-three with potential sideboard.
- In BO3, sideboard options and game 2/3 adjustments matter — card performance may differ between game 1 (fresh) and games 2–3 (post-sideboard knowledge).
- The current meta appears to feature a mix of tempo (Amethyst/Steel), aggro (Amber/Emerald), and midrange (Ruby/Sapphire) archetypes based on this dataset.
