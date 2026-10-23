
## 2026-10-23
- Python edit scripts sent through the shell turn `\t` into a real TAB. Build escapes with
  `chr(92) + 't'` and assert `chr(9) not in` the file after editing.
- Spec inputs are always standard normal: `torch.rand()` in a spec is not what the runner feeds,
  constants never reach `ref`, and named intermediates become new inputs.
- Measure every speed change several times; the shared runner is noisy (+-20%), and ptxas
  reacts to small PTX changes (one extra guard add cost a02 25%).
- Where emmy's C and torch disagree (int // and % on negatives, f64 computed in f32), match
  torch: expected outputs are torch's, so emmy's behaviour there can only lose.
- Before refactoring a shared path (value kinds, conversions), save HEAD in a worktree and diff
  the PTX of every test IR: only intended changes should show up.

- **A PASS needs a comparison line.** submit.py checks only the outputs present in expected.npz; a probe
  generated with an empty ref "passes" with exit 0 and no PASS line. Before trusting a result, see
  `[entry] PASS name: max abs err` for the final output.
- **When the output disagrees with torch, check the IR before the compiler.** Recompute the reference
  the way the IR says (e.g. truncated K) - if that matches, the bug is emmy's, and emmy's own reference
  kernel will show it too.

