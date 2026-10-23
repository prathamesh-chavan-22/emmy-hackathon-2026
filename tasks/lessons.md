
## 2026-10-23
- Python edit scripts sent through the shell turn `\t` into a real TAB. Build escapes with
  `chr(92) + 't'` and assert `chr(9) not in` the file after editing.
- Spec inputs are always standard normal: `torch.rand()` in a spec is not what the runner feeds,
  constants never reach `ref`, and named intermediates become new inputs.
- Measure every speed change several times; the shared runner is noisy (+-20%), and ptxas
  reacts to small PTX changes (one extra guard add cost a02 25%).
