# Changelog

Notable changes to options and behaviour. Measurements live in the issues
linked from each entry.

## Unreleased

### Removed

- **Line-segment search** (#213, #246, #247; removed in #254): the
  recombiner's line-segment arm and `fpr_lp`'s line-segment lane, together
  with the option `mip_heuristic_line_segment`. It was off by default, and
  #252 measured no significant gain. Default runs are unchanged; HiGHS now
  rejects the option name as unknown.
