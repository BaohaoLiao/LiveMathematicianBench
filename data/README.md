# livemath-v7-0316

Four-month LiveMath package built from the `v5 -> ge5 -> hard` pipeline.

Months included:

- `202511`: full=`337`, ge5=`141`, hard=`29`, hard_acc=`0.4138`
- `202512`: full=`472`, ge5=`213`, hard=`52`, hard_acc=`0.2885`
- `202601`: full=`403`, ge5=`192`, hard=`46`, hard_acc=`0.3913`
- `202602`: full=`390`, ge5=`188`, hard=`50`, hard_acc=`0.4000`

Overall hard-set accuracy on `gpt-5.4_2026-03-05` `medium`:

- `65/177 = 0.3672`

Per-month layout:

- `full/qaEval_<month>_full.json`
- `ge5/qaEval_<month>_ge5.json`
- `hard/qaEval_<month>_ge5_hard.json`
- `hard/accuracy_test_<month>_medium_filter2.json`
- `hard/selected_hard_ge5_filter1.json`
- `summary.json`
