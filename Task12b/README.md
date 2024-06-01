Run model:

`python model.py --input 11B2_golden --stage A`

`python model.py --input 11B2_golden --stage B`

Run validation code to get the metrics. Noted that the input model should contain the golden answers:

`python model.py --input 11B2_golden --stage A --validate`

`python model.py --input 11B2_golden --stage B --validate`