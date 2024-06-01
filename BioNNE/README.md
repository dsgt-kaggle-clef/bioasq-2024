1. Run model

```
python en_model.py --save-prediction
```

2. Inspect the output jsonl file prediction.
For example

```
grep pair output/en_1711833816.jsonl | jq .
```


