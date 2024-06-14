# DKN

## Performance:
Testrun without entities, default (english) glove embeddings with following parameters.
Dataset=DEMO, History=5, Epochs=2:
```json
{'auc': 0.551, 'group_auc': 0.5441, 'mean_mrr': 0.3385, 'ndcg@5': 0.3769, 'ndcg@10': 0.457}
```

Dataset=SMALL, History=50, Epochs=2:
```json
{'auc': 0.5556, 'group_auc': 0.5484, 'mean_mrr': 0.3386, 'ndcg@5': 0.379, 'ndcg@10': 0.4579}
```

Dataset=LARGE, History=5, Epochs=2:
```json
{'auc': 0.5547, 'group_auc': 0.553, 'mean_mrr': 0.3426, 'ndcg@5': 0.3829, 'ndcg@10': 0.4619}
```
