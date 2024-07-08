# DKN

## Performance

### With Glove Embeddings and no entities

Testrun without entities, default (english) glove embeddings with following parameters.
Dataset=DEMO, History=5, Epochs=2:
```json
{'auc': 0.551, 'group_auc': 0.5441, 'mean_mrr': 0.3385, 'ndcg@5': 0.3769, 'ndcg@10': 0.457}
```

Dataset=SMALL, History=50, Epochs=2, RUNTIME=12min:
```json
{'auc': 0.5556, 'group_auc': 0.5484, 'mean_mrr': 0.3386, 'ndcg@5': 0.379, 'ndcg@10': 0.4579}
```

Dataset=LARGE, History=5, Epochs=2, RUNTIME=8h51min:
```json
{'auc': 0.5547, 'group_auc': 0.553, 'mean_mrr': 0.3426, 'ndcg@5': 0.3829, 'ndcg@10': 0.4619}
```

Datset=SMALL, History=5, Epochs=10, RUNTIME=1h7min:
```json
{'auc': 0.545, 'group_auc': 0.5407, 'mean_mrr': 0.3359, 'ndcg@5': 0.3734, 'ndcg@10': 0.4546}
```

Datset=SMALL, History=50, Epochs=10, RUNTIME=0h57min:
```json
{'auc': 0.6196, 'group_auc': 0.6129, 'mean_mrr': 0.4146, 'ndcg@5': 0.4557, 'ndcg@10': 0.5228}
```


### With Danish Word2Vec Embeddings and Wikidata Entity Embeddings

Datset=SMALL, History=50, Epochs=10, RUNTIME=:
```json
{'auc': 0.6834, 'group_auc': 0.6798, 'mean_mrr': 0.4783, 'ndcg@5': 0.5241, 'ndcg@10': 0.5789}
```
