import requests
import json

BASE = 'http://127.0.0.1:5000'

print('=== 1. 测试首页 ===')
r = requests.get(BASE)
print(f'状态码: {r.status_code}, 内容长度: {len(r.text)}')

print()
print('=== 2. 上传 iris_sample.csv ===')
with open('iris_sample.csv', 'rb') as f:
    r = requests.post(f'{BASE}/api/upload', files={'file': f})
print(f'状态码: {r.status_code}')
data = r.json()
dataset_id = data['dataset_id']
print(f'Dataset ID: {dataset_id}')
print(f'数据形状: {data["shape"]}')
print(f'列名: {data["columns"]}')

print()
print('=== 3. 训练模型 ===')
payload = {
    'dataset_id': dataset_id,
    'target_column': 'species',
    'test_size': 0.2,
    'criterion': 'gini',
    'random_state': 42
}
r = requests.post(f'{BASE}/api/train', json=payload)
print(f'状态码: {r.status_code}')
result = r.json()
if r.status_code == 200:
    model_id = result['model_id']
    print(f'Model ID: {model_id}')
    print(f'训练集准确率: {result["train_accuracy"]}')
    print(f'测试集准确率: {result["test_accuracy"]}')
    print(f'树深度: {result["tree_structure"]["depth"]}')
    print(f'叶子节点数: {result["tree_structure"]["leaf_count"]}')
    print(f'特征重要性: {json.dumps(result["feature_importance"], indent=2, ensure_ascii=False)}')
else:
    print(f'错误: {result}')

print()
print('=== 4. 预测新样本 ===')
samples = [
    {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
    {'sepal_length': 6.0, 'sepal_width': 2.7, 'petal_length': 5.1, 'petal_width': 1.6},
    {'sepal_length': 7.2, 'sepal_width': 3.2, 'petal_length': 6.0, 'petal_width': 1.8}
]
r = requests.post(f'{BASE}/api/predict', json={
    'model_id': model_id,
    'samples': samples
})
print(f'状态码: {r.status_code}')
pred = r.json()
if r.status_code == 200:
    for i, p in enumerate(pred['predictions']):
        print(f'样本{i+1}: 预测类别 = {p["prediction"]}, 概率 = {json.dumps(p["probabilities"], ensure_ascii=False)}')
else:
    print(f'错误: {pred}')

print()
print('=== 5. 获取模型列表 ===')
r = requests.get(f'{BASE}/api/models')
print(f'状态码: {r.status_code}')
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

print()
print('=== 6. 获取数据集列表 ===')
r = requests.get(f'{BASE}/api/datasets')
print(f'状态码: {r.status_code}')
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

print()
print('✅ 所有测试通过！')
