import requests
import json

BASE = 'http://127.0.0.1:5000'

print('=' * 70)
print('📊 特征重要性功能验证测试')
print('=' * 70)
print()

print('=== 步骤 1：上传数据并训练模型 ===')
with open('iris_sample.csv', 'rb') as f:
    r = requests.post(f'{BASE}/api/upload', files={'file': f})
data = r.json()
dataset_id = data['dataset_id']
print(f'数据集上传成功，样本数: {data["shape"]["rows"]}')

train_payload = {
    'dataset_id': dataset_id,
    'target_column': 'species',
    'max_depth': '6',
    'ccp_alpha': 0.0,
    'test_size': 0.3,
    'random_state': 42
}
r = requests.post(f'{BASE}/api/train', json=train_payload)
train_result = r.json()
model_id = train_result['model_id']
print(f'模型训练成功，ID: {model_id[:10]}...')
print(f'训练准确率: {train_result["train_accuracy"]:.4f}')
print(f'测试准确率: {train_result["test_accuracy"]:.4f}')
print()

print('=' * 70)
print('🔍 测试 1：训练接口返回增强特征重要性')
print('=' * 70)
if 'feature_importance_enhanced' in train_result:
    fi = train_result['feature_importance_enhanced']
    print(f'✅ 包含增强特征重要性字段')
    print(f'   非零特征数: {fi.get("non_zero_count", "N/A")}')
    print(f'   Top 特征: {fi.get("top_features", "N/A")}')
    print(f'   无贡献特征: {fi.get("zero_features", "N/A")}')
    print()
    print('   详细排序特征重要性:')
    for f in fi.get('sorted', []):
        level_map = {'critical': '🔴核心', 'high': '🟠重要', 'medium': '🔵中等', 'low': '🟣微弱', 'none': '⚫无'}
        level = level_map.get(f.get('importance_level', ''), '')
        print(f'     #{f["rank"]:2d} {f["feature"]:15s} {f["importance_percent"]:6.2f}%  {level}  累计: {f["cumulative_percent"]:6.2f}%')
else:
    print('❌ 缺少增强特征重要性字段')
print()

print('=' * 70)
print('🔍 测试 2：独立特征重要性 API - 完整格式')
print('=' * 70)
r = requests.get(f'{BASE}/api/models/{model_id}/feature_importance')
fi_full = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    print(f'目标列: {fi_full["target_column"]}')
    fi = fi_full['feature_importance']
    print(f'总特征数: {len(fi["sorted"])}')
    print(f'非零贡献: {fi["non_zero_count"]} 个特征')
print()

print('=' * 70)
print('🔍 测试 3：独立特征重要性 API - sorted 格式 (top_n=3)')
print('=' * 70)
r = requests.get(f'{BASE}/api/models/{model_id}/feature_importance?format=sorted&top_n=3')
fi_sorted = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    for f in fi_sorted['feature_importance']['features']:
        print(f'  {f["rank"]}. {f["feature"]}: {f["importance_percent"]}% ({f["importance_level"]})')
print()

print('=' * 70)
print('🔍 测试 4：独立特征重要性 API - summary 格式')
print('=' * 70)
r = requests.get(f'{BASE}/api/models/{model_id}/feature_importance?format=summary')
fi_summary = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    s = fi_summary['feature_importance']
    print(f'  非零特征数: {s["non_zero_count"]}')
    print(f'  Top 特征: {s["top_features"]}')
    print(f'  无贡献特征: {s["zero_features"]}')
    print(f'  总重要性: {s["total_importance"]}')
print()

print('=' * 70)
print('🔍 测试 5：预测接口 + 特征贡献度分析')
print('=' * 70)
samples = [
    {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
    {'sepal_length': 6.0, 'sepal_width': 2.7, 'petal_length': 5.1, 'petal_width': 1.6}
]
r = requests.post(f'{BASE}/api/predict', json={
    'model_id': model_id,
    'samples': samples,
    'return_contribution': True
})
pred_result = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    print(f'预测样本数: {pred_result["prediction_count"]}')
    if 'feature_importance_summary' in pred_result:
        s = pred_result['feature_importance_summary']
        print(f'模型关键特征: {s["top_features"]}')
    print()
    for i, p in enumerate(pred_result['predictions']):
        print(f'--- 样本 #{i+1} ---')
        print(f'  预测类别: {p["prediction"]}')
        print(f'  各类别概率: {json.dumps(p["probabilities"], ensure_ascii=False)}')
        if 'feature_contribution' in p:
            fc = p['feature_contribution']
            print(f'  决策深度: {fc["decision_depth"]} 层')
            print(f'  叶子节点: #{fc["leaf_id"]} (包含 {fc["leaf_samples"]} 个训练样本)')
            print(f'  特征贡献:')
            for pc in fc.get('prediction_contribution', []):
                print(f'    • {pc["decision"]} ({pc["contribution_percent"]}%)')
            print(f'  完整决策路径:')
            path_str = ' → '.join([
                f'{dp["feature"]} {dp["direction"]} {dp["threshold"]}'
                for dp in fc.get('decision_path', [])
            ])
            print(f'    {path_str}')
        print()

print('=' * 70)
print('🔍 测试 6：普通预测（不返回贡献度）')
print('=' * 70)
r = requests.post(f'{BASE}/api/predict', json={
    'model_id': model_id,
    'samples': samples,
    'return_contribution': False
})
pred_simple = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    p = pred_simple['predictions'][0]
    print(f'预测: {p["prediction"]}')
    print(f'包含 feature_contribution: {"feature_contribution" in p}')
print()

print('=' * 70)
print('✅ 特征重要性功能验证测试完成！')
print('=' * 70)
