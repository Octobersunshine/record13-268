import requests
import json
import time

BASE = 'http://127.0.0.1:5000'

print('=' * 70)
print('🧪 过拟合修复验证测试')
print('=' * 70)
print()

print('=== 步骤 1：上传 iris_sample.csv ===')
with open('iris_sample.csv', 'rb') as f:
    r = requests.post(f'{BASE}/api/upload', files={'file': f})
data = r.json()
dataset_id = data['dataset_id']
print(f'数据量: {data["shape"]["rows"]} 行, 数据集ID: {dataset_id[:8]}...')
print()

print('=' * 70)
print('🔴 测试 A：不限制 max_depth（旧行为）')
print('=' * 70)
payload_none = {
    'dataset_id': dataset_id,
    'target_column': 'species',
    'max_depth': '',
    'ccp_alpha': 0.0,
    'min_samples_leaf': 1,
    'test_size': 0.3,
    'random_state': 42
}
r = requests.post(f'{BASE}/api/train', json=payload_none)
result_none = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    print(f'使用 max_depth = {result_none.get("max_depth_used")} (自动推荐)')
    print(f'训练准确率: {result_none["train_accuracy"]:.4f}')
    print(f'测试准确率: {result_none["test_accuracy"]:.4f}')
    print(f'准确率差距: {result_none["accuracy_gap"]:.4f}')
    print(f'树深度: {result_none["tree_structure"]["depth"]}')
    print(f'叶子节点数: {result_none["tree_structure"]["leaf_count"]}')
    print(f'是否有过拟合警告: {result_none.get("has_overfitting_warning", False)}')
    warnings = result_none.get('overfitting_warnings', [])
    if warnings:
        print('过拟合警告信息:')
        for w in warnings:
            print(f'  ⚠️  {w}')
    print()

print('=' * 70)
print('🟡 测试 B：强制 max_depth=1（严重欠拟合参考）')
print('=' * 70)
payload_shallow = {
    'dataset_id': dataset_id,
    'target_column': 'species',
    'max_depth': '1',
    'ccp_alpha': 0.0,
    'min_samples_leaf': 1,
    'test_size': 0.3,
    'random_state': 42
}
r = requests.post(f'{BASE}/api/train', json=payload_shallow)
result_shallow = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    print(f'使用 max_depth = {result_shallow.get("max_depth_used")}')
    print(f'训练准确率: {result_shallow["train_accuracy"]:.4f}')
    print(f'测试准确率: {result_shallow["test_accuracy"]:.4f}')
    print(f'准确率差距: {result_shallow["accuracy_gap"]:.4f}')
    print(f'树深度: {result_shallow["tree_structure"]["depth"]}')
    print(f'是否有过拟合警告: {result_shallow.get("has_overfitting_warning", False)}')
    warnings = result_shallow.get('overfitting_warnings', [])
    if warnings:
        print('过拟合警告信息:')
        for w in warnings:
            print(f'  ⚠️  {w}')
    print()

print('=' * 70)
print('🟢 测试 C：智能推荐 + 剪枝 (ccp_alpha=0.01)')
print('=' * 70)
payload_pruned = {
    'dataset_id': dataset_id,
    'target_column': 'species',
    'max_depth': '',
    'ccp_alpha': 0.01,
    'min_samples_leaf': 3,
    'test_size': 0.3,
    'random_state': 42
}
r = requests.post(f'{BASE}/api/train', json=payload_pruned)
result_pruned = r.json()
print(f'状态码: {r.status_code}')
if r.status_code == 200:
    print(f'使用 max_depth = {result_pruned.get("max_depth_used")}')
    print(f'训练准确率: {result_pruned["train_accuracy"]:.4f}')
    print(f'测试准确率: {result_pruned["test_accuracy"]:.4f}')
    print(f'准确率差距: {result_pruned["accuracy_gap"]:.4f}')
    print(f'树深度: {result_pruned["tree_structure"]["depth"]}')
    print(f'叶子节点数: {result_pruned["tree_structure"]["leaf_count"]}')
    print(f'是否有过拟合警告: {result_pruned.get("has_overfitting_warning", False)}')
    warnings = result_pruned.get('overfitting_warnings', [])
    if warnings:
        print('过拟合警告信息:')
        for w in warnings:
            print(f'  ⚠️  {w}')
    print()

print('=' * 70)
print('📊 修复效果对比')
print('=' * 70)
print(f'{"配置":<35} {"训练Acc":>10} {"测试Acc":>10} {"差距":>10} {"树深度":>8}')
print('-' * 75)
if r.status_code == 200 and result_shallow:
    tr_acc = result_shallow['train_accuracy']
    te_acc = result_shallow['test_accuracy']
    gap = result_shallow['accuracy_gap']
    depth = result_shallow['tree_structure']['depth']
    print(f'{"max_depth=1 (欠拟合)":<35} {tr_acc:>10.4f} {te_acc:>10.4f} {gap:>10.4f} {depth:>8}')
if result_none:
    tr_acc = result_none['train_accuracy']
    te_acc = result_none['test_accuracy']
    gap = result_none['accuracy_gap']
    depth = result_none['tree_structure']['depth']
    print(f'{"自动推荐max_depth (修复前)":<35} {tr_acc:>10.4f} {te_acc:>10.4f} {gap:>10.4f} {depth:>8}')
if result_pruned:
    tr_acc = result_pruned['train_accuracy']
    te_acc = result_pruned['test_accuracy']
    gap = result_pruned['accuracy_gap']
    depth = result_pruned['tree_structure']['depth']
    print(f'{"智能推荐+剪枝 (修复后)":<35} {tr_acc:>10.4f} {te_acc:>10.4f} {gap:>10.4f} {depth:>8}')
print()
print('=' * 70)
print('✅ 过拟合修复验证完成！')
print('=' * 70)
