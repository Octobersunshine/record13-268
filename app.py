import os
import pickle
import tempfile
import uuid
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.mkdtemp(prefix='dt_uploads_')
MODEL_FOLDER = tempfile.mkdtemp(prefix='dt_models_')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

model_store = {}
data_store = {}


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def compute_enhanced_feature_importance(clf, feature_cols):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = []
    cumulative = 0.0
    for rank, idx in enumerate(indices, 1):
        feat_name = feature_cols[idx]
        importance = float(importances[idx])
        cumulative += importance

        if importance >= 0.5:
            level = 'critical'
        elif importance >= 0.2:
            level = 'high'
        elif importance >= 0.05:
            level = 'medium'
        elif importance > 0:
            level = 'low'
        else:
            level = 'none'

        sorted_features.append({
            'rank': rank,
            'feature': feat_name,
            'importance': importance,
            'importance_percent': round(importance * 100, 2),
            'cumulative_importance': round(cumulative, 6),
            'cumulative_percent': round(cumulative * 100, 2),
            'importance_level': level
        })

    non_zero = [f for f in sorted_features if f['importance'] > 0]
    zero_features = [f['feature'] for f in sorted_features if f['importance'] == 0]

    return {
        'raw': dict(zip(feature_cols, importances.tolist())),
        'sorted': sorted_features,
        'non_zero_count': len(non_zero),
        'zero_features': zero_features,
        'top_features': [f['feature'] for f in non_zero[:5]] if non_zero else [],
        'total_importance': round(float(np.sum(importances)), 6)
    }


def compute_sample_feature_contribution(clf, feature_cols, sample, label_encoders=None):
    tree = clf.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value

    if hasattr(sample, 'iloc'):
        x_sample = sample
    else:
        x_sample = pd.DataFrame([sample], columns=feature_cols)

    x_sample_encoded = x_sample.copy()
    if label_encoders:
        for col in feature_cols:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    x_sample_encoded[col] = le.transform([str(x_sample_encoded[col].iloc[0])])[0]
                except:
                    pass

    node_indicator = clf.decision_path(x_sample_encoded)
    leaf_id = clf.apply(x_sample_encoded)[0]

    sample_id = 0
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]
    ]

    path_features = []
    for node_id in node_index:
        if leaf_id == node_id:
            continue
        feat_idx = feature[node_id]
        if feat_idx < 0 or feat_idx >= len(feature_cols):
            continue
        feat_name = feature_cols[feat_idx]
        orig_val = x_sample.iloc[0, feat_idx]
        thresh = threshold[node_id]

        if children_left[node_id] != children_right[node_id]:
            encoded_val = x_sample_encoded.iloc[0, feat_idx]
            if encoded_val <= thresh:
                direction = '≤'
                next_node = children_left[node_id]
            else:
                direction = '>'
                next_node = children_right[node_id]

            node_value = value[node_id]
            if node_value.shape[0] > 0:
                class_counts = node_value[0].astype(int)
                total = class_counts.sum()
            else:
                class_counts = []
                total = 0

            path_features.append({
                'feature': feat_name,
                'value': float(orig_val) if isinstance(orig_val, (np.floating, np.integer)) else orig_val,
                'threshold': round(float(thresh), 4),
                'direction': direction,
                'node_id': int(node_id),
                'next_node': int(next_node),
                'node_samples': int(total),
                'class_distribution': class_counts.tolist() if len(class_counts) > 0 else []
            })

    leaf_value = value[leaf_id][0]
    leaf_total = leaf_value.sum()

    prediction_contribution = []
    for feat in path_features:
        prediction_contribution.append({
            'feature': feat['feature'],
            'value': feat['value'],
            'decision': f"{feat['feature']} = {feat['value']} {feat['direction']} {feat['threshold']}",
            'contribution_percent': round(100.0 / len(path_features), 1) if path_features else 0
        })

    return {
        'decision_path': path_features,
        'prediction_contribution': prediction_contribution,
        'leaf_id': int(leaf_id),
        'leaf_samples': int(leaf_total),
        'decision_depth': len(path_features)
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '未找到文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': '仅支持 CSV 格式文件'}), 400

    dataset_id = str(uuid.uuid4())
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}.csv")
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        shape = df.shape
        preview = df.head(10).to_dict(orient='records')
        dtypes = df.dtypes.astype(str).to_dict()

        data_store[dataset_id] = {
            'filepath': filepath,
            'filename': file.filename,
            'columns': columns,
            'shape': {'rows': shape[0], 'cols': shape[1]},
            'preview': preview,
            'dtypes': dtypes,
            'upload_time': datetime.now().isoformat()
        }

        return jsonify({
            'dataset_id': dataset_id,
            'filename': file.filename,
            'columns': columns,
            'shape': {'rows': shape[0], 'cols': shape[1]},
            'preview': preview,
            'dtypes': dtypes
        })

    except Exception as e:
        return jsonify({'error': f'解析 CSV 失败: {str(e)}'}), 500


@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    if dataset_id not in data_store:
        return jsonify({'error': '数据集不存在'}), 404
    return jsonify(data_store[dataset_id])


@app.route('/api/train', methods=['POST'])
def train_model():
    req = request.get_json()
    dataset_id = req.get('dataset_id')
    target_col = req.get('target_column')
    feature_cols = req.get('feature_columns')
    test_size = float(req.get('test_size', 0.2))
    random_state = int(req.get('random_state', 42))
    criterion = req.get('criterion', 'gini')
    max_depth = req.get('max_depth')
    min_samples_split = int(req.get('min_samples_split', 2))
    min_samples_leaf = int(req.get('min_samples_leaf', 1))
    ccp_alpha = float(req.get('ccp_alpha', 0.0))

    if not dataset_id or not target_col:
        return jsonify({'error': '缺少必要参数: dataset_id 和 target_column'}), 400

    if dataset_id not in data_store:
        return jsonify({'error': '数据集不存在'}), 404

    try:
        df = pd.read_csv(data_store[dataset_id]['filepath'])
        n_samples = len(df)
        n_features = len([c for c in df.columns if c != target_col]) if not feature_cols else len(feature_cols)

        if target_col not in df.columns:
            return jsonify({'error': f'目标列 {target_col} 不存在'}), 400

        if not feature_cols:
            feature_cols = [c for c in df.columns if c != target_col]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return jsonify({'error': f'特征列不存在: {missing}'}), 400

        import math
        suggested_max_depth = max(3, min(20, int(math.log2(max(n_samples, 2))) - 1))

        if max_depth is None or max_depth == '' or (isinstance(max_depth, str) and max_depth.strip() == ''):
            max_depth = suggested_max_depth
            user_specified_depth = False
        else:
            max_depth = int(max_depth)
            user_specified_depth = True

        if min_samples_leaf < 1:
            min_samples_leaf = 1
        if min_samples_split < 2:
            min_samples_split = 2
        if ccp_alpha < 0:
            ccp_alpha = 0.0

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        if y.dtype == 'object' or y.dtype.name == 'category':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            label_encoders['__target__'] = target_encoder

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth if max_depth else None,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        )

        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        acc_gap = train_acc - test_acc
        overfitting_warnings = []

        if acc_gap > 0.15:
            overfitting_warnings.append(
                f'检测到严重过拟合：训练准确率比测试高 {acc_gap*100:.1f} 个百分点'
            )
        elif acc_gap > 0.08:
            overfitting_warnings.append(
                f'存在轻度过拟合：训练准确率比测试高 {acc_gap*100:.1f} 个百分点'
            )

        actual_depth = clf.get_depth()
        if not user_specified_depth and actual_depth >= max_depth:
            overfitting_warnings.append(
                f'树已达到最大深度限制 {max_depth}，建议尝试增大 max_depth 或调整 ccp_alpha'
            )

        if ccp_alpha == 0.0 and acc_gap > 0.1:
            suggested_alpha = round(acc_gap * 0.01, 4)
            overfitting_warnings.append(
                f'建议启用剪枝：尝试设置 ccp_alpha ≈ {suggested_alpha} 来缓解过拟合'
            )

        if min_samples_leaf == 1 and acc_gap > 0.1:
            suggested_leaf = max(2, int(n_samples * 0.02))
            overfitting_warnings.append(
                f'建议增大 min_samples_leaf 到 {suggested_leaf} 左右，限制叶子节点样本数'
            )

        class_names = None
        if '__target__' in label_encoders:
            class_names = label_encoders['__target__'].classes_.tolist()
            y_test_labels = label_encoders['__target__'].inverse_transform(y_test)
            y_test_pred_labels = label_encoders['__target__'].inverse_transform(y_test_pred)
            report = classification_report(y_test_labels, y_test_pred_labels, output_dict=True)
        else:
            report = classification_report(y_test, y_test_pred, output_dict=True)

        feature_importance_full = compute_enhanced_feature_importance(clf, feature_cols)
        feature_importance = feature_importance_full['raw']

        model_id = str(uuid.uuid4())
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': clf,
                'feature_columns': feature_cols,
                'target_column': target_col,
                'label_encoders': label_encoders,
                'class_names': class_names
            }, f)

        tree_structure = convert_numpy({
            'depth': clf.get_depth(),
            'leaf_count': clf.get_n_leaves(),
            'node_count': clf.tree_.node_count
        })

        report = convert_numpy(report)
        class_names = convert_numpy(class_names)

        model_store[model_id] = {
            'model_path': model_path,
            'dataset_id': dataset_id,
            'feature_columns': feature_cols,
            'target_column': target_col,
            'class_names': class_names,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'classification_report': report,
            'feature_importance': feature_importance,
            'feature_importance_full': feature_importance_full,
            'tree_structure': tree_structure,
            'train_time': datetime.now().isoformat()
        }

        response = {
            'model_id': model_id,
            'train_accuracy': round(float(train_acc), 4),
            'test_accuracy': round(float(test_acc), 4),
            'accuracy_gap': round(float(acc_gap), 4),
            'feature_importance': feature_importance,
            'feature_importance_enhanced': feature_importance_full,
            'tree_structure': tree_structure,
            'classification_report': report,
            'class_names': class_names,
            'max_depth_used': max_depth,
            'suggested_max_depth': suggested_max_depth,
            'user_specified_depth': user_specified_depth,
            'overfitting_warnings': overfitting_warnings
        }

        if overfitting_warnings:
            response['has_overfitting_warning'] = True
        else:
            response['has_overfitting_warning'] = False

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'训练失败: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    result = []
    for mid, info in model_store.items():
        result.append({
            'model_id': mid,
            'target_column': info['target_column'],
            'test_accuracy': round(info['test_accuracy'], 4),
            'train_time': info['train_time']
        })
    return jsonify({'models': result})


@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    if model_id not in model_store:
        return jsonify({'error': '模型不存在'}), 404
    info = model_store[model_id].copy()
    info.pop('model_path', None)
    return jsonify(info)


@app.route('/api/models/<model_id>/feature_importance', methods=['GET'])
def get_feature_importance(model_id):
    if model_id not in model_store:
        return jsonify({'error': '模型不存在'}), 404

    info = model_store[model_id]

    format_type = request.args.get('format', 'full')
    top_n = request.args.get('top_n', type=int, default=None)
    min_importance = request.args.get('min_importance', type=float, default=0.0)

    if 'feature_importance_full' in info:
        fi = info['feature_importance_full']
    else:
        clf = None
        with open(info['model_path'], 'rb') as f:
            saved = pickle.load(f)
            clf = saved['model']
        fi = compute_enhanced_feature_importance(clf, info['feature_columns'])

    result = {}

    if format_type == 'raw':
        result = fi['raw']
    elif format_type == 'sorted':
        sorted_feats = fi['sorted']
        if min_importance > 0:
            sorted_feats = [f for f in sorted_feats if f['importance'] >= min_importance]
        if top_n:
            sorted_feats = sorted_feats[:top_n]
        result = {'features': sorted_feats}
    elif format_type == 'summary':
        result = {
            'non_zero_count': fi['non_zero_count'],
            'zero_features': fi['zero_features'],
            'top_features': fi['top_features'],
            'total_importance': fi['total_importance']
        }
    else:
        result = fi

    return jsonify({
        'model_id': model_id,
        'target_column': info['target_column'],
        'feature_importance': result
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    req = request.get_json()
    model_id = req.get('model_id')
    samples = req.get('samples')
    return_contribution = req.get('return_contribution', False)

    if not model_id or not samples:
        return jsonify({'error': '缺少必要参数: model_id 和 samples'}), 400

    if model_id not in model_store:
        return jsonify({'error': '模型不存在'}), 404

    try:
        model_path = model_store[model_id]['model_path']
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)

        clf = saved['model']
        feature_cols = saved['feature_columns']
        label_encoders = saved['label_encoders']
        class_names = saved['class_names']

        if isinstance(samples, dict):
            samples = [samples]

        df = pd.DataFrame(samples)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return jsonify({'error': f'样本缺少特征列: {missing}'}), 400

        X = df[feature_cols].copy()
        for col in X.columns:
            if col in label_encoders:
                le = label_encoders[col]
                known = set(le.classes_)
                X[col] = X[col].apply(lambda x: x if x in known else le.classes_[0])
                X[col] = le.transform(X[col].astype(str))

        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        result = []
        for i, pred in enumerate(predictions):
            if '__target__' in label_encoders:
                pred_label = label_encoders['__target__'].inverse_transform([pred])[0]
            else:
                pred_label = pred.item() if hasattr(pred, 'item') else pred

            prob_dict = {}
            if class_names:
                for j, cls in enumerate(class_names):
                    prob_dict[cls] = round(float(probabilities[i][j]), 4)
            else:
                classes = clf.classes_
                for j, cls in enumerate(classes):
                    cls_key = cls.item() if hasattr(cls, 'item') else cls
                    prob_dict[str(cls_key)] = round(float(probabilities[i][j]), 4)

            pred_result = {
                'prediction': pred_label,
                'probabilities': prob_dict
            }

            if return_contribution:
                try:
                    orig_sample = samples[i] if isinstance(samples[i], dict) else dict(zip(feature_cols, X.iloc[i].tolist()))
                    contribution = compute_sample_feature_contribution(
                        clf, feature_cols, orig_sample, label_encoders
                    )
                    pred_result['feature_contribution'] = contribution
                except Exception as ce:
                    pred_result['feature_contribution_error'] = str(ce)

            result.append(pred_result)

        response = {
            'model_id': model_id,
            'predictions': result,
            'prediction_count': len(result)
        }

        if return_contribution:
            feature_importance_summary = None
            if 'feature_importance_full' in model_store[model_id]:
                fi = model_store[model_id]['feature_importance_full']
                feature_importance_summary = {
                    'top_features': fi['top_features'],
                    'non_zero_count': fi['non_zero_count']
                }
            response['feature_importance_summary'] = feature_importance_summary

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500


@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    result = []
    for did, info in data_store.items():
        result.append({
            'dataset_id': did,
            'filename': info['filename'],
            'shape': info['shape'],
            'upload_time': info['upload_time']
        })
    return jsonify({'datasets': result})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件过大，最大支持 100MB'}), 413


if __name__ == '__main__':
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"模型目录: {MODEL_FOLDER}")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
