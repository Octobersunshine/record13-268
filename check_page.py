import requests
r = requests.get('http://127.0.0.1:5000/')
print(f'状态码: {r.status_code}')
print(f'内容长度: {len(r.text)}')
print(f'包含"决策树": {"决策树" in r.text}')
print(f'包含"特征重要性": {"特征重要性" in r.text}')
print(f'包含"特征贡献分析": {"特征贡献分析" in r.text}')
print(f'包含"显示特征贡献分析": {"显示特征贡献分析" in r.text}')
