


import json
with open('/sharefiles2/lichengzhou/datasets/sonar_mnist/annotations/instances_10train.json', 'r') as f:
    anns = json.load(f)
print("标注数量:", len(anns['annotations']))
print("图片数量:", len(anns['images']))