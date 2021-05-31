import json
import random
with open('data/baidu/train.json', 'r') as f:
    with open('data/baidu/dev.json', 'w') as dev:
        with open('data/baidu/train2.json', 'w') as tra:
            json_line = []
            for line in f.readlines():
                json_line.append(json.loads(line))

            random.shuffle(json_line)
            split = int(len(json_line) * 0.2)
            for dev_data in json_line[:split]:
                dev.writelines(json.dumps(dev_data, ensure_ascii=False) + '\n')
            for train_data in json_line[split:]:
                tra.writelines(json.dumps(train_data, ensure_ascii=False) + '\n')








