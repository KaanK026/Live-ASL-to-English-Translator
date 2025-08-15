import json

def _class_to_idx_json(dataset, output_path='class_to_idx.json'):
    with open(output_path, 'w') as f:
        json.dump(dataset.class_to_idx, f)


def load_idx_to_class(mapping_path='class_to_idx.json'):
    with open(mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    return {int(v): k for k, v in class_to_idx.items()}