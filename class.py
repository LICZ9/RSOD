import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Count annotations and images in a COCO annotation file.')
    parser.add_argument(
        'annotation_file', type=Path, help='Path to the annotation JSON file')
    args = parser.parse_args()

    with args.annotation_file.expanduser().open(encoding='utf-8-sig') as f:
        anns = json.load(f)
    print("标注数量:", len(anns['annotations']))
    print("图片数量:", len(anns['images']))


if __name__ == '__main__':
    main()
