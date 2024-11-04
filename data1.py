from datasets import load_dataset

# Tải dữ liệu từ tệp CSV với encoding UTF-8
dataset = load_dataset('csv', data_files='dataset.csv', encoding='utf-8')

# Kiểm tra dữ liệu
print(dataset['train'][0])
