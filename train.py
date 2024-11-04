from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Tải dữ liệu từ tệp CSV với encoding UTF-8
dataset = load_dataset('csv', data_files='dataset.csv', encoding='utf-8')

# Kiểm tra dữ liệu
print(dataset['train'][0])

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer.pad_token = tokenizer.eos_token  # Đặt padding token là eos_token

# Hàm tiền xử lý
def preprocess_function(examples):
    # Tokenize câu hỏi và câu trả lời
    inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=128)
    outputs = tokenizer(examples['answer'], truncation=True, padding='max_length', max_length=128)

    inputs['labels'] = outputs['input_ids']  # Đặt labels bằng input_ids của câu trả lời
    return inputs

# Tokenize dữ liệu
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load mô hình BART
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# Đặt các tham số huấn luyện
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Tạo Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

# Bắt đầu huấn luyện
trainer.train()

# Hàm sinh văn bản
def generate_text(prompt, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  # Ngăn lặp lại n-grams
        early_stopping=True
    )
    
    # Giải mã văn bản được sinh
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts

# Chạy vòng lặp trong terminal để hỏi đáp
if __name__ == '__main__':
    print("Chatbot đã sẵn sàng! Nhập câu hỏi của bạn hoặc gõ 'exit' để thoát.")
    
    while True:
        prompt = input("Bạn: ")
        if prompt.lower() == "exit":
            print("Kết thúc phiên trò chuyện.")
            break
        
        generated_texts = generate_text(prompt, max_length=2000, num_return_sequences=1)
        
        for i, text in enumerate(generated_texts):
            print(f"Chatbot: {text}")
