# Chatbot Using the BART Model 

## Introduction
This is the source code for a chatbot built using the BART (Bidirectional and Autoregressive Transformers for pre-training) model, fine-tuned on a CSV question-answer dataset. This chatbot can understand and respond to user questions in a natural way.

## Requirements
- **Transformers library**: Install using `pip install transformers`
- **Datasets library**: Install using `pip install datasets`
- **CSV data file**: A file containing columns for `question` and `answer`, encoded in UTF-8

## Directory Structure
- `dataset.csv`: The question-answer data file
- `results`: A directory to save training results (optional, created by Trainer)
- `logs`: A directory to save training logs (optional, created by Trainer)

## Usage
1. **Prepare Data**: Ensure that your `dataset.csv` file is in the same directory as the source code.
![image](https://github.com/user-attachments/assets/48f258b2-3b2d-4d54-816e-0669fbd02c67)

2. **Run Training**: Execute the script using the command `python train.py`. The training process will save results to the `results` directory and logs to the `logs` directory (if configured).
![image](https://github.com/user-attachments/assets/ceae2399-5c5b-4066-8131-46aa0a0309b1)

3. **Run Interactive Mode**: After training is complete, the script will switch to interactive mode. You can enter your questions, and the chatbot will respond. Enter `exit` to quit the program.

## Code Description
- The script begins by importing the necessary libraries.
- The `load_dataset` function is used to load data from the CSV file.
- The code checks a sample of the data to ensure it loads correctly.
- The `BartTokenizer` is initialized from the `facebook/bart-large` model.
- The `preprocess_function` performs data preprocessing, including tokenizing questions and answers and setting labels using the input IDs of the answers.
- The data is tokenized into tensors using `tokenizer.map`.
- The `BartForConditionalGeneration` model is loaded from `facebook/bart-large`.
- Training parameters are set up in `TrainingArguments`.
- The `Trainer` is initialized to manage the training process.
- The `generate_text` function uses the model to generate text responses for an input question.
- The main loop allows the user to interact with the chatbot.

## Notes
- You can change training parameters (number of epochs, batch size, etc.) in `TrainingArguments` to customize the training process.
- Depending on the dataset size and computational resources, the training process may take a considerable amount of time.
- The quality of the chatbot's responses depends on the quality of the training dataset.

## References
- **Transformers library**: [Transformers Documentation](https://huggingface.co/docs/transformers/en/index)
- **Datasets library**: [Datasets Documentation](https://huggingface.co/docs/datasets/en/index)
- **BART model**: [BART Model Documentation](https://huggingface.co/docs/transformers/en/model_doc/marian)

## How to run 
1. Run dataset.py to export dataset
2. Then run train.py to train ( make sure you have installed all the libraries ) - wait about 30min to train

=> after training, you will see this 2 folder ( which is folder include model ang results )

![image](https://github.com/user-attachments/assets/b134d1ae-d8fa-4beb-8e60-76a1b8039ea0)

3. Then you can type your question and ChatBot will answer

![image](https://github.com/user-attachments/assets/9010837f-928d-451e-a1b3-db16ac8e7378)
![image](https://github.com/user-attachments/assets/c6535dea-19e5-4798-8f01-75725b4d99c5)

## This code belongs to ConThoBanSung, please ask me to take the code first
