from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
import os
from datasets import Dataset

dataset_file = "custom_dataset.json"  # Ensure this file exists in the same directory

def preprocess(data):
    inputs = [f"Context: {item['context']}\nQ: {item['question']}\nA: {item['answer']}" for item in data]
    return Dataset.from_dict({"text": inputs})

with open(dataset_file, "r") as f:
    dataset = json.load(f)
processed_dataset = preprocess(dataset)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)

dataset_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

os.makedirs("./fine_tuned_gpt2_model", exist_ok=True)
model.save_pretrained("./fine_tuned_gpt2_model")
tokenizer.save_pretrained("./fine_tuned_gpt2_model")

fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_gpt2_model")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_gpt2_model")

def generate_response(prompt):
    inputs = fine_tuned_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = fine_tuned_model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    return fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)

prompts = [
    "Who is Gandhi?",
    "Who created Python?",
    "What is the largest ocean on Earth?"
]
responses = [generate_response(prompt) for prompt in prompts]

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}\nResponse: {response}\n")
