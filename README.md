#  Fine-Tuning Transformers for Question Answering Using BERT-Base-Uncased

This project titled **“Fine-Tuning Transformers for Question Answering Using BERT-Base-Uncased”** was developed as part of my **AI Engineering Bootcamp**, focusing on fine-tuning a pre-trained Transformer model to perform **extractive Question Answering (QA)** on the **SQuAD (Stanford Question Answering Dataset)**. The main goal was to train a model capable of identifying precise text spans from a given context that answer a specific question. I utilized the `bert-base-uncased` model from Hugging Face’s Transformers library and worked with subsets of 4,000 training and 2,000 validation samples from the SQuAD dataset for efficient experimentation. A custom preprocessing function was implemented to tokenize each question-context pair while carefully mapping answer start and end positions within the context, ensuring span-based prediction learning. The model was fine-tuned using the Hugging Face `Trainer` API with parameters including a learning rate of 3e-5, batch size of 16, weight decay of 0.01, and 3 training epochs. Evaluation was performed using the official SQuAD metric, achieving an **Exact Match (EM)** score of **66.05%** and an **F1** score of **74.66%**, which demonstrated strong extractive capabilities even on a smaller dataset.  

After fine-tuning, I created an inference pipeline that could answer unseen questions such as *“Who developed the theory of relativity?”* → **Albert Einstein**, and *“Where is the Eiffel Tower located?”* → **Paris, France**, confirming the model’s ability to generalize beyond training samples. This project provided hands-on experience in fine-tuning large language models, designing preprocessing pipelines for span-based QA, leveraging Hugging Face’s `Trainer` and `Pipeline` APIs for streamlined training and inference, and evaluating model performance using SQuAD metrics.  

---

###  Key Highlights:
- Dataset: **SQuAD** from Hugging Face  
- Model: **BERT-Base-Uncased** fine-tuned for extractive QA  
- Training: 3 epochs, batch size 16, learning rate 3e-5, weight decay 0.01  
- Evaluation Metrics: **Exact Match (66.05%)**, **F1 (74.66%)**  
- Frameworks: Transformers, Datasets, Evaluate, PyTorch  
- Tools: Hugging Face `Trainer` & `Pipeline` APIs for training and inference  

---

###  Future Improvements:
- Experiment with larger transformer architectures like **BERT-Large** or **RoBERTa**  
- Apply model **distillation or quantization** for faster inference  
- Deploy the QA model as an **interactive web app** using Streamlit or Flask  

---

**Author:** Md. Shoaib Ahmed
