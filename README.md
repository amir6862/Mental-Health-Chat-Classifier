# Mental-Health-Chat-Classifier
# 🧠 Mental Health Chat Classifier

A machine learning project that classifies chat messages into mental health categories: **Depression**, **Anxiety**, or **Neutral**. Built with scikit-learn and deployed as an interactive Streamlit web app.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.25.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates end-to-end machine learning pipeline for text classification in mental health domain:

- **Data Collection**: Custom dataset with depression, anxiety, and neutral text samples
- **Preprocessing**: Text cleaning, tokenization, lemmatization, and TF-IDF vectorization
- **Model Training**: Multiple algorithms comparison (Logistic Regression, Random Forest, SVM)
- **Evaluation**: Comprehensive metrics and confusion matrix analysis
- **Deployment**: Interactive Streamlit web application

## 📊 Features

### 🔍 Core Functionality
- **Single Message Analysis**: Classify individual chat messages
- **Batch Processing**: Analyze multiple messages simultaneously
- **Confidence Scores**: Get prediction confidence for each classification
- **Feature Analysis**: See which words contribute most to predictions

### 📈 Visualization
- Interactive probability charts
- Confusion matrix heatmaps
- Category distribution pie charts
- Real-time prediction results

### 🛡️ Safety Features
- Input validation and error handling
- Mental health crisis resources
- Clear disclaimers about tool limitations

## 📁 Project Structure

```
mental-health-chat-classifier/
├── data/
│   ├── raw_data.csv              # Generated sample dataset
│   └── processed_data.csv        # Cleaned and preprocessed data
├── model/
│   ├── mental_health_classifier.pkl    # Trained model
│   ├── tfidf_vectorizer.pkl           # TF-IDF transformer
│   ├── label_encoder.pkl              # Label encoder
│   ├── model_info.pkl                 # Model metadata
│   └── confusion_matrix.png           # Evaluation visualization
├── notebooks/
│   └── train_model.ipynb              # Jupyter notebook version
├── app/
│   └── app.py                         # Streamlit web application
├── generate_sample_data.py            # Dataset generation script
├── data_preprocessing.py              # Text preprocessing utilities
├── train_model.py                     # Model training pipeline
├── model_inference.py                 # Prediction utilities
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/mental-health-chat-classifier.git
cd mental-health-chat-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Sample Data
```bash
python generate_sample_data.py
```

### 4. Preprocess Data
```bash
python data_preprocessing.py
```

### 5. Train Model
```bash
python train_model.py
```

### 6. Run Web App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📈 Model Performance

### Training Results
- **Best Model**: Logistic Regression
- **Test Accuracy**: ~85-90% (varies with random seed)
- **Classes**: Depression, Anxiety, Neutral

### Evaluation Metrics
```
                precision    recall  f1-score   support
   Depression       0.85      0.82      0.84        17
      Anxiety       0.89      0.89      0.89        18
      Neutral       0.85      0.88      0.86        17

     accuracy                           0.87        52
    macro avg       0.86      0.86      0.86        52
 weighted avg       0.87      0.87      0.86        52
```

## 🔧 Technical Details

### Data Preprocessing Pipeline
1. **Text Cleaning**: Remove URLs, mentions, special characters
2. **Tokenization**: Split text into individual words
3. **Stopword Removal**: Filter common English stopwords
4. **Lemmatization**: Reduce words to root forms
5. **Vectorization**: Convert to TF-IDF features (5000 features max)

### Model Architecture
- **Algorithm**: Logistic Regression with hyperparameter tuning
- **Features**: TF-IDF vectors with unigrams and bigrams
- **Regularization**: L2 penalty to prevent overfitting
- **Cross-validation**: 5-fold CV for model selection

### Web Application
- **Framework**: Streamlit for rapid prototyping
- **Visualization**: Plotly for interactive charts
- **Deployment**: Ready for Streamlit Cloud deployment



## 🧪 Usage Examples

### Command Line Prediction
```python
from model_inference import MentalHealthPredictor

predictor = MentalHealthPredictor()
result = predictor.predict("I feel so anxious about everything")

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing
```python
texts = [
    "I feel hopeless and empty inside",
    "Having a wonderful day today", 
    "Can't stop worrying about everything"
]

results = predictor.predict_batch(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: {result['predicted_class']}")
```

## 🔬 Model Interpretation

### Feature Analysis
The model uses TF-IDF to identify important words:

**Depression Indicators:**
- "empty", "hopeless", "tired", "worthless"
- "can't", "nothing", "anymore"

**Anxiety Indicators:** 
- "worried", "anxious", "panic", "racing"
- "what if", "terrified", "scared"

**Neutral Indicators:**
- "good", "today", "work", "friends"
- "planning", "enjoying", "normal"

## ⚠️ Important Disclaimers

1. **Educational Purpose Only**: This tool is for demonstration and learning
2. **Not Medical Advice**: Cannot replace professional mental health assessment
3. **Limited Training Data**: Based on sample dataset, not comprehensive
4. **Bias Considerations**: May not generalize to all populations or contexts

### Crisis Resources
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Improvement
- [ ] Larger, more diverse training dataset
- [ ] Advanced NLP models (BERT, RoBERTa)
- [ ] More mental health categories
- [ ] Severity scoring within categories
- [ ] Multi-language support
- [ ] Real-time chat integration

## 📚 Dataset Sources

Current dataset is synthetically generated for demonstration. For production use, consider:

- **CLPsych Shared Tasks**: Academic mental health datasets
- **Reddit Mental Health**: r/depression, r/anxiety subreddits (with permission)
- **Twitter Mental Health**: Academic research datasets
- **Crisis Text Line**: Anonymized conversation data (if available)

## 🛠️ Development Setup

### Advanced Features (Optional)
```bash
# Install additional NLP libraries
pip install spacy transformers torch

# Download spaCy model
python -m spacy download en_core_web_sm

# For BERT-based models
pip install sentence-transformers
```

### Jupyter Notebook Development
```bash
# Install Jupyter
pip install jupyter notebook

# Start notebook server
jupyter notebook notebooks/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML library
- **Streamlit** for making web apps simple
- **NLTK** for natural language processing tools
- Mental health researchers for domain insights

## 📞 Contact

**Your Name** - amirabbas7511@gmail.com


---

⭐ **If this project helped you, please give it a star!** ⭐
