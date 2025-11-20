# Multimodal Financial Forecasting with Modified Time-LLM

A novel financial forecasting framework that enhances time-series prediction accuracy by fusing multimodal data (news text, chart images, and time-series) using a modified Time-LLM architecture.

## 📋 Overview

This project aims to improve financial forecasting performance by leveraging multiple data modalities beyond traditional time-series data. By incorporating news sentiment and chart pattern recognition alongside numerical time-series, we create a more comprehensive view of market dynamics.

### Key Innovation

- **Multimodal Fusion**: Combines three data sources:
  - 📊 Time-series numerical data (price, volume, indicators)
  - 📰 News text data (financial news, reports, social media)
  - 📈 Chart images (candlestick patterns, technical indicators)
- **Modified Time-LLM Architecture**: Extends the Time-LLM baseline with cross-modal attention mechanisms
- **Enhanced Prediction**: Leverages the reasoning capabilities of Large Language Models for financial forecasting

## 🎯 Motivation

Traditional time-series forecasting models often ignore valuable contextual information available in news and visual chart patterns. This project addresses that gap by:

1. Incorporating textual sentiment and events that drive market movements
2. Leveraging visual pattern recognition from chart images
3. Using LLMs to understand complex market narratives
4. Achieving better prediction accuracy than unimodal approaches

## 🏗️ Architecture

### Base: Time-LLM

We use [Time-LLM](https://github.com/KimMeen/Time-LLM) as our performance baseline. Time-LLM reprograms time-series data into the embedding space of pre-trained Large Language Models for forecasting.

### Our Modifications

```
┌─────────────────────────────────────────────────────────────┐
│                    Multimodal Input Layer                    │
├──────────────────┬──────────────────┬───────────────────────┤
│  Time-Series     │   News Text      │   Chart Images        │
│  Encoder         │   Encoder        │   Encoder             │
│  (Patching)      │   (BERT/LLaMA)   │   (Vision Transform)  │
└────────┬─────────┴─────────┬────────┴──────────┬────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
                   ┌─────────────────────┐
                   │  Cross-Modal        │
                   │  Attention Fusion   │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Modified Time-LLM  │
                   │  Backbone (LLaMA)   │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Forecasting Head   │
                   └──────────┬──────────┘
                              ▼
                        Predictions
```

### Key Components

1. **Multimodal Encoders**
   - Time-series: Patching and embedding mechanism from Time-LLM
   - Text: Pre-trained language model encoder (BERT/RoBERTa)
   - Vision: Vision Transformer (ViT) or CNN-based encoder

2. **Cross-Modal Attention**
   - Aligns temporal information across modalities
   - Learns inter-modal dependencies
   - Weighted fusion based on relevance

3. **LLM Backbone**
   - Frozen or fine-tuned LLaMA/GPT model
   - Processes fused multimodal representations
   - Generates contextually-aware forecasts

## ✨ Features

- **Flexible Data Fusion**: Supports any combination of modalities
- **Modular Architecture**: Easy to swap encoders or LLM backbones
- **Pre-trained Models**: Leverages existing LLMs and vision models
- **Interpretability**: Attention weights reveal which modality influences predictions
- **Scalable**: Efficient training with parameter-efficient fine-tuning (LoRA, Adapter)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/dusgnwjd/3rd_BRP.git
cd 3rd_BRP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Time-LLM baseline
pip install time-llm  # or follow Time-LLM installation instructions
```

### Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
timm>=0.9.0
pandas>=1.5.0
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 📊 Dataset

### Expected Data Format

The model expects three aligned data streams:

1. **Time-Series Data** (`data/timeseries/`)
   - Format: CSV with columns [timestamp, price, volume, ...]
   - Frequency: Daily, hourly, or minute-level
   - Features: OHLCV, technical indicators

2. **News Data** (`data/news/`)
   - Format: JSON with fields [timestamp, title, content, source]
   - Aligned with time-series timestamps
   - Sources: Financial news, earnings reports, social media

3. **Chart Images** (`data/charts/`)
   - Format: PNG/JPG images
   - Content: Candlestick charts, technical indicator overlays
   - Naming: `{timestamp}.png` for alignment

### Sample Dataset Structure

```
data/
├── timeseries/
│   ├── AAPL_2020-2023.csv
│   └── GOOGL_2020-2023.csv
├── news/
│   ├── AAPL_news_2020-2023.json
│   └── GOOGL_news_2020-2023.json
└── charts/
    ├── AAPL/
    │   ├── 2020-01-01.png
    │   └── ...
    └── GOOGL/
        └── ...
```

### Data Collection

We recommend using:
- **Time-series**: Yahoo Finance, Alpha Vantage, Binance API
- **News**: NewsAPI, Bloomberg Terminal, Twitter API
- **Charts**: Generate from time-series using `mplfinance` or `plotly`

## 💻 Usage

### Training

```python
from models.multimodal_timellm import MultimodalTimeLLM
from data.dataloader import MultimodalDataLoader

# Initialize model
model = MultimodalTimeLLM(
    llm_backbone="llama-7b",
    fusion_method="cross_attention",
    forecast_horizon=7  # 7-day forecast
)

# Load data
dataloader = MultimodalDataLoader(
    timeseries_path="data/timeseries/",
    news_path="data/news/",
    charts_path="data/charts/",
    batch_size=32
)

# Train
model.train(
    dataloader=dataloader,
    epochs=50,
    learning_rate=1e-4,
    use_lora=True  # Parameter-efficient fine-tuning
)
```

### Inference

```python
# Load trained model
model = MultimodalTimeLLM.load("checkpoints/best_model.pt")

# Prepare input
timeseries = load_timeseries("data/test/AAPL_recent.csv")
news = load_news("data/test/AAPL_news_recent.json")
chart = load_image("data/test/AAPL_chart.png")

# Predict
forecast, attention_weights = model.predict(
    timeseries=timeseries,
    news=news,
    chart=chart,
    horizon=7
)

print(f"7-day forecast: {forecast}")
print(f"Attention weights: {attention_weights}")  # Which modality contributed most
```

### Evaluation

```python
from evaluation.metrics import evaluate_forecast

# Evaluate on test set
results = evaluate_forecast(
    model=model,
    test_dataloader=test_loader,
    metrics=["MSE", "MAE", "MAPE", "R2"]
)

print(results)
```

## 📈 Performance Baseline

We compare against Time-LLM (unimodal) and other baselines:

| Model | MSE ↓ | MAE ↓ | MAPE ↓ | R² ↑ |
|-------|-------|-------|--------|------|
| ARIMA | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD | TBD |
| **Time-LLM** (baseline) | TBD | TBD | TBD | TBD |
| **Ours (Multimodal)** | TBD | TBD | TBD | TBD |

*Note: Results will be updated as experiments are completed*

### Ablation Study

To understand each modality's contribution:

| Configuration | MSE ↓ | Improvement |
|---------------|-------|-------------|
| Time-series only (Time-LLM) | TBD | Baseline |
| + News text | TBD | TBD% |
| + Chart images | TBD | TBD% |
| + Both (Full model) | TBD | TBD% |

## 📁 Project Structure

```
3rd_BRP/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── timeseries/
│   ├── news/
│   └── charts/
├── models/
│   ├── multimodal_timellm.py
│   ├── encoders/
│   │   ├── timeseries_encoder.py
│   │   ├── text_encoder.py
│   │   └── vision_encoder.py
│   └── fusion/
│       ├── cross_attention.py
│       └── late_fusion.py
├── data/
│   ├── dataloader.py
│   ├── preprocessing.py
│   └── augmentation.py
├── training/
│   ├── trainer.py
│   └── callbacks.py
├── evaluation/
│   ├── metrics.py
│   └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate_charts.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualization.ipynb
└── tests/
    ├── test_models.py
    ├── test_dataloader.py
    └── test_preprocessing.py
```

## 🔬 Experiments

### Planned Experiments

1. **Modality Ablation**: Evaluate contribution of each modality
2. **Fusion Strategies**: Compare early vs. late vs. cross-attention fusion
3. **LLM Backbone Comparison**: Test different LLMs (LLaMA, GPT, Mistral)
4. **Fine-tuning Strategies**: Full fine-tuning vs. LoRA vs. Adapter vs. Frozen
5. **Forecast Horizons**: Short-term (1-7 days) vs. long-term (30+ days)
6. **Market Conditions**: Bull market, bear market, high volatility periods

### Hyperparameters to Tune

- Learning rate and schedule
- Batch size
- Fusion layer depth
- Attention head count
- LoRA rank (if using parameter-efficient fine-tuning)
- Input sequence length
- Chart image resolution

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 models/ data/ training/
black models/ data/ training/

# Type checking
mypy models/ data/ training/
```

## 🗺️ Roadmap

### Phase 1: Foundation (Current)
- [x] Project setup and README
- [ ] Implement multimodal data loaders
- [ ] Integrate Time-LLM baseline
- [ ] Basic training pipeline

### Phase 2: Core Development
- [ ] Implement cross-modal attention fusion
- [ ] Add vision and text encoders
- [ ] Complete training and evaluation scripts
- [ ] Benchmark against Time-LLM baseline

### Phase 3: Optimization
- [ ] Hyperparameter tuning
- [ ] Parameter-efficient fine-tuning (LoRA)
- [ ] Model compression and deployment
- [ ] Real-time inference optimization

### Phase 4: Advanced Features
- [ ] Add more modalities (sentiment, macro-economic indicators)
- [ ] Multi-asset forecasting
- [ ] Uncertainty quantification
- [ ] Interpretability tools (attention visualization)

### Phase 5: Production
- [ ] REST API for inference
- [ ] Model versioning and tracking (MLflow)
- [ ] Continuous retraining pipeline
- [ ] Documentation and tutorials

## 📚 References

### Time-LLM
- Paper: [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728)
- GitHub: [https://github.com/KimMeen/Time-LLM](https://github.com/KimMeen/Time-LLM)

### Related Work
- **Multimodal Learning**: "Multimodal Machine Learning: A Survey and Taxonomy" (Baltrušaitis et al., 2019)
- **Financial Forecasting with LLMs**: "Large Language Models for Time Series: A Survey" (Jin et al., 2023)
- **Vision Transformers**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- **Cross-Modal Attention**: "Attention is All You Need" (Vaswani et al., 2017)

### Datasets
- **Financial Time-series**: [Yahoo Finance](https://finance.yahoo.com/), [Kaggle Financial Datasets](https://www.kaggle.com/search?q=financial+time+series)
- **Financial News**: [Kaggle News Datasets](https://www.kaggle.com/search?q=financial+news), NewsAPI
- **Stock Charts**: Generate from time-series using `mplfinance`

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{3rd_brp_2024,
  title={Multimodal Financial Forecasting with Modified Time-LLM},
  author={3rd BRP Team},
  year={2024},
  url={https://github.com/dusgnwjd/3rd_BRP}
}

@article{timellm2023,
  title={Time-LLM: Time Series Forecasting by Reprogramming Large Language Models},
  author={Ming, Jin and others},
  journal={arXiv preprint arXiv:2310.01728},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Time-LLM team for the baseline architecture
- Hugging Face for transformer models
- PyTorch team for the deep learning framework
- The open-source community for various tools and libraries

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [https://github.com/dusgnwjd/3rd_BRP/issues](https://github.com/dusgnwjd/3rd_BRP/issues)
- Project Maintainer: [@dusgnwjd](https://github.com/dusgnwjd)

---

**Note**: This is an active research project. Performance metrics and implementation details will be updated as the project progresses.