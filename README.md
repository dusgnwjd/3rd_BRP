# 3rd_BRP

📈 Financial Forecasting by Fusing Visual Streams
Context-Aware Time-Series Forecasting by Fusing Numerical Data with Chart Visuals

📖 Introduction
This project introduces a novel multi-modal framework designed to enhance financial time-series forecasting. While traditional Large Language Models (LLMs) rely solely on numerical or textual data, they often fail to capture the visual and geometric patterns (e.g., "head and shoulders", "support/resistance levels") that human experts rely on.

We propose a 2-stream architecture that empowers the model to "see" candlestick charts and fuse this visual insight with temporal-numerical data, achieving superior forecasting accuracy compared to uni-modal baselines.

💡 Motivation
Existing approaches in financial forecasting have a critical limitation:

Numerical-Only Models (Time-LLM, LSTM): Can analyze historical price trends but remain "blind" to geometric chart patterns.

Vision-Only Models: Can classify chart shapes but often lack the granular temporal context of time-series data.

Our framework bridges this gap. We are motivated by the intuition that a truly informed prediction requires simultaneously reading the numbers and seeing the charts—just as a professional trader would.

🚀 Key Features
Multi-Modal Fusion: Combines numerical time-series data with candlestick chart images.

2-Stream Architecture:

Visual Stream: Processes chart images using a Vision Encoder (e.g., ViT) to extract geometric features.

Temporal Stream: Processes numerical/textual data using a Time-LLM backbone.

TVCA Module: Utilizes a Temporal-Visual Cross-Attention mechanism to dynamically weigh the importance of visual patterns based on the current market context.

🏗️ Architecture
(Insert your architecture diagram here, e.g., assets/architecture.png)

The model operates in two main stages:

Dual Encoding: Parallel processing of the image stream and the time-series/text stream.

Cross-Modal Fusion: The temporal embedding acts as a Query to attend to the visual Keys/Values extracted from the chart, generating a context-aware fused representation.

🛠️ Experimental Setup
We validate our model using real-world financial data generated via yahoo-finance and mplfinance.

Datasets: S&P 500 (^GSPC), KOSPI, Gold, and Crude Oil.

Baselines:

LSTM (Numerical-only)

Time-LLM (Text + Numerical)

Vision-Only (Chart Images)

Metrics: MAPE, AUC-ROC, Confusion Matrix.
