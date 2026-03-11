# Senior Applied Scientist Study Agenda

---
## Document Outline

### 1. Causal Inference (Primary Spike)
- 1.1 Foundations [C]
- 1.2 Classic Methods [C]
- 1.3 Modern Causal ML [H]
- 1.4 Libraries & Tools [H]
- 1.5 Advanced Topics [M]
- 1.6 Applications [H]

### 2. Probability & Statistics Foundations
- 2.1 Probability Basics [H]
- 2.2 Distributions [H]
- 2.3 Statistical Inference [H]

### 3. ML/DL Fundamentals
- 3.1 Supervised Learning [C]
- 3.2 Unsupervised Learning [M]
- 3.3 Deep Learning Basics [H]
- 3.4 Feature Engineering [H]
- 3.5 Model Selection & Tuning [H]

### 4. Time Series & Forecasting
- 4.1 Fundamentals [C]
- 4.2 Classical Methods [H]
- 4.3 Modern Methods [H]
- 4.4 Practical Considerations [H]
- 4.5 Uncertainty Quantification [M]

### 5. NLP, RAG & Agentic AI
- 5.1 NLP Fundamentals [M]
- 5.2 Transformers & LLMs [H]
- 5.3 RAG [H]
- 5.4 Agentic AI [M]
- 5.5 Causal + AI Intersection [M]

### 6. ML System Design
- 6.1 Design Patterns [H]
- 6.2 Key Components [M]
- 6.3 Case Studies [H]

### 7. Coding & Algorithms
- 7.1 Core Patterns for AS [H]
- 7.2 Python for ML [H]
- 7.3 Coding Best Practices [M]

### 8. Product Sense & Communication
- 8.1 Problem Formulation [C]
- 8.2 Stakeholder Communication [H]
- 8.3 Case Study Practice [H]

### 9. Breadth Topics (Low Priority)
- 9.1 Markov Models [L]
- 9.2 Reinforcement Learning [L]

### 10. Optional Topics
- 10.1 LLM Fine-Tuning [OPTIONAL]
- 10.2 Large-Scale Processing [OPTIONAL]
- 10.3 Deep Framework Proficiency [OPTIONAL]

### Quick Reference
- [Priority Matrix](#quick-reference-priority-matrix)
- [Suggested Learning Order](#suggested-learning-order)
- [Study Approach by Priority](#study-approach-by-priority)
Comprehensive study curriculum for Senior Applied Scientist roles targeting economics/supply chain positions. Causal inference is the primary spike with supporting skills in ML, forecasting, and AI. Importance levels determine time investment: **[C]** Critical (deep study), **[H]** High (strong competence), **[M]** Medium (moderate effort), **[L]** Low (basics for breadth).

---

## Importance Legend & Time Strategy
| Label | Meaning | Interview Weight | Study Strategy |
|-------|---------|------------------|----------------|
| **[C]** | Critical - Must master | Asked in 80%+ of interviews | Deep study, hands-on projects |
| **[H]** | High - Strong competence | Asked in 50-80% of interviews | Solid understanding, practice |
| **[M]** | Medium - Should know well | Asked in 20-50% of interviews | Moderate effort, key concepts |
| **[L]** | Low - Familiarity sufficient | Rarely tested directly | Basics only, high-level discussion |

---

## 1. Causal Inference (Primary Spike)

> Your differentiation. Must be undeniably strong.

### 1.1 Foundations **[C]**
- [ ] 1.1.1 Potential Outcomes Framework (Rubin) **[C]**
  - Counterfactuals and treatment effects (ATE, ATT, CATE) **[C]**
  - SUTVA assumptions **[H]**
  - Assignment mechanisms **[H]**
- [ ] 1.1.2 Structural Causal Models (Pearl) **[C]**
  - Directed Acyclic Graphs (DAGs) **[C]**
  - d-separation and conditional independence **[C]**
  - do-calculus basics **[M]**
- [ ] 1.1.3 Identification Strategies **[C]**
  - Selection on observables vs unobservables **[C]**
  - Backdoor criterion **[C]**
  - Front-door criterion **[M]**

### 1.2 Classic Methods **[C]**
- [ ] 1.2.1 Randomized Experiments **[C]**
  - A/B testing design and analysis **[C]**
  - Power analysis and sample size **[C]**
  - Multiple testing corrections **[H]**
- [ ] 1.2.2 Difference-in-Differences (DiD) **[C]**
  - Parallel trends assumption **[C]**
  - Event studies **[H]**
  - Staggered DiD (Callaway-Sant'Anna, Sun-Abraham) **[M]**
- [ ] 1.2.3 Regression Discontinuity (RDD) **[H]**
  - Sharp vs fuzzy RD **[H]**
  - Bandwidth selection **[M]**
  - Validity checks **[H]**
- [ ] 1.2.4 Instrumental Variables (IV) **[C]**
  - Relevance and exclusion conditions **[C]**
  - 2SLS implementation **[H]**
  - Weak instruments problem **[H]**
- [ ] 1.2.5 Synthetic Control Method **[C]**
  - Donor pool selection **[H]**
  - Pre-treatment fit **[H]**
  - Inference (permutation tests) **[M]**

### 1.3 Modern Causal ML **[H]**
- [ ] 1.3.1 Double/Debiased Machine Learning (DML) **[H]**
  - Cross-fitting procedure **[H]**
  - Nuisance parameter estimation **[M]**
  - When to use vs traditional methods **[C]**
- [ ] 1.3.2 Causal Forests **[H]**
  - Heterogeneous treatment effects **[H]**
  - Honest estimation **[M]**
  - Variable importance for effect modifiers **[M]**
- [ ] 1.3.3 Meta-Learners **[H]**
  - S-learner, T-learner, X-learner **[H]**
  - DR-learner **[M]**
  - Trade-offs and when to use each **[C]**

### 1.4 Libraries & Tools **[H]**
- [ ] 1.4.1 DoWhy **[H]**
  - Causal graph specification **[H]**
  - Identification and estimation **[H]**
  - Refutation tests **[H]**
- [ ] 1.4.2 EconML **[H]**
  - DML implementation **[H]**
  - Causal forests **[H]**
  - Policy learning **[M]**
- [ ] 1.4.3 CausalML **[M]**
  - Uplift modeling **[H]**
  - Meta-learners **[M]**

### 1.5 Advanced Topics **[M]**
- [ ] 1.5.1 Sensitivity Analysis **[M]**
  - Omitted variable bias bounds **[M]**
  - Rosenbaum bounds **[L]**
  - E-values **[L]**
- [ ] 1.5.2 Partial Identification **[L]**
  - Bounds without point identification **[L]**
  - Manski bounds **[L]**
- [ ] 1.5.3 Mediation Analysis **[M]**
  - Direct and indirect effects **[M]**
  - Causal mediation analysis **[M]**

### 1.6 Applications **[H]**
- [ ] 1.6.1 Marketing & Uplift **[H]**
  - Uplift modeling for targeting **[H]**
  - Customer lifetime value attribution **[M]**
- [ ] 1.6.2 Economics-Based Prediction **[H]**
  - Policy evaluation **[H]**
  - Demand estimation **[H]**
- [ ] 1.6.3 Supply Chain Causal Problems **[H]**
  - Supplier effect estimation
  - Intervention analysis

---

## 2. Probability & Statistics Foundations

> Foundation for all quantitative work. Must be solid.

### 2.1 Probability Basics **[H]**
- [ ] 2.1.1 Combinatorics **[M]**
  - Permutations and combinations **[M]**
  - Counting principles **[M]**
- [ ] 2.1.2 Probability Rules **[H]**
  - Conditional probability **[H]**
  - Independence **[H]**
  - Law of total probability **[M]**
- [ ] 2.1.3 Bayes' Theorem **[C]**
  - Intuition and formula **[C]**
  - Prior, likelihood, posterior **[H]**
  - Common interview applications **[C]**

### 2.2 Distributions **[H]**
- [ ] 2.2.1 Discrete Distributions **[H]**
  - Bernoulli, Binomial **[H]**
  - Poisson **[H]**
  - When to use each **[C]**
- [ ] 2.2.2 Continuous Distributions **[H]**
  - Normal (Gaussian) **[C]**
  - Exponential **[M]**
  - Uniform **[M]**
- [ ] 2.2.3 Key Theorems **[H]**
  - Central Limit Theorem (intuition + application) **[C]**
  - Law of Large Numbers **[H]**

### 2.3 Statistical Inference **[H]**
- [ ] 2.3.1 Estimation **[H]**
  - Point estimates **[H]**
  - Confidence intervals **[C]**
  - Standard error **[H]**
- [ ] 2.3.2 Hypothesis Testing **[C]**
  - Null and alternative hypotheses **[C]**
  - p-values and significance **[C]**
  - Type I and Type II errors **[C]**
  - Power analysis **[H]**
- [ ] 2.3.3 Common Tests **[H]**
  - t-tests (one-sample, two-sample, paired) **[H]**
  - Chi-square tests **[M]**
  - ANOVA basics **[M]**

---

## 3. ML/DL Fundamentals

> Core competence for any AS role.

### 3.1 Supervised Learning **[C]**
- [ ] 3.1.1 Core Concepts **[C]**
  - Bias-variance tradeoff (must explain intuitively) **[C]**
  - Overfitting and underfitting **[C]**
  - Train/validation/test splits **[C]**
  - Cross-validation strategies **[H]**
- [ ] 3.1.2 Linear Models **[C]**
  - Linear/logistic regression **[C]**
  - Regularization (L1, L2, ElasticNet) **[C]**
  - Feature scaling importance **[H]**
- [ ] 3.1.3 Tree-Based Methods **[C]**
  - Decision trees (splitting criteria, pruning) **[H]**
  - Random Forests (bagging, feature importance) **[C]**
  - Gradient Boosting (XGBoost, LightGBM, CatBoost) **[C]**
  - Hyperparameter tuning strategies **[H]**
- [ ] 3.1.4 Model Evaluation **[C]**
  - Classification: precision, recall, F1, AUC-ROC, AUC-PR **[C]**
  - Regression: MSE, MAE, MAPE, R-squared **[C]**
  - Calibration and reliability diagrams **[M]**
  - Confusion matrix interpretation **[C]**

### 3.2 Unsupervised Learning **[M]**
- [ ] 3.2.1 Clustering **[M]**
  - K-means (initialization, elbow method) **[H]**
  - Hierarchical clustering **[M]**
  - DBSCAN **[M]**
- [ ] 3.2.2 Dimensionality Reduction **[M]**
  - PCA (intuition and interpretation) **[H]**
  - t-SNE and UMAP (visualization) **[M]**

### 3.3 Deep Learning Basics **[H]**
- [ ] 3.3.1 Neural Network Fundamentals **[H]**
  - Forward and backward propagation **[H]**
  - Activation functions (ReLU, sigmoid, softmax) **[H]**
  - Loss functions **[H]**
  - Optimizers (SGD, Adam) **[M]**
- [ ] 3.3.2 Practical Considerations **[M]**
  - Batch normalization **[M]**
  - Dropout **[H]**
  - Learning rate scheduling **[M]**
  - Early stopping **[H]**
- [ ] 3.3.3 Architectures Overview **[H]**
  - MLPs for tabular data **[H]**
  - CNNs for images (conceptual) **[M]**
  - RNNs/LSTMs for sequences (conceptual) **[M]**
  - Transformers (attention mechanism intuition) **[H]**

### 3.4 Feature Engineering **[H]**
- [ ] 3.4.1 Numerical Features **[H]**
  - Scaling, normalization **[H]**
  - Binning, log transforms **[M]**
  - Handling outliers **[H]**
- [ ] 3.4.2 Categorical Features **[H]**
  - One-hot encoding **[H]**
  - Target encoding **[M]**
  - Embedding approaches **[M]**
- [ ] 3.4.3 Missing Data **[H]**
  - Imputation strategies **[H]**
  - Missingness patterns (MCAR, MAR, MNAR) **[M]**

### 3.5 Model Selection & Tuning **[H]**
- [ ] 3.5.1 Hyperparameter Optimization **[H]**
  - Grid search, random search **[H]**
  - Bayesian optimization (conceptual) **[M]**
- [ ] 3.5.2 Model Comparison **[H]**
  - Statistical tests for model comparison **[M]**
  - Ensemble methods **[H]**

---

## 4. Time Series & Forecasting

> Critical for demand/supply chain roles.

### 4.1 Fundamentals **[C]**
- [ ] 4.1.1 Time Series Components **[C]**
  - Trend, seasonality, cycles, noise **[C]**
  - Additive vs multiplicative decomposition **[H]**
  - STL decomposition **[H]**
- [ ] 4.1.2 Stationarity **[C]**
  - Definition and importance **[C]**
  - ADF test, KPSS test **[H]**
  - Differencing for stationarity **[H]**

### 4.2 Classical Methods **[H]**
- [ ] 4.2.1 ARIMA Family **[H]**
  - AR, MA, ARMA, ARIMA components **[H]**
  - ACF/PACF interpretation **[H]**
  - Model selection (AIC, BIC) **[M]**
  - SARIMA for seasonality **[H]**
- [ ] 4.2.2 Exponential Smoothing **[M]**
  - Simple, Holt, Holt-Winters **[M]**
  - ETS framework **[M]**

### 4.3 Modern Methods **[H]**
- [ ] 4.3.1 Prophet **[H]**
  - Additive model components **[H]**
  - Handling holidays and events **[M]**
  - Changepoint detection **[M]**
  - When to use vs alternatives **[C]**
- [ ] 4.3.2 ML-Based Forecasting **[H]**
  - Feature engineering for time series **[H]**
  - XGBoost/LightGBM for forecasting **[H]**
  - Lag features, rolling statistics **[H]**
- [ ] 4.3.3 Neural Forecasters **[M]**
  - N-BEATS (conceptual) **[L]**
  - Temporal Fusion Transformer (conceptual) **[L]**

### 4.4 Practical Considerations **[H]**
- [ ] 4.4.1 Evaluation Metrics **[C]**
  - MAPE, SMAPE, RMSE, MAE **[C]**
  - When each is appropriate **[C]**
  - Weighted metrics for hierarchies **[M]**
- [ ] 4.4.2 Cross-Validation for Time Series **[H]**
  - Walk-forward validation **[H]**
  - Expanding vs sliding window **[H]**
- [ ] 4.4.3 Multiple Seasonalities **[M]**
  - Daily + weekly + yearly **[M]**
  - TBATS, Prophet, Fourier terms **[M]**
- [ ] 4.4.4 Hierarchical Forecasting **[M]**
  - Top-down, bottom-up, middle-out **[M]**
  - Reconciliation methods **[L]**

### 4.5 Uncertainty Quantification **[M]**
- [ ] 4.5.1 Prediction Intervals **[M]**
  - Parametric intervals **[M]**
  - Bootstrapping **[M]**
  - Conformal prediction **[L]**

---

## 5. NLP, RAG & Agentic AI

> Differentiator and fallback option.

### 5.1 NLP Fundamentals **[M]**
- [ ] 5.1.1 Text Preprocessing **[M]**
  - Tokenization, stemming, lemmatization **[M]**
  - Stop words, n-grams **[L]**
- [ ] 5.1.2 Text Representations **[M]**
  - Bag of Words, TF-IDF **[M]**
  - Word embeddings (Word2Vec, GloVe) **[M]**
  - Sentence embeddings **[H]**
- [ ] 5.1.3 Common Tasks **[M]**
  - Text classification **[M]**
  - Named Entity Recognition (NER) **[M]**
  - Sentiment analysis **[M]**
  - Document similarity **[M]**

### 5.2 Transformers & LLMs **[H]**
- [ ] 5.2.1 Transformer Architecture **[H]**
  - Self-attention mechanism (intuition) **[H]**
  - Positional encoding **[M]**
  - Encoder vs decoder **[H]**
- [ ] 5.2.2 LLM Landscape **[H]**
  - GPT, BERT, T5 differences **[H]**
  - Open vs closed models **[M]**
  - Model selection criteria **[H]**
- [ ] 5.2.3 Using LLMs **[H]**
  - Prompt engineering basics **[H]**
  - Few-shot learning **[H]**
  - Temperature and sampling **[M]**

### 5.3 RAG (Retrieval-Augmented Generation) **[H]**
- [ ] 5.3.1 RAG Architecture **[H]**
  - Embedding models **[H]**
  - Vector databases (conceptual) **[M]**
  - Retrieval strategies **[H]**
  - Generation with context **[H]**
- [ ] 5.3.2 Practical Implementation **[M]**
  - Chunking strategies **[M]**
  - Metadata filtering **[M]**
  - Reranking **[M]**
- [ ] 5.3.3 RAG Evaluation **[H]**
  - Retrieval metrics (recall, MRR) **[M]**
  - Generation quality **[H]**
  - Hallucination detection **[H]**

### 5.4 Agentic AI **[M]**
- [ ] 5.4.1 Agent Patterns **[M]**
  - ReAct pattern **[M]**
  - Tool use and function calling **[H]**
  - Planning and reasoning **[M]**
- [ ] 5.4.2 Agent Frameworks **[M]**
  - LangChain/LlamaIndex basics **[M]**
  - Multi-agent systems (conceptual) **[L]**
  - MCP (Model Context Protocol) basics **[M]**
- [ ] 5.4.3 Agent Challenges **[M]**
  - Reliability and error handling **[M]**
  - Human-in-the-loop **[H]**
  - Evaluation strategies **[M]**

### 5.5 Causal + AI Intersection **[M]**
- [ ] 5.5.1 LLM Evaluation with Causal Lens **[M]**
  - Confounding in LLM benchmarks **[L]**
  - Causal approaches to evaluation **[M]**
- [ ] 5.5.2 LLM-Assisted Causal Discovery **[M]**
  - LLMs for hypothesis generation **[M]**
  - Hybrid human-AI causal analysis **[M]**

---

## 6. ML System Design

> Required for senior roles.

### 6.1 Design Patterns **[H]**
- [ ] 6.1.1 Batch Prediction Systems **[H]**
  - Offline scoring pipelines **[H]**
  - Feature computation **[M]**
  - Model refresh strategies **[M]**
- [ ] 6.1.2 Real-Time Prediction Systems **[H]**
  - Online serving architecture **[H]**
  - Latency considerations **[H]**
  - Caching strategies **[M]**
- [ ] 6.1.3 Training Pipelines **[M]**
  - Data ingestion **[M]**
  - Feature engineering **[H]**
  - Model training and validation **[H]**
  - Artifact management **[L]**

### 6.2 Key Components **[M]**
- [ ] 6.2.1 Feature Stores **[M]**
  - Online vs offline features **[M]**
  - Feature freshness **[L]**
  - Point-in-time correctness **[M]**
- [ ] 6.2.2 Model Serving **[M]**
  - REST APIs vs batch **[M]**
  - Containerization (Docker basics) **[M]**
  - Scaling considerations **[L]**
- [ ] 6.2.3 Monitoring & Observability **[H]**
  - Model performance monitoring **[H]**
  - Data drift detection **[H]**
  - Alerting strategies **[M]**

### 6.3 Case Studies **[H]**
- [ ] 6.3.1 Recommendation System **[H]**
  - Candidate generation + ranking **[H]**
  - Feature engineering for recommendations **[M]**
- [ ] 6.3.2 Demand Forecasting System **[C]**
  - Hierarchical predictions **[H]**
  - Serving forecasts to downstream systems **[M]**
- [ ] 6.3.3 Fraud/Anomaly Detection **[M]**
  - Class imbalance handling **[H]**
  - Real-time scoring requirements **[M]**

---

## 7. Coding & Algorithms

> Maintenance mode - you've already covered DSA.

### 7.1 Core Patterns for AS **[H]**
- [ ] 7.1.1 Arrays & Hashing **[H]**
  - Two pointers, sliding window **[H]**
  - Hash maps for lookups **[H]**
- [ ] 7.1.2 Trees **[H]**
  - Binary tree traversals **[H]**
  - BST operations **[M]**
  - Common tree problems **[H]**
- [ ] 7.1.3 Dynamic Programming **[H]**
  - 1D and 2D DP patterns **[H]**
  - Memoization vs tabulation **[H]**
  - Common problems (knapsack, LIS, paths) **[H]**
- [ ] 7.1.4 Graphs **[M]**
  - BFS/DFS **[M]**
  - Basic shortest path **[L]**

### 7.2 Python for ML **[H]**
- [ ] 7.2.1 NumPy/Pandas Proficiency **[C]**
  - Vectorized operations **[H]**
  - GroupBy and aggregations **[C]**
  - Merging and joins **[H]**
- [ ] 7.2.2 Scikit-learn Fluency **[H]**
  - Pipeline construction **[H]**
  - Custom transformers **[M]**
  - Model persistence **[M]**
- [ ] 7.2.3 SQL Proficiency **[H]**
  - Window functions **[H]**
  - CTEs **[M]**
  - Complex joins and aggregations **[H]**

### 7.3 Coding Best Practices **[M]**
- [ ] 7.3.1 Clean Code **[M]**
  - Readable, modular code **[M]**
  - Error handling **[M]**
  - Type hints **[L]**
- [ ] 7.3.2 Testing Basics **[M]**
  - Unit tests **[M]**
  - Data validation **[M]**

---

## 8. Product Sense & Communication

> What makes you senior.

### 8.1 Problem Formulation **[C]**
- [ ] 8.1.1 Business to Technical Translation **[C]**
  - Identifying the right metric **[C]**
  - Defining success criteria **[C]**
  - Scoping problems appropriately **[H]**
- [ ] 8.1.2 When NOT to Use ML **[C]**
  - Rule-based alternatives **[H]**
  - Cost-benefit analysis **[C]**
  - Technical debt considerations **[M]**

### 8.2 Stakeholder Communication **[H]**
- [ ] 8.2.1 Explaining Models **[H]**
  - Non-technical explanations **[H]**
  - Uncertainty communication **[H]**
  - Limitations and caveats **[H]**
- [ ] 8.2.2 Trade-off Discussions **[H]**
  - Accuracy vs interpretability **[H]**
  - Speed vs accuracy **[M]**
  - Build vs buy **[M]**

### 8.3 Case Study Practice **[H]**
- [ ] 8.3.1 Metric Definition Cases **[H]**
  - "How would you measure success for X?" **[H]**
- [ ] 8.3.2 Ambiguous Problem Cases **[H]**
  - "We have problem X, what would you do?" **[H]**
- [ ] 8.3.3 Trade-off Cases **[H]**
  - "Our model has issue Y, how do you handle it?" **[H]**

---

## 9. Breadth Topics (Low Priority)

> Basics only for high-level discussion. Spend minimal time.

### 9.1 Markov Models **[L]**
- [ ] 9.1.1 Markov Chains **[L]**
  - State transition basics **[L]**
  - Stationary distributions (conceptual) **[L]**
  - When they're used **[L]**
- [ ] 9.1.2 Hidden Markov Models **[L]**
  - High-level intuition only **[L]**
  - Example applications (NLP, speech) **[L]**

### 9.2 Reinforcement Learning **[L]**
- [ ] 9.2.1 Core Concepts Only **[L]**
  - Agent, environment, reward **[L]**
  - Exploration vs exploitation **[L]**
  - Policy vs value functions (conceptual) **[L]**
- [ ] 9.2.2 When RL is Used **[L]**
  - Recommendation systems **[L]**
  - Robotics, gaming **[L]**
  - When NOT to use RL **[L]**

---

## 10. Optional Topics

> Complete main sections first. Visit these only after finishing everything else.

### 10.1 LLM Fine-Tuning **[OPTIONAL]**
- [ ] 10.1.1 Fine-Tuning Approaches **[OPTIONAL]**
  - Full fine-tuning vs parameter-efficient (LoRA, QLoRA) **[OPTIONAL]**
  - When to fine-tune vs prompt engineering **[OPTIONAL]**
- [ ] 10.1.2 Practical Fine-Tuning **[OPTIONAL]**
  - Dataset preparation **[OPTIONAL]**
  - Training considerations **[OPTIONAL]**
  - Evaluation strategies **[OPTIONAL]**
- [ ] 10.1.3 Open Models **[OPTIONAL]**
  - Llama family overview **[OPTIONAL]**
  - Mistral, Phi overview **[OPTIONAL]**

### 10.2 Large-Scale Processing **[OPTIONAL]**
- [ ] 10.2.1 PySpark Basics **[OPTIONAL]**
  - DataFrame operations **[OPTIONAL]**
  - Spark ML overview **[OPTIONAL]**
  - When to use Spark vs Pandas **[OPTIONAL]**
- [ ] 10.2.2 Distributed Training Concepts **[OPTIONAL]**
  - Data parallelism **[OPTIONAL]**
  - Model parallelism (conceptual) **[OPTIONAL]**

### 10.3 Deep Framework Proficiency **[OPTIONAL]**
- [ ] 10.3.1 PyTorch **[OPTIONAL]**
  - Tensor operations **[OPTIONAL]**
  - Building simple models **[OPTIONAL]**
  - Training loops **[OPTIONAL]**
- [ ] 10.3.2 TensorFlow/Keras **[OPTIONAL]**
  - High-level API usage **[OPTIONAL]**
  - Model building basics **[OPTIONAL]**

---

## Quick Reference: Priority Matrix

| Area | Critical [C] | High [H] | Medium [M] | Low [L] |
|------|-------------|----------|------------|---------|
| **Causal Inference** | Foundations, Classic Methods | Causal ML, Libraries, Applications | Advanced Topics | - |
| **Prob & Stats** | - | Probability, Distributions, Inference | - | - |
| **ML/DL** | Supervised Learning | DL Basics, Feature Eng, Tuning | Unsupervised | - |
| **Time Series** | Fundamentals | Classical, Modern, Practical | Neural, Hierarchical, UQ | - |
| **NLP/RAG/Agentic** | - | Transformers, RAG | NLP Basics, Agentic, Causal+AI | - |
| **System Design** | - | Patterns, Case Studies | Components | - |
| **Coding** | - | Core Patterns, Python/SQL | Graphs, Best Practices | - |
| **Product Sense** | Problem Formulation | Communication, Cases | - | - |
| **Breadth** | - | - | - | Markov, RL |
| **Optional** | - | - | - | Fine-tuning, Spark, Frameworks |

---

## Suggested Learning Order

### Phase 1: Core Foundations
1. **Causal Inference 1.1-1.2** → Build your spike foundation
2. **Prob & Stats 2.1-2.3** → Statistical foundations
3. **ML/DL 3.1, 3.4** → Core ML competence

### Phase 2: Depth & Breadth
4. **Time Series 4.1-4.4** → Practical forecasting skills
5. **Causal Inference 1.3-1.4, 1.6** → Modern causal ML + applications
6. **ML/DL 3.3, 3.5** → Deep learning basics + tuning

### Phase 3: Differentiation
7. **NLP/RAG 5.2-5.3** → AI literacy and fallback
8. **System Design 6.1, 6.3** → Senior-level patterns
9. **Product Sense 8.1-8.3** → Interview polish

### Phase 4: Breadth & Polish
10. **Agentic AI 5.4-5.5** → Differentiator
11. **Breadth Topics 9.1-9.2** → Quick pass for discussion ability
12. **Optional 10.x** → Only if time permits

---

## Study Approach by Priority

| Priority | Depth | Approach |
|----------|-------|----------|
| **[C] Critical** | Mastery | Deep study, hands-on projects, practice explaining, must be able to teach others |
| **[H] High** | Strong competence | Solid understanding, some hands-on, know key concepts and trade-offs |
| **[M] Medium** | Working knowledge | Focused study, understand when/why to use, can discuss intelligently |
| **[L] Low** | Familiarity | Quick pass, high-level intuition, enough for breadth in conversation |
| **[OPTIONAL]** | As needed | Only after completing all core sections |

> [!NOTE]
> Time investment depends on topic complexity and breadth, not just priority level. A medium-priority broad topic (e.g., Unsupervised Learning) may require more hours than a critical but narrow topic.
