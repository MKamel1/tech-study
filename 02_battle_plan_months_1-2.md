# Battle Plan - Months 1-2: Foundations & Project #1

**Duration**: 8 weeks (after Phase 0)  
**Total Hours**: Month 1: 169h (original 165h + 4h additions) | Month 2: 128h (original 120h + 8h additions)  
**Purpose**: Build PhD-level ML/Causal AI foundations + Complete Project #1

---

## 📋 MONTH 1 OVERVIEW

**CRITICAL UPDATE**: You need to LEARN advanced causal inference methods (IV, RDD, CATE) fresh, not just refresh. Month 1 causal time increased from 8h → 30h.

**🔴 INTEGRATION NOTE**: Month 1 has MINIMAL additions (just +4h) to the original plan. The main content below is preserved from the original battle plan with small enhancements marked as **[ADDED]**.

---

## MONTH 1: DEEP LEARNING + CAUSAL INFERENCE LEARNING

### **Week 1: Hybrid Intuition-Based Learning** (60 hours)

**WHY**: Build deep understanding through visual intuition + systematic coverage + problem-first learning + teaching to learn.

#### Tasks:

**LEARNING PHILOSOPHY**: Cover ALL material systematically WHILE building deep \"why-focused\" intuition.

#### 1. **Math Foundations for ML** (15 hours) ← **HYBRID: Visual + Coverage + Active**

**VISUAL FOUNDATION FIRST** (5h):
- **3Blue1Brown \"Essence of Linear Algebra\"** (3h)
  - Free: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
  - **Why**: Builds geometric intuition - matrices as transformations, eigenvalues as invariant directions
  - **Watch**: All 15 videos, handwritten notes
- **3Blue1Brown \"Essence of Calculus\"** (2h)
  - Chain rule visualization, gradients as \"slope in multiple dimensions\"

**SYSTEMATIC COVERAGE** (8h):
- **Goodfellow \"Deep Learning\"** - Chapters 2, 3, 4 (ALL chapters)
  - Ch 2 (Linear Algebra): 3h - verify your visual intuition from 3B1B
  - Ch 3 (Probability & Information Theory): 3h
  - Ch 4 (Numerical Computation): 2h - optimization, overflow
  - **Access**: ~$80 or free at deeplearningbook.org

**ACTIVE PRACTICE** (2h):
- **Try BEFORE reading solutions**:
  - Derive ridge regression closed form (30min attempt, then verify)
  - Sketch gradient descent proof (30min)
- **Goodfellow exercises**: Ch 2-4 selected problems (1h)

**RATIONALE**: 3B1B builds geometric intuition PhD didn't emphasize, Goodfellow gives rigorous foundations, practice cements understanding.

---

#### 2. **Probability & Statistics Foundations** (15 hours) ← **HYBRID: Problem-First + Coverage**

**PROBLEM-FIRST CHALLENGES** (5h):
**Before reading, attempt**:
1. **Derive Beta-Binomial conjugacy** (1.5h):
   - Given: Beta prior, Binomial likelihood
   - Try: Find posterior using Bayes' rule
   - **Then verify**: Casella Ch 3-4

2. **Sketch CLT proof** (1.5h):
   - Why do sums of RVs → Normal?
   - Try using moment generating functions
   - **Then read**: Casella Ch 5

3. **Derive MLE vs MAP** (2h):
   - Example: Coin flips
   - When are they equal? Why?
   - **Then verify**: Casella Ch 7

**SYSTEMATIC COVERAGE** (8h):
- **Casella & Berger \"Statistical Inference\"** - Chapters 1-5 (ALL chapters)
  - Ch 1-2 (Probability Theory, Transformations): 3h
  - Ch 3 (Common Families): 2h - exponential families
  - Ch 4 (Multiple Random Variables): 2h
  - Ch 5 (Properties of Random Samples): 1h - convergence
  - **Access**: ~$120

**[ADDED] +2h Distribution Relationships Extension**:
- **Geometric memoryless property** (prove it)
- **Poisson → Exponential connection** (derive inter-arrival times)
- Sum of exponentials = Gamma distribution
- **Resource**: Casella Ch 3, StatQuest videos

**FEYNMAN TEACHING** (2h):
- **Record yourself explaining** (5min each):
  - \"What's MLE and WHY does it work?\"
  - \"Explain sufficient statistics to a PM\"
  - \"Why is Beta-Binomial conjugate?\"
- **If you stumble** → re-read and re-derive

**SUPPLEMENT**: \"All of Statistics\" by Wasserman (~$60, quick reference)

**RATIONALE**: Problem-first mirrors your PhD approach (derived GEV before using it). Casella gives rigorous foundations. Feynman technique ensures understanding not memorization.

---

#### 3. **Causal Inference - Rigorous Foundations** (18 hours) ← **HYBRID: Intuition + Derive + Coverage**

**START WITH DAG INTUITION** (3h):
- **Pearl \"Book of Why\"** - Chapters 1, 3, 7 (~$18)
  - WHY causality ≠ correlation
  - DAG formalism, do-calculus
- **Draw 15 DAGs**: Education→Income (confounders?), Ad→Purchase (selection bias?)

**DERIVE METHODS YOURSELF** (6h):
**Before reading, design estimators**:

1. **Instrumental Variables** (2h):
   - Scenario: Want X→Y effect but X confounded
   - Given: Instrument Z affects X but not Y directly  
   - **Challenge**: How estimate effect using Z?
   - **Likely**: You'll derive something close to 2SLS
   - **Then verify**: Angrist Ch 3

2. **Difference-in-Differences** (2h):
   - Data: Treatment + control groups, before + after
   - **Challenge**: How estimate treatment effect?
   - **Likely path**: Difference, then difference-in-differences
   - **Then check**: Angrist Ch 4 (parallel trends)

3. **Regression Discontinuity** (2h):
   - Scholarship at GPA > 3.5, does it help?
   - **Challenge**: Use discontinuity for estimation
   - **Then learn**: Angrist Ch 5

**SYSTEMATIC COVERAGE** (9h):
- **Angrist & Pischke \"Mostly Harmless Econometrics\"** - Chapters 1-6 (ALL chapters, ~$55)
  - Ch 1-2 (Causal Questions, Regression): 2h
  - Ch 3 (Instrumental Variables): 2h - compare to YOUR derivation
  - Ch 4 (Difference-in-Differences): 1.5h - parallel trends deep dive
  - Ch 5 (Regression Discontinuity): 2h - bandwidth selection
  - Ch 6 (Quantile Regression): 1.5h - heterogeneous effects

**SUPPLEMENTS**:
- Imbens & Rubin \"Causal Inference...\" (~$75, potential outcomes deep dive)
- Mixtape by Cunningham (free, code examples)

**[ADDED] +2h MCMC Theory** (Casella Ch 11):
- Metropolis-Hastings algorithm
- Gibbs Sampling fundamentals
- Convergence diagnostics (R-hat, ESS)
- **Why**: Connects to Bayesian tooling in Month 2

**RATIONALE**: Pearl builds DAG intuition. Deriving estimators yourself = PhD-level understanding. Angrist provides rigor. You'll understand WHY assumptions matter, not just state them.

---

#### 4. **Advanced Causal Inference - CATE & Modern Methods** (8 hours)

**PRIMARY RESOURCE**: \"Causal Machine Learning\" course (Microsoft Research)
- Free: https://causal-machine-learning.github.io/
- **Focus**:
  - Conditional Average Treatment Effects (CATE)
  - Double Machine Learning (DML)
  - Meta-learners (S-learner, T-learner, X-learner, DR-learner)
- **Time**: Video lectures (4h) + papers (4h)

**Papers** (included in 8h):
- \"Double/Debiased Machine Learning\" (Chernozhukov et al. 2018) - 2h
- \"Metalearners for HTE\" (Künzel et al. 2019) - 1h

**Practice**:
- Implement S-learner, T-learner from scratch
- **Interview test**: \"Explain Double ML and WHY it works\"

**RATIONALE**: Project #2 uses CATE - need theoretical foundation before building.

---

#### 5. **Environment Setup** (3 hours)
- RTX 3090: CUDA 12, PyTorch 2.x, Ollama (Llama 3 8B)
- Tools: DoWhy, EconML, HuggingFace, LangChain, W&B
- **[ADDED]**: Install `statsmodels`, `prophet`, `hmmlearn`, `lifelines` (from Phase 0)
- OpenAI API key ($100 budget for GPT-4)

---

#### 6. **Coding Warm-up** (2 hours)
- 5 LeetCode problems (review mode - Arrays, Hashmaps)
- **Rationale**: Shake off rust, don't over-invest Week 1

---

### **WEEK 1 TOTAL**: 64 hours (original 60h + 4h additions)

**Breakdown**:
- Math (15h): 3B1B (5h) + Goodfellow Ch 2-4 (8h) + Practice (2h)
- Stats (17h): Problem-first (5h) + Casella Ch 1-5 (8h) + Distribution relationships (+2h) + Feynman (2h)
- Causal (20h): Pearl (3h) + Derive yourself (6h) + Angrist Ch 1-6 (9h) + MCMC (+2h)
- Advanced CATE (8h)
- Environment (3h)
- Coding (2h)

**DELIVERABLES**:
- ✅ Visual geometric intuition (3Blue1Brown)
- ✅ ALL book chapters covered (Goodfellow 2-4, Casella 1-5, Angrist 1-6)
- ✅ Derived methods from first principles (IV, DID, RDD)
- ✅ Taught concepts via Feynman technique (recorded explanations)
- ✅ Deep \"why-focused\" understanding, not memorization
- ✅ **[ADDED]** Distribution relationships mastered, MCMC fundamentals

---

### **Week 2-3: Deep Learning + CATE - Hybrid Intuition Approach** (30 hours/week = 60h total)

**WHY**: Build deep understanding of neural networks and CATE estimation through first-principles derivation and implementation.

#### Tasks:

**LEARNING PHILOSOPHY**: Derive before implementing, understand WHY before WHAT.

#### 1. **Deep Learning - Theory First** (16 hours over 2 weeks) ← **HYBRID**

**DERIVE BACKPROP YOURSELF** (4h):
- **Challenge**: Given a 2-layer neural network, derive backprop from scratch
- **Start with**: Chain rule, computational graph
- **Try to derive**: Weight update rules without looking at solutions
- **Then verify**: Goodfellow Ch 6 or 3B1B backprop video
- **Why**: Interviews ask \"derive backprop\" - you need to build it from first principles

**SYSTEMATIC COVERAGE** (8h):
- **Goodfellow \"Deep Learning\"** - Chapters 6-8 (continuation):
  - Ch 6 (Deep Feedforward Networks): 3h - verify your backprop derivation
  - Ch 7 (Regularization): 3h - WHY dropout works, WHY batch norm
  - Ch 8 (Optimization): 2h - SGD variants, why momentum, why Adam
- **Reading method**: For each technique, ask \"WHY does this work?\"

**ACTIVE IMPLEMENTATION** (4h):
- **From scratch**: Implement 2-layer NN in NumPy (no PyTorch)
- **Then**: Implement dropout, batch norm manually
- **Compare**: PyTorch implementation to verify understanding

**RATIONALE**: Deriving backprop = PhD-level understanding. You'll explain WHY gradients vanish, not just that they do.

---

#### 2. **Transformer Architecture - Build Intuition** (14 hours) ← **WHY-FOCUSED**

**VISUAL FIRST** (2h):
- **3Blue1Brown \"Attention in transformers\"** (if available)
- **Jay Alammar \"The Illustrated Transformer\"** (visual guide)
- **Understand**: WHY attention? What problem does it solve (vs RNNs)?

**IMPLEMENT FROM SCRATCH** (8h):
- **Karpathy \"Let's build GPT\"** (video 6h + implementation 2h)
- **Build**: Transformer block from zero
- **Understand every line**: WHY scaled dot-product? WHY layer norm here?

**PAPER DEEP DIVE** (4h):
- **\"Attention Is All You Need\"** + **GPT-2 paper**
- **Annotate with WHY**:
  - WHY multi-head attention? (different representation subspaces)
  - WHY positional encoding? (no recurrence = no order info)
  - WHY layer norm not batch norm? (sequence length varies)

**INTERVIEW PREP**:
- **Can you explain**: \"Walk me through attention mechanism on whiteboard\"
- **Can you derive**: Attention formula, complexity O(n²)

**RATIONALE**: AS interviews test transformer internals. You'll know WHY modern LLMs work, not just how to use APIs.

---

#### 3. **CATE Estimation - Design Your Own** (12 hours) ← **PROBLEM-FIRST**

**CHALLENGE YOURSELF** (4h):
- **Problem**: You have treatment, outcome, and features. How would YOU estimate heterogeneous effects?
- **Likely approach**: 
  - Naive: Separate models for treated/control (you'll invent T-learner!)
  - Or: Single model with treatment interaction (you'll invent S-learner!)
- **Try** Implement your approach on toy data

**THEN LEARN FORMAL METHODS** (6h):
- **Microsoft Research \"Causal ML\" course** (lectures):
  - S-learner: Single model with treatment as feature
  - T-learner: Separate models (what you likely invented!)
  - X-learner: More sophisticated (propensity weighting)
  - DR-learner: Doubly robust (combines everything)
- **Compare**: How does your approach differ? Why are formal methods better?

**IMPLEMENT PROPERLY** (2h):
- **From scratch**: S-learner, T-learner in NumPy
- **Then compare**: EconML implementations
- **Understand**: When does each meta-learner shine?

**PAPERS** (included in 6h):
- \"Metalearners for HTE\" (Künzel 2019) - WHY X-learner works

**RATIONALE**: You invented statistical methods in PhD by solving problems first. Same approach here - design estimators, then learn formal theory.

---

#### 4. **Coding Maintenance** (7 hours over 2 weeks)
- 5 problems/week (10 total)
- **Focus**: Trees, Graphs
- **Approach**: Understand pattern, not memorize

---

#### 5. **Feynman Teaching** (3 hours over 2 weeks)
- **Record yourself** (10min each):
  - \"How does backprop work?\" (whiteboard)
  - \"Why attention mechanism?\" (explain to PM)
  - \"When to use T-learner vs X-learner?\" (decision tree)

---

### **WEEKS 2-3 TOTAL**: 60 hours (30h/week) - NO CHANGES from original

**Breakdown**:
- Deep Learning theory + implementation: 16h
- Transformer from scratch: 14h
- CATE (design your own + formal methods): 12h
- Papers deep dive: 4h (embedded in above)
- Coding: 7h
- Feynman teaching: 3h
- Buffer: 4h

**DELIVERABLES**:
- ✅ Derived backprop from first principles
- ✅ Implemented transformer from scratch (understand every line)
- ✅ Designed your own CATE estimator (then learned formal methods)
- ✅ Can explain WHY each technique works (not just apply it)

---

### **Week 4: LLM Tools + Causal Libraries - Understanding the WHY** (35 hours)

**WHY**: Prepare for projects with deep tool understanding, not just API memorization.

#### Tasks:

**LEARNING PHILOSOPHY**: For each tool, understand WHY it exists and what problem it solves.

#### 1. **HuggingFace - Understanding Internals** (10 hours) ← **HYBRID**

**WHY-FOCUSED LEARNING** (4h):
- **Before using HF**: Understand tokenization from first principles
  - WHY subword tokenization? (vs word-level, char-level)
  - Try: Implement BPE tokenizer manually (30min sketch)
  - **Then**: Use HF tokenizer, understand what it's doing
- **Model loading**: WHY checkpoint formats? What's actually saved?

**PRACTICAL MASTERY** (6h):
- **HuggingFace NLP Course** - Chapters 1-4
- **Hands-on**: Fine-tune DistilBERT on sentiment analysis
- **Understand**: What layers are frozen? Why? What's being updated?

**RATIONALE**: Knowing WHY tokenization works > just calling `tokenizer.encode()`

---

#### 2. **DoWhy - Causal Graph Understanding** (8 hours) ← **THEORY-FIRST**

**UNDERSTAND THE FRAMEWORK** (3h):
- **WHY DoWhy exists**: What problem does it solve?
- **Key insight**: Separates causal model from estimation
- **Compare**: DoWhy vs manual implementation (when would you use each?)

**PRACTICAL IMPLEMENTATION** (5h):
- **All identification strategies**:
  - Backdoor adjustment (confounding)
  - IV estimation
  - RDD
  - Frontdoor criterion
- **For each**: Draw DAG, understand WHY DoWhy chose this estimand
- **Deliverable**: Notebook with 5 examples + your annotations

**RATIONALE**: You derived IV/DID yourself in Week 1. Now see how DoWhy implements them.

---

#### 3. **EconML - Meta-learner Deep Dive** (9 hours) ← **COMPARE YOUR DESIGN**

**RECALL YOUR CATE DESIGN** (1h):
- Review: What meta-learner did YOU invent in Weeks 2-3?
- **Hypothesis**: You likely designed something close to T-learner

**SYSTEMATIC COMPARISON** (6h):
- **Implement all 4 meta-learners** on same dataset:
  - S-learner (single model)
  - T-learner (separate models - your design?)
  - X-learner (propensity weighting)
  - DR-learner (doubly robust)
- **For each**: Understand WHY it works, when it fails
- **Compare**: Your design vs EconML. What did you miss? Why?

**THEORY VERIFICATION** (2h):
- **Read**: EconML source code for one meta-learner
- **Understand**: Implementation details, numerical stability tricks

**RATIONALE**: You invented estimators in Week 2-3. Now see production implementation.

---

#### 4. **LangChain - Agent Architecture** (6 hours) ← **CONCEPTUAL FIRST**

**WHY AGENTS?** (2h):
- **Problem**: LLMs can't use tools directly
- **Solution**: Agent frameworks (LangChain, LlamaIndex, CrewAI)
- **Understand**: ReAct pattern (Reasoning + Acting)

**PRACTICAL** (4h):
- **DeepLearning.AI \"LangChain for LLM Apps\"**
- **Focus**: Chains, prompts, agents, memory
- **Build**: Simple agent with tools (calculator, search)

**RATIONALE**: Project #1 (CausalRAG) needs LangChain. Understand architecture now.

---

#### 5. **Coding + Feynman** (2 hours)
- 5 LeetCode problems (Binary Search, Backtracking)
- **Teach**: Record \"Why does BPE tokenization work?\" (5min)

---

### **WEEK 4 TOTAL**: 35 hours - NO CHANGES from original

**Breakdown**:
- HuggingFace (understand WHY): 10h
- DoWhy (all methods + theory): 8h
- EconML (compare to your design): 9h
- LangChain (agent architecture): 6h
- Coding: 2h

**DELIVERABLES**:
- ✅ Understand WHY tokenization works (not just how to use HF)
- ✅ Implemented all DoWhy identification strategies with understanding
- ✅ Compared your meta-learner design to EconML implementations
- ✅ Built simple LangChain agent, understand ReAct pattern

---

### **Week 4 Weekend: [NEW] System Design Session #1** (4h)

**Preparation (2h)**:
- **Read**: "System Design Interview" (Alex Xu) - Chapters 1 (Scale from Zero to Millions) & 4 (Rate Limiter basics).
- **Focus**: Understand Load Balancers, API Gateways, and why Latency matters.
- **Why**: You cannot design a service without knowing what a "Service" is.

**Goal**: Start the "System Design interaction loop" early. Don't wait until Month 7.

**Topic: "Design a Real-time Causal Inference Service"**
- **Scenario**: Uber wants to estimate treatment effect (discount) for a user in < 200ms when they open the app.
- **Constraints**: 100M DAU, low latency, high availability.

**Activities**:
1. **Back-of-envelope**: Estimate QPS (Queries Per Second).
2. **High-Level Design**:
   - Client -> Load Balancer -> CATE Service -> Model Store
   - Feature Store (Redis) for real-time user features
3. **Deep Dive**:
   - How to simple serve S-learner? (ONNX runtime?)
   - How to update models daily? (Airflow pipeline)
   
**Deliverable**: Whiteboard diagram (photo saved to `/design/session1_causal_serving.png`)

---

## MONTH 1 DELIVERABLE:
- ✅ **Visual + Systematic Learning**: 3B1B + ALL book chapters (Goodfellow 2-8, Casella 1-5, Angrist 1-6)
- ✅ **Derived from First Principles**: Backprop, IV, DID, RDD, ridge regression, Beta-Binomial
- ✅ **Designed Your Own**: CATE meta-learner (before learning formal methods)
- ✅ **Implemented from Scratch**: Transformer, S/T-learners, 2-layer NN
- ✅ **Deep Understanding**: Can explain WHY each method works, not just apply it
- ✅ **Teaching Practice**: Feynman technique recordings for all major concepts
- ✅ 22 NeetCode problems solved (maintaining proficiency)
- ✅ **[ADDED]** MCMC theory (R-hat, ESS), Distribution relationships

**TOTAL MONTH 1**: ~169 hours (original 165h + 4h additions) = **42.25h/week**

**WHY THIS WORKS (Your PhD Pattern)**:
1. **Visual First**: 3B1B builds intuition (like understanding physics of traffic flow)
2. **Derive Before Reading**: IV/DID/backprop from scratch (like deriving GEV model)
3. **Design Your Own**: Invent CATE estimator (like inventing safety metrics)
4. **Then Learn Formally**: Books verify your intuition (like reading methods papers after solving problem)
5. **Teach to Cement**: Feynman technique (like teaching students solidified your PhD knowledge)

**Investment**:
- **Books**: ~$255 (Goodfellow $80 + Casella $120 + Angrist $55)
- **Time**: 169h @ $500K salary = opportunity cost ~$41K, but 10x ROI in interview performance

**Result**: PhD-level understanding of ML/Causal AI in 4 weeks. Can derive on whiteboard, explain trade-offs, critique assumptions.

---

## MONTH 2: LLM FINE-TUNING + BAYESIAN TOOLING + PROJECT #1

**🔴 INTEGRATION NOTE**: Month 2 adds a NEW Week 7 (Bayesian Tooling, 6h) BEFORE the original Week 7-8 (Project #1).

---

### **Week 5-6: Fine-Tuning Mastery** (28-30 hours/week) - NO CHANGES

#### Tasks:
1. **Fine-Tuning Practice** (16 hours over 2 weeks)
   - **Method**: LoRA fine-tuning of Llama 3 8B
   - **Dataset**: ArXiv papers on causal inference (use Hugging Face datasets)
   - **Goal**: Fine-tune model to answer causal inference questions
   - **Evaluation**: Perplexity, sample outputs
   
   **RATIONALE**: Demonstrates ability to adapt LLMs. May be useful for projects.
   
   **ALTERNATIVE**: Skip fine-tuning, use base models → ⚠️ **MISSING SKILL**. AS interviews often ask about fine-tuning trade-offs.

2. **Reading**: LoRA, QLoRA, PEFT papers (4 hours)
   - Understand parameter-efficient methods
   
   **RATIONALE**: \"Why does LoRA work?\" is an interview question.

3. **Causal Inference Deep Dive** (8 hours/week × 2 weeks = 16 hours)
   - **Resource**: \"The Effect\" by Nick Huntington-Klein (online book)
   - **Focus**: Chapters on IV, DID, RDD, Synthetic Control
   - **Practice**: Implement each method from scratch in NumPy
   
   **RATIONALE**: Interviews: \"Derive the 2SLS estimator.\" Need to go beyond DoWhy API.
   
   **ALTERNATIVE**: Just use library functions → ❌ **SHALLOW**. AS roles expect deep understanding.

4. **Coding Maintenance** (8 hours total)
   - 10 problems/week, Dynamic Programming focus

---

### **Week 7: [NEW] Bayesian Tooling & Advanced Methods** (6 hours)

**🔴 THIS IS A NEW ADDITION** - Fills critical gap identified in research (116 Stan mentions, GP questions)

**Schedule**:

#### **Monday (2h): Stan Basics**
- **Theory**: Stan vs PyMC comparison (when to use each)
- **Bayesian workflow**: Prior predictive → Fit → Posterior predictive → MCMC diagnostics
- **Hands-on**: Implement hierarchical Bayesian model in Stan
  - Example: Hierarchical model for A/B test across multiple markets
- **Resource**: Stan User's Guide Ch 1-2
- **Deliverable**: Stan .stan file for hierarchical model

#### **Tuesday (2h): Variational Inference**
- **Theory**: ELBO derivation (Evidence Lower Bound)
  - KL divergence between variational distribution q(z) and true posterior p(z|x)
  - Why VI? (Faster than MCMC, scales to big data)
- **ADVI**: Automatic Differentiation Variational Inference
- **Hands-on**: Implement simple VI in PyTorch
  - Use PyTorch distributions library
  - Optimize ELBO via gradient ascent
- **Resource**: Bishop Ch 10 (if available), PyTorch tutorials
- **Deliverable**: VI implementation notebook

#### **Wednesday (2h): Gaussian Processes**
- **Theory**: GP regression basics
  - Prior: function is drawn from GP
  - Posterior: condition on observed data
  - RBF (Radial Basis Function) kernel intuition
- **Bayesian optimization**: GP for hyperparameter tuning
- **Hands-on**: GP regression with `scikit-learn.gaussian_process`
- **Exercise**: Use GP for Bayesian hyperparameter optimization
- **Resource**: scikit-learn GP tutorial, Rasmussen & Williams book (reference)
- **Deliverable**: GP regression + Bayesian optimization notebook

**Integration Points**:
- Stan: Use in Project #3 (Bayesian A/B test analysis)
- VI: Understanding for LLM fine-tuning (variational methods)
- GP: Useful for Project #3 Sequential Design Agent (Bayesian optimization)

**Resources**:
- Stan User's Guide: https://mc-stan.org/docs/stan-users-guide/
- PyTorch Distributions: https://pytorch.org/docs/stable/distributions.html
- Scikit-learn GP: https://scikit-learn.org/stable/modules/gaussian_process.html

**Success Criteria**:
- [ ] Can choose between Stan vs PyMC for different problems
- [ ] Can explain ELBO and when VI beats MCMC
- [ ] Can use GP for Bayesian optimization

---

### **Week 8-9: PROJECT #1 BUILD** (35 hours/week = 70h total)

**🔴 TIMELINE NOTE**: What was originally "Week 7-8" is now "Week 8-9" due to Bayesian tooling insertion.

**PROJECT #1: CausalRAG - Enterprise Search with Causal Reasoning**

**OBJECTIVE**: Build a graph-based retrieval system that answers \"why\" questions by discovering causal chains, not just matching keywords.

**THE PROBLEM**: Standard RAG systems hallucinate on causal queries. Ask \"Why did revenue drop?\" and it retrieves documents with \"revenue\" + \"drop\" but misses the *causal mechanism* if described differently (e.g., \"margin compression\").

**OUR SOLUTION**: Extract causal relationships (A → B) during indexing, retrieve by traversing causal graphs, use Chain-of-Retrieval (CoRAG) for multi-step discovery.

#### **Core Architecture**:
1. **Phase 1: Causal Graph Construction** (20h)
   - Dataset: Amazon Reviews 2023 (500K reviews subset, 571M available)
   - LLM extraction: GPT-4 or Llama-3-70B extracts \"Feature X → Rating Y\" edges
   - Store: Neo4j (causal graph) + Milvus (vector embeddings)
   - Validate: Time-ordering constraints, remove cycles

2. **Phase 2: Chain-of-Retrieval (CoRAG)** (25h)
   - Query matching: \"Why low ratings?\" → Match to \"Low Rating\" node
   - Causal expansion: BFS upstream 3 hops to find causes
   - Path scoring: LLM evaluates discovered causal chains
   - Iterative retrieval: Multiple steps like a detective

3. **Phase 3: Integration & Demo** (15h)
   - Causally-informed generation: LLM answers grounded in graph
   - Streamlit demo: Visualize causal chains, show reasoning
   - Reduces hallucination: Can't invent causal links

4. **Phase 4: Benchmarking** (10h)
   - StrategyQA dataset (2,780 reasoning questions)
   - Custom Amazon Reviews benchmark (100 \"why\" questions)
   - Metrics: Exact Match, Faithfulness, Hallucination Rate
   - Expected: +10-15% vs standard RAG

#### **Tech Stack**:
- **Framework**: LangChain (orchestration), NetworkX (graph algorithms)
- **Storage**: Neo4j (causal graph), Milvus/Chroma (vectors)
- **LLM**: GPT-4 for extraction + generation, Llama-3-8B for cost savings
- **Demo**: Streamlit Cloud

#### **Why This Design** (vs Generic RAG):
- ✅ **Not just vector search**: Causal graph traversal discovers hidden mechanisms
- ✅ **Publishable**: ACL 2025 paper exists, we implement + benchmark = ACL 2026 target
- ✅ **Directly solves $B problem**: Google Search hallucinations, Amazon Q analyst bottleneck
- ✅ **Interview gold**: \"I built a system that reduces hallucinations by 12% using causal reasoning\"

#### **Technology Stack (Enhanced)**:
- **Framework**: LangChain, NetworkX
- **Storage**: Neo4j, Milvus
- **Engineering**: Docker, FastAPI, Pydantic, Pre-commit hooks

#### **[ADDED] Engineering Rigor Requirements**:
To pass the "Applied Scientist" bar, this project must be **Production-Ready**:
1. **Containerization**:
   - **Dockerfile**: Multi-stage build (python:slim), optimized layers.
   - **Docker Compose**: Spin up Neo4j + Milvus + App with one command (`docker-compose up`).
2. **API Standard**:
   - **FastAPI**: Expose retrieval as REST endpoint (`POST /retrieve`).
   - **Pydantic**: Strict typing for request/response.
   - **Swagger/OpenAPI**: Auto-generated documentation.
3. **Code Quality**:
   - **Pre-commit hooks**: `black` (formatting), `ruff` (linting).
   - **Type hints**: 100% type coverage (`mypy` strict mode).

#### **Target Companies**:
- Google (AI Overviews), Amazon (Q), Microsoft (Copilot), Meta (Search)

#### **Time Breakdown** (70 hours total):
- Week 8 (35h): 
  - Graph construction pipeline (15h)
  - LLM extraction on 500K reviews (10h)
  - Neo4j setup + validation (10h)
- Week 9 (35h): 
  - CoRAG retrieval logic (15h)
  - Streamlit demo + visualization (10h)
  - StrategyQA benchmarking (10h)

#### **Deliverables**:
- GitHub repo: Graph construction + CoRAG retrieval + evaluation
- Streamlit demo: Live causal chain visualization
- Technical report: EM scores, hallucination rates vs baselines
- Optional: Workshop paper draft (for ACL 2026 submission in Month 6)

#### **Evaluation** (CRITICAL - 5 hours in Week 9):
- **Test Set**: 50 causal inference questions (e.g., \"What's the effect of education on earnings?\")
- **Metrics**:
  - Method selection accuracy (did agent choose correct approach?)
  - Reasoning quality (scored 1-5 by you + GPT-4-as-judge)
  - Estimate accuracy (compare to known results from papers)
- **W&B Weave Integration**: Track agent executions, visualize performance

**WHY EVALUATION MATTERS**: AS interviews ask \"How do you know it works?\" Need rigorous answer.

#### **Documentation**:
- README with architecture diagram (Mermaid)
- Demo video (3-5 min)
- **Blog post**: \"Building an AI Agent for Causal Inference\" (8h)

#### **ALTERNATIVE DESIGNS**:

**Option A** (Current): Multi-agent LangChain system
- **Pros**: Flexible, good for complex reasoning
- **Cons**: Can be slow, harder to debug

**Option B**: Single LLM with tool use (function calling)
- **Pros**: Simpler, faster
- **Cons**: Less sophisticated reasoning, harder to separate concerns
- **Verdict**: ⚠️ **Use for Project #2 instead** to show framework breadth

**Option C**: Use CrewAI instead of LangChain
- **Pros**: Better multi-agent orchestration, faster
- **Cons**: Less flexible for custom chains
- **Verdict**: ⚠️ **Save CrewAI for Project #3**

**Coding Maintenance**: 8 hours (over 2 weeks for Project #1)

---

## MONTH 2 DELIVERABLE:
- ✅ Fine-tuned Llama 3 model for causal inference QA
- ✅ **[ADDED]** Stan hierarchical model, VI implementation, GP regression notebook
- ✅ **Project #1 complete** (live GitHub repo + demo + blog post)
- ✅ W&B Weave evaluation dashboard
- ✅ 80 NeetCode problems total (maintaining sharpness)

**TOTAL MONTH 2**: ~128 hours (original 120h + 8h Bayesian tooling) = **32h/week**

---

## MONTHS 1-2 SUMMARY

**Total Time**: 297 hours (169h Month 1 + 128h Month 2)
**Average**: 37h/week over 8 weeks
**Additions**: +12h total (+4h Month 1, +8h Month 2)

**Critical Skills Gained**:
1. PhD-level ML theory (backprop derivation, transformer from scratch)
2. Rigorous causal inference (IV, DID, RDD from first principles)
3. Modern CATE methods (S/T/X/DR-learners)
4. **[ADDED]** Classical ML (from Phase 0: ARIMA, HMMs, EM, ICA)
5. **[ADDED]** Bayesian tooling (Stan, VI, GPs)
6. LLM tools (HuggingFace, LangChain, DoWhy, EconML)
7. First hybrid project (CausalRAG)

**Interview Readiness** (Month 1-2):
- Can derive on whiteboard: backprop, IV, DID, attention mechanism
- Can implement from scratch: transformer, Viterbi, EM, CATE estimators
- Can explain WHY: every method, not just API usage
- 1 project complete with rigorous evaluation

---

**Next**: Proceed to Months 3-5 (file: `03_battle_plan_months_3-5.md`) for Projects #2, #3, #4

**Status**: ✅ Months 1-2 plan complete with all original content + targeted additions
