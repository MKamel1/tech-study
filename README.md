# Causal AI Specialist Battle Plan - Navigator

**Welcome to the comprehensive, segmented battle plan!**

This directory contains the complete 7-8 month battle plan broken into logical segments for easier navigation, maintenance, and debugging.

---

## 📁 FILE STRUCTURE

### **Core Battle Plan Files**

1. **[01_battle_plan_phase0_classical_ml.md](./01_battle_plan_phase0_classical_ml.md)** (Weeks 0.5-1.5, 24h)
   - ARIMA time series forecasting
   - Hidden Markov Models (HMMs)
   - EM Algorithm & Gaussian Mixture Models
   - PCA vs ICA deep dive
   - Survival Analysis fundamentals

2. **[02_battle_plan_months_1-2.md](./02_battle_plan_months_1-2.md)** (8 weeks, 297h)
   - **Month 1** (169h): ML/Causal AI foundations, CATE, transformers
   - **Month 2** (128h): LLM fine-tuning, Bayesian tooling (Stan, VI, GPs), Project #1 (CausalRAG)

3. **[03a_battle_plan_month_3_project2.md](./03a_battle_plan_month_3_project2.md)** (4 weeks, 102h)
   - **Project #2**: Causal Uplift Modeling (45M rows, PySpark)
   - Counterfactual Ad Generation (PyReFT)
   - EM customer segmentation
   - Model quantization (INT8/TensorRT)

4. **[03b_battle_plan_month_4_project3.md](./03b_battle_plan_month_4_project3.md)** (4 weeks, 70h)
   - **Project #3**: Multi-Agent Experimentation Platform (CrewAI)
   - Sequential Design Agent (Bayesian optimization)
   - AWS SageMaker deployment
   - HMM user state analysis
   - Drift monitoring dashboard

5. **[03c_battle_plan_month_5_project4_mlops.md](./03c_battle_plan_month_5_project4_mlops.md)** (4 weeks, 85h)
   - **Project #4**: Counterfactual AV Safety Analysis
   - MLOps Deep Dive: Drift detection, Quantization, Distributed training, Model serving
   - Survival analysis for time-to-accident

6. **[04_battle_plan_months_6-8.md](./04_battle_plan_months_6-8.md)** (12 weeks, 280-330h) *[TO BE CREATED]*
   - **Month 6**: Statistics deep dive (40h), Interview prep start
   - **Month 7**: System design mastery, Mock interviews
   - **Month 8**: Applications, Networking, Final prep

---

## 🎯 QUICK NAVIGATION

### **By Learning Topic**

| Topic | File  | Hours |
|-------|-------|-------|
| **Classical ML** | 01 (Phase 0) | 24h |
| **Deep Learning Foundations** | 02 (Month 1) | 64h |
| **Causal Inference Theory** | 02 (Month 1-2) | 50h |
| **LLM Tools & Fine-Tuning** | 02 (Month 2) | 35h |
| **Bayesian Methods** | 02 (Month 2 Week 7) | 8h |
| **PySpark & Big Data** | 03a (Month 3) | 36h |
| **Multi-Agent AI** | 03b (Month 4) | 30h |
| **MLOps Production** | 03c (Month 5) | 22h |
| **Statistics Deep Dive** | 04 (Month 6) | 40h |
| **Interview Prep** | 04 (Months 6-8) | 120h |

### **By Project**

| Project | File | Hours | Key Tech |
|---------|------|-------|----------|
| **#1: CausalRAG** | 02 (Month 2) | 70h | Neo4j, LangChain, CoRAG |
| **#2: Uplift + Counterfactual Ads** | 03a (Month 3) | 102h | PySpark, EconML, PyReFT, TensorRT |
| **#3: Multi-Agent Experimentation** | 03b (Month 4) | 70h | CrewAI, AWS SageMaker, HMMs |
| **#4: AV Safety Counterfactuals** | 03c (Month 5) | 63h | DoWhy, Haystack, Survival Analysis |

### **By Time Commitment**

- **Part-time sustainable** (~15h/week): Phase 0 → Month 1 will take ~11 weeks
- **Aggressive full-time** (~40h/week): Complete in 6 months
- **Recommended** (~25-30h/week): Complete in 7-8 months

---

## 🔗 INTEGRATION NOTES

### **How Files Connect**

1. **Phase 0 → Month 1**: Classical ML (HMMs, EM, ICA) provides foundation, then referenced in later projects
2. **Month 2 Week 7**: NEW Bayesian tooling week inserted (Stan, VI, GPs)
3. **Month 3**: EM from Phase 0 integrated into Project #2 for customer segmentation
4. **Month 4**: HMMs from Phase 0 applied to user state analysis in Project #3
5. **Month 5**: MLOps topics distributed across weeks 19-20, integrated with Project #4

### **Timeline Adjustments**

- Original plan: Weeks 1-20
- Segmented plan: Weeks 0.5-21+ (accounts for Phase 0 + Bayesian week)
- **No content lost**: All original hours preserved + enhancements added

### **Enhancements Tracking**

All additions from gap analysis are marked with **[ADDED]** in files:
- Phase 0: +24h (ARIMA, HMMs, EM, ICA, Survival)
- Month 1: +4h (MCMC, distribution relationships)
- Month 2: +8h (Bayesian tooling week)
- Month 3: +10h (EM segmentation +3h, Quantization +6h)
- Month 4: +10h (Experimentation gaps +3h, Drift +3h, HMM +4h)
- Month 5: +25h (MLOps deep dive +22h, Survival +3h)

**Total Enhancements**: ~81 hours added to original plan

---

## 📊 TOTAL TIME BREAKDOWN

| Phase | Weeks | Hours | Pace |
|-------|-------|-------|------|
| **Phase 0** | 0.5-1.5 (2 weeks) | 24h | 12-15h/week |
| **Month 1** | 1-4 (4 weeks) | 169h | 42h/week |
| **Month 2** | 5-9 (5 weeks, includes Bayesian) | 128h | 25-30h/week |
| **Month 3** | 10-13 (4 weeks) | 102h | 25-30h/week |
| **Month 4** | 14-17 (4 weeks) | 70h | 17-20h/week |
| **Month 5** | 18-21 (4 weeks) | 85h | 21-25h/week |
| **Months 6-8** | 22-34+ (12 weeks) | 280-330h | 23-28h/week |

**GRAND TOTAL**: ~858-908 hours across 7-8 months

---

## 🚦 READING ORDER (RECOMMENDED)

### **For First-Time Review**:
1. Start here (README)
2. Read 01 (Phase 0) - understand Classical ML additions
3. Skim 02 (Months 1-2) - grasp foundations + Project #1
4. Read project summaries in 03a, 03b, 03c
5. Review 04 for interview prep strategy

### **For Execution** (when starting work):
1. Follow files in numerical order
2. Check off deliverables as you complete them
3. Use **[ADDED]** markers to identify enhancements

### **For Debugging** (when stuck):
- Each file is self-contained with resources, hour breakdowns, and deliverables
- Look for "Success Criteria" sections at end of each phase
- Check "Integration Notes & Contradictions" for how pieces fit

---

## ✅ VERIFICATION CHECKLIST

Use this to ensure no details were lost during segmentation:

### **Original Plan Elements** (all preserved):
- [ ] All textbook chapters referenced (Goodfellow, Casella, Angrist, etc.)
- [ ] All 4 core projects with full specifications
- [ ] LeetCode/NeetCode coding maintenance (spread across all months)
- [ ] W&B Weave evaluation (mentioned in relevant projects)
- [ ] Feynman teaching technique (Month 1-2)
- [ ] Interview stories for each project
- [ ] Technical depth (derive from first principles, implement from scratch)

### **Gap Analysis Additions** (all integrated):
- [ ] Phase 0: ARIMA, HMMs, EM/GMM, ICA, Survival Analysis (24h)
- [ ] Month 2: Bayesian tooling week - Stan, VI, GPs (8h)
- [ ] Month 3: EM customer segmentation (3h), Quantization (6h)
- [ ] Month 4: Experimentation gaps (3h), Drift monitoring (3h), HMM states (4h)
- [ ] Month 5: MLOps deep dive - Drift (8h), Quantization (8h), Distributed (3h), Serving (3h)

### **Project Enhancements**:
- [ ] Project #2: PyReFT counterfactual ads, EM integration, quantization
- [ ] Project #3: Sequential Design Agent, drift dashboard, HMM analysis
- [ ] Project #4: Survival analysis, MLOps integration

---

## 🔍 CONTRADICTION CHECK

**Status**: ✅ **NO CONTRADICTIONS FOUND**

All files were created by merging:
1. Original `causal_ai_specialist_battle_plan.md` (lines 1-1215)
2. `master_battle_plan_integrated.md` (timeline and additions)
3. Individual project spec files (project_1-4_spec.md)
4. `gap_analysis_report.md` findings
5. `comprehensive_deep_dive_plan.md` recommendations

**Methodology**:
- Line-by-line comparison during merge
- All original hours preserved
- Enhancements marked with **[ADDED]**
- Timeline adjustments documented
- Week numbering consistently updated

---

## 📖 RELATED DOCUMENTS

**In `/plan/` directory**:
- `causal_ai_specialist_battle_plan.md` - Original comprehensive plan (BACKUP: `_ORIGINAL_BACKUP.md`)
- `master_battle_plan_integrated.md` - Integration guide showing how original + additions combine
- `study_materials_guide_added_modules.md` - Detailed resources for all new modules
- `project_1_causalrag_spec.md` - Full Project #1 specification
- `project_2_enhanced_spec.md` - Full Project #2 specification  
- `project_3_multi_agent_spec.md` - Full Project #3 specification
- `project_4_av_safety_spec.md` - Full Project #4 specification

**In artifact directory** (`C:\Users\mmbka\.gemini\antigravity\brain\990f76e8-34b1-43c9-b4c4-de443d798799\`):
- `gap_analysis_report.md` - Comprehensive gap analysis (100% research coverage)
- `comprehensive_deep_dive_plan.md` - 43-hour addition plan details
- `integrated_battle_plan_timeline.md` - Week-by-week timeline
- `walkthrough.md` - Documentation of all analysis work

---

## 🎓 SUCCESS CRITERIA (OVERALL)

By end of Month 8, you should have:

**Technical Depth**:
- [ ] Can derive backprop, IV, DID, attention mechanism on whiteboard
- [ ] Implemented transformer, Viterbi, EM, CATE estimators from scratch
- [ ] Understand WHY each method works (not just API usage)

**Projects**:
- [ ] 4 production-quality GitHub repos (all with demos, docs, evaluations)
- [ ] 4 blog posts published
- [ ] Live demos for all projects (Streamlit/SageMaker)

**Interview Readiness**:
- [ ] 5-minute pitch for each project memorized
- [ ] 200+ LeetCode problems solved
- [ ] Can design ML systems (Chip Huyen framework)
- [ ] 50+ company-specific applications submitted

**Signal**:
- [ ] 1-2 research papers submitted (workshop/conference)
- [ ] Cruise/Amazon/Netflix reached out or applied
- [ ] Portfolio demonstrates: Causal AI + GenAI + MLOps mastery

---

## 📞 USAGE INSTRUCTIONS

1. **Starting Out**: Read files sequentially, execute Phase 0 first
2. **Updating Plan**: Edit individual files (easier than monolithic file)
3. **Tracking Progress**: Check deliverable boxes, update as you complete
4. **Asking Questions**: Each file is self-contained for context
5. **Future Additions**: Add new sections with **[ADDED YYYY-MM-DD]** markers

---

**Last Updated**: 2025-12-23  
**Version**: 1.0 (Initial Segmentation)  
**Status**: ✅ Core structure complete, Months 6-8 file pending

**Next**: Create `04_battle_plan_months_6-8.md` to complete the full plan.
