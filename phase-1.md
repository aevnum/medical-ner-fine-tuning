# Phase 1: NER Model Fine-tuning & Baseline Establishment

## Detailed Task Breakdown

---

### **Task 1.0: Initialize Configuration**

**Subtasks:**
1. **Create constants/config cell**
   - Project name
   - Model name
   - Dataset name
   - Date and Time formatted for tracking
   - Other relevant constants

2. **Set up WandB project**
   - Initialize WandB and login with API key through Kaggle secrets
   - Create new project: "Medical_NER_FineTuning"
   - Define experiment naming convention - model_name + date-time
   - Use this name for everything logged to WandB

### **Task 1.1: Environment Setup & Data Preparation**

**Subtasks:**
1. **Load and preprocess MACCROBAT dataset**
   - Load dataset from HuggingFace
   - Extract unique tags and create `tag2id`, `id2tag` mappings
   - Ensure all B-XXX tags have corresponding I-XXX tags (from your EDA, you know this is needed)

2. **Meaningful treatment of rare entity types**
   - Split entity types into tiers based on frequency (from your EDA)
   - Drop the necessary rare entity types if needed (e.g., those with <50 instances) or keep them for context and accept lower performance
   - Split data: train/validation/test (if not already split, use stratified split) (total 400 samples, train=340, val=30, test=30)

3. **Create data collator and tokenization function**
   - Handle subword tokenization alignment with BIO tags
   - Deal with special tokens ([CLS], [SEP])
   - Pad sequences appropriately
   - Create DataLoader for batching

4. **Define evaluation utilities**
   - seqeval metric for NER (handles BIO tagging correctly)
   - Custom metric functions for per-entity-type performance
   - Confidence extraction utilities

**Deliverable:** Clean, tokenized dataset ready for training

---

### **Task 1.2: Fine-tune PubMedBERT (Baseline)**

**Subtasks:**
1. **Model initialization**
   - Load `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
   - Initialize `AutoModelForTokenClassification` with correct `num_labels`
   - Set up tokenizer with proper special token handling

2. **Training configuration**
   - Learning rate: 2e-5 (or 5e-5, test both)
   - Batch size: 16 or 32 (depending on Kaggle GPU)
   - Epochs: 3-5
   - Warmup steps: 500
   - Weight decay: 0.01
   - Gradient accumulation if needed
   - Report to WandB

3. **Training loop with monitoring**
   - Track training loss
   - Validation loss and F1 after each epoch
   - Early stopping based on validation F1
   - Save best checkpoint to WandB

4. **Basic evaluation**
   - Overall Precision, Recall, F1 (both micro and macro) using seqeval
   - Per-entity-type metrics
   - Report results to WandB as summary metrics
   - Save model and tokenizer artifacts to WandB
   - Generate predictions on test set

**Deliverable:** Trained PubMedBERT model with baseline metrics

---

### **Task 1.3: Fine-tune Additional Models**

**Subtasks:**
1. **BioBERT fine-tuning**
   - Load `dmis-lab/biobert-v1.1`
   - Same training configuration as PubMedBERT
   - Train and evaluate

2. **Alternative model exploration** (pick 1-2)
   - `bioformers/bioformer-16L` (new efficient small model)
   - `emilyalsentzer/Bio_ClinicalBERT` (clinical domain)

3. **Model comparison matrix**
   - Create comparison table: Model | F1 | Precision | Recall | Training Time | Inference Speed
   - Identify best overall model
   - Identify model best for specific entity types

**Deliverable:** 2-3 trained models with comparative analysis

---

### **Task 1.4: Per-Entity-Type Performance Analysis**

**Subtasks:**
1. **Extract predictions by entity type**
   - Group predictions by the 41 entity types in MACCROBAT
   - Calculate P/R/F1 for each type

2. **Tier-based analysis** (from your EDA)
   - **Frequent entities (>1000):** Age, Sex, Date, Diagnostic_procedure, Sign_symptom, etc.
   - **Medium entities (100-1000):** Family_history, Coreference, Distance, etc.
   - **Rare entities (<100):** Weight, Height, Color, Texture, Mass, etc.
   - Compare performance across tiers

3. **Identify systematic weaknesses**
   - Which entity types have lowest F1?
   - Are rare entities harder to predict? (Expected: yes)
   - Which types get confused with each other?

4. **Visualization**
   - Bar chart: F1 score by entity type (sorted)
   - Scatter plot: Entity frequency vs F1 score
   - Heatmap: Confusion matrix for top 15 entity types

**Deliverable:** Detailed breakdown showing model strengths/weaknesses by entity type

---

### **Task 1.5: Error Analysis Framework**

**Subtasks:**
1. **Boundary error detection**
   - Partial matches: "myocardial infarction" predicted as just "myocardial"
   - Over-prediction: "myocardial" predicted as "myocardial infarction"
   - Calculate: exact match rate vs partial match rate

2. **Entity type confusion analysis**
   - Build confusion matrix: True label vs Predicted label
   - Identify common confusions (e.g., Disease_disorder ↔ Sign_symptom)
   - Quantify: What % of errors are type confusion vs boundary errors?

3. **Multi-token entity analysis**
   - From EDA: Some entities are 5+ tokens long
   - Measure: F1 on single-token vs multi-token entities
   - Hypothesis: Longer entities → more boundary errors

4. **Collect error examples**
   - Sample 20-30 mispredictions
   - Categorize: boundary error, type confusion, missed entity, false positive
   - Document patterns (e.g., "abbreviations often missed")

**Deliverable:** Error analysis report with categories and examples

---

### **Task 1.6: Confidence Score Analysis**

**Subtasks:**
1. **Extract prediction confidence scores**
   - Get softmax probabilities from model outputs
   - For each predicted token, store max probability as confidence
   - For span-level confidence, use minimum token confidence in span

2. **Confidence distribution analysis**
   - Plot histogram: confidence scores for correct vs incorrect predictions
   - Calculate: Average confidence for TP, FP, FN
   - Expected: Correct predictions have higher confidence

3. **Threshold analysis for production**
   - Test thresholds: [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
   - For each threshold, calculate:
     - **Precision:** Of predictions above threshold, what % are correct?
     - **Coverage:** What % of entities pass the threshold?
     - **Recall:** What % of true entities are captured at this threshold?

4. **Precision-coverage tradeoff visualization**
   - Line plot: X-axis = threshold, Y-axis = precision/coverage
   - Identify "sweet spot" for production (e.g., 0.85 threshold → 92% precision, 75% coverage)

**Deliverable:** Confidence threshold recommendations for production deployment

---

### **Task 1.7: Inference Speed & Scalability Testing**

**Subtasks:**
1. **Measure inference latency**
   - Time per document (batch size = 1)
   - Time per batch (batch size = 8, 16, 32)
   - Compare across models (PubMedBERT vs BioBERT)

2. **Memory profiling**
   - Peak GPU memory during training
   - Peak GPU memory during inference
   - Can we run on CPU if needed? (Latency increase?)

3. **Throughput estimation**
   - Documents per second on Kaggle GPU
   - Estimate: Time to process 10,000 clinical notes

4. **Scalability considerations**
   - Batch processing vs streaming
   - Model quantization potential (FP16, INT8)

**Deliverable:** Performance benchmarks for production planning

---

### **Task 1.8: Production-Readiness Metrics**

**Subtasks:**
1. **Create clinical coding simulation**
   - For each test document, simulate "auto-code vs manual review" decision
   - Rule: Auto-code if confidence > threshold, else flag for review
   - Calculate automation rate at different thresholds

2. **High-value entity analysis**
   - From your EDA: Disease_disorder, Therapeutic_procedure, Medication are critical for billing
   - Measure: F1 specifically on these high-value entity types
   - Calculate: Precision on high-value entities at 0.85 confidence

3. **False positive impact analysis**
   - Medical coding context: False positives can cause billing errors
   - Measure: FP rate by entity type
   - Identify: Which entity types have highest FP risk?

4. **Create "production readiness report"**
   - Overall accuracy at confidence thresholds
   - Automation rate vs manual review rate
   - High-risk entity types requiring human validation
   - Recommended deployment strategy

**Deliverable:** Production readiness assessment document

---

### **Task 1.9: Model Selection & Documentation**

**Subtasks:**
1. **Select best model for Phase 2**
   - Criteria: F1 score, inference speed, rare entity performance
   - Document tradeoffs: "PubMedBERT has +2% F1 but 1.5x slower than BioBERT"

2. **Create Phase 1 summary notebook**
   - Training results comparison table
   - Key visualizations (confusion matrix, confidence plot, threshold analysis)
   - Error analysis examples
   - Production metrics summary

3. **Save model artifacts**
   - Best checkpoint
   - Tokenizer
   - tag2id/id2tag mappings
   - Evaluation predictions (for Phase 3 error analysis)

4. **Write Phase 1 insights document**
   - Key findings: "Rare entities have 15% lower F1 than frequent entities"
   - Surprising results: "Medication entities confused with Disease 8% of the time"
   - Recommendations: "Use 0.85 confidence threshold for 90% precision, 73% coverage"

**Deliverable:** Best model selected, documented, and ready for Phase 2 + Phase 1 report

---

## Expected Outputs After Phase 1:

✅ **2-3 fine-tuned NER models** (PubMedBERT, BioBERT, +1 optional)  
✅ **Comprehensive evaluation metrics** (overall + per-entity-type + per-tier)  
✅ **Error analysis report** with categorized failures  
✅ **Confidence threshold recommendations** (precision-coverage tradeoff)  
✅ **Production readiness assessment** (automation rates, high-risk entities)  
✅ **Model selection rationale** for Phase 2  
✅ **Phase 1 summary notebook** with key insights  

---

## Suggested Order of Execution:

0. **Task 1.0** → Config + WandB setup
1. **Task 1.1** → Data prep (1 run)
2. **Task 1.2** → PubMedBERT baseline (first model)
3. **Task 1.4, 1.5, 1.6** → Deep analysis on PubMedBERT (understand one model well first)
4. **Task 1.3** → Train additional models
5. **Task 1.7** → Speed benchmarks
6. **Task 1.8** → Production metrics
7. **Task 1.9** → Synthesis and documentation