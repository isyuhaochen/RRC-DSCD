# **Think Wider, Detect Sharper**

## *Reinforced Reference Coverage for Document-Level Self-Contradiction Detection*

This repository contains the official implementation of the paper:
**Think Wider, Detect Sharper: Reinforced Reference Coverage for Document-Level Self-Contradiction Detection**

---

## 📦 Install Environment

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. 🏗️ Data Construction

Training data is constructed with **StorySumm**, **REPLIQA**, and **CoT Distillation** using **DeepSeek R1**.

```bash
bash script/data_constructor.sh
```

---

### 2. 🧠 Train the Model (SFT + RL)

**Step 1: Supervised Fine-Tuning (SFT)**

```bash
bash script/sft.sh
```

**Step 2: Reinforcement Learning Fine-Tuning (RL)**

```bash
bash script/rl.sh
```

---

### 3. 📊 Evaluate the Model

Evaluate (Base, CoT, SFT, RL) on the **ContraDoc** dataset.

```bash
bash script/test.sh
```


