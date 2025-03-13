# Federated Learning for Personalized E-Commerce Recommendations: Enhancing Privacy and Efficiency

## 🚀 Problem Statement 🛒
E-commerce platforms rely heavily on **personalized recommendations** to enhance **user experience and drive sales**. However, traditional centralized recommendation models **pose significant privacy risks** due to sensitive user data being collected on central servers.

### **🔹 Why Federated Learning (FL)?**
Federated Learning (FL) enables **decentralized model training** across multiple user devices while ensuring **data privacy**. Instead of sending raw user data to a central server, FL allows devices to **train models locally and share only model updates**.

### **⚠️ Challenges in FL for E-Commerce:**
- **Non-IID Data:** Users have different shopping patterns, making learning inconsistent.
- **Slow Convergence:** Model updates vary across clients, affecting efficiency.
- **High Communication Costs:** Frequent updates can increase network load.
- **Model Poisoning Attacks:** Malicious clients can degrade recommendations.

This project **addresses these challenges** by incorporating **advanced FL techniques** like:

✅ **FedProx** for handling non-IID data  
✅ **FedAvg+ (Adaptive Aggregation)** for optimized updates  
✅ **Compression Methods** to reduce communication overhead  
✅ **Robust Security Measures** to defend against poisoning attacks  

---

## 📊 Dataset: Amazon Product Review Dataset
The **Amazon Product Review Dataset** is used to simulate user interactions with e-commerce products. It includes:
- **Product details**: Name, category, subcategory, price, discounts
- **User feedback**: Ratings, number of reviews
- **Behavioral insights**: Purchase likelihood (synthetically generated)

### **🔹 Data Partitioning for FL**
To replicate real-world **federated learning scenarios**, the dataset is **partitioned across multiple FL clients** based on:
- **User preferences**: Some users prioritize **ratings**, while others focus on **high prices**.
- **Non-IID distribution**: Each FL client has a unique subset of the dataset.

---

## 🛠️ **Tech Stack & Methodology**
### **🔹 Technologies Used**
- **Frameworks**: PyTorch, Flower (`flwr`)
- **Data Processing**: Pandas, NumPy
- **Communication**: Apache Kafka (Future Integration)
- **Deployment**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

### **🔹 Federated Learning Approach**
1. **FL Clients**: Train recommendation models locally on partitioned data.
2. **FL Server**: Aggregates client updates using **FedAvg+/FedProx**.
3. **Security & Optimization**:
   - **Compression** to reduce communication costs.
   - **Poisoning detection** for robustness.
4. **Final Deployment**: Model improvements are sent back to clients.

---

## 👥 Team Members & Responsibilities
Our team collaborates effectively with well-defined roles:

| 👤 Team Member | 🎯 Role | 
|--------------|--------|
| **Sathwik** | 🔹 Team Lead & Model Architect | 
| **Zachary** | 🔹 Data & Optimization Specialist | 
| **Mylie** | 🔹 Privacy & Evaluation | 

---

## 🏗️ **How to Contribute?**
### 🔹 **For Team Members**
1. **Clone the repository**  
   ```bash
   git clone https://github.com/sathwikabbaraju/Federated-Learning-Ecommerce-Recommendation.git
   cd Federated-Learning-Ecommerce-Recommendation
