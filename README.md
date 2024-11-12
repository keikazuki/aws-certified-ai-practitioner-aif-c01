# AWS Certified AI Practitioner (AIF-C01) Exam Questions - From Internet

---

**1. Question:**  
A company has built a chatbot that can respond to natural language questions with images. The company wants to ensure that the chatbot does not return inappropriate or unwanted images.  
Which solution will meet these requirements?

**Options:**  
- **A. Implement moderation APIs.**  
- **B. Retrain the model with a general public dataset.**  
- **C. Perform model validation.**  
- **D. Automate user feedback integration.**

**Correct Answer:** **A. Implement moderation APIs.**

**Explanation:**  
To prevent inappropriate or unwanted images from being returned, **moderation APIs** are commonly used. These APIs can analyze and filter content based on defined criteria for appropriateness, identifying any explicit, violent, or otherwise undesired content. This provides a proactive, automated approach to content moderation, making it a practical solution for ensuring content safety in real-time interactions.

**Explanation of Other Options:**
- **B. Retrain the model with a general public dataset**: Retraining on a general dataset doesn’t specifically address moderation. While it may help improve general responses, it won’t reliably prevent inappropriate content.
- **C. Perform model validation**: Model validation is necessary to ensure general performance and accuracy but does not inherently include image moderation.
- **D. Automate user feedback integration**: Automating feedback may help improve the system gradually but is a slower process and does not prevent unwanted images in real-time.

---

**2. Question:**  
An accounting firm wants to implement a large language model (LLM) to automate document processing. The firm must proceed responsibly to avoid potential harms.  
What should the firm do when developing and deploying the LLM? (Select TWO.)

**Options:**  
- **A. Include fairness metrics for model evaluation.**  
- **B. Adjust the temperature parameter of the model.**  
- **C. Modify the training data to mitigate bias.**  
- **D. Avoid overfitting on the training data.**  
- **E. Apply prompt engineering techniques.**

**Correct Answers:** **A. Include fairness metrics for model evaluation** and **C. Modify the training data to mitigate bias.**

**Explanation:**  
To responsibly develop and deploy an LLM for document processing, it's essential to address ethical considerations like fairness and bias:

- **A. Include fairness metrics for model evaluation**: Incorporating fairness metrics ensures that the model is evaluated for equitable performance across different demographic or user groups. This helps in identifying and addressing biases that could lead to unfair outcomes.

- **C. Modify the training data to mitigate bias**: Bias in the training data can lead to biased outputs, which could be problematic in sensitive applications like accounting. Adjusting the training data to reduce or eliminate bias is a key step in building a responsible and fair LLM.

**Explanation of Other Options:**
- **B. Adjust the temperature parameter of the model**: This parameter controls randomness in generation but does not directly impact fairness, bias, or responsible deployment.
- **D. Avoid overfitting on the training data**: Avoiding overfitting is essential for model performance but doesn’t specifically address ethical concerns like fairness or bias.
- **E. Apply prompt engineering techniques**: Prompt engineering can improve model responses but does not address fairness or bias, which are critical for responsible deployment.

---

From: https://www.p2pcerts.com/amazon/aif-c01-dumps.html

### **Question # 1**
A company has thousands of customer support interactions per day and wants to analyze these interactions to identify frequently asked questions and develop insights. Which AWS service can the company use to meet this requirement?

**Options:**  
- **A. Amazon Lex**  
- **B. Amazon Comprehend**  
- **C. Amazon Transcribe**  
- **D. Amazon Translate**  

**Correct Answer:** **B. Amazon Comprehend**

**Explanation:**  
**Amazon Comprehend** is a natural language processing (NLP) service that can analyze text to extract insights such as frequently asked questions, customer sentiment, and key topics. This makes it ideal for analyzing large volumes of customer interactions.

---

### **Question # 2**
A company has a database of petabytes of unstructured data from internal sources. The company wants to transform this data into a structured format so that its data scientists can perform machine learning (ML) tasks. Which service will meet these requirements?

**Options:**  
- **A. Amazon Lex**  
- **B. Amazon Rekognition**  
- **C. Amazon Kinesis Data Streams**  
- **D. AWS Glue**  

**Correct Answer:** **D. AWS Glue**

**Explanation:**  
**AWS Glue** is an ETL (Extract, Transform, Load) service designed to transform and catalog large amounts of unstructured data into a structured format. This is ideal for data scientists who need structured data for ML tasks.

---

### **Question # 3**
Which AWS service or feature can help an AI development team quickly deploy and consume a foundation model (FM) within the team's VPC?

**Options:**  
- **A. Amazon Personalize**  
- **B. Amazon SageMaker JumpStart**  
- **C. PartyRock, an Amazon Bedrock Playground**  
- **D. Amazon SageMaker endpoints**  

**Correct Answer:** **B. Amazon SageMaker JumpStart**

**Explanation:**  
**Amazon SageMaker JumpStart** offers pre-trained models, including foundation models, which can be easily deployed and used within a VPC. This service is specifically designed to help teams quickly access, deploy, and integrate ML models.

---

### **Question # 4**
A company wants to use a large language model (LLM) on Amazon Bedrock for sentiment analysis. The company wants to classify the sentiment of text passages as positive or negative. Which prompt engineering strategy meets these requirements?

**Options:**  
- **A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.**  
- **B. Provide a detailed explanation of sentiment analysis and how LLMs work in the prompt.**  
- **C. Provide the new text passage to be classified without any additional context or examples.**  
- **D. Provide the new text passage with a few examples of unrelated tasks, such as text summarization or question answering.**  

**Correct Answer:** **A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.**

**Explanation:**  
Including labeled examples in the prompt gives the LLM context for how to classify sentiment, improving its accuracy in recognizing positive or negative sentiment in new text passages.

---

### **Question # 5**
A company has installed a security camera. The company uses an ML model to evaluate the security camera footage for potential thefts. The company has discovered that the model disproportionately flags people who are members of a specific ethnic group. Which type of bias is affecting the model output?

**Options:**  
- **A. Measurement bias**  
- **B. Sampling bias**  
- **C. Observer bias**  
- **D. Confirmation bias**  

**Correct Answer:** **B. Sampling bias**

**Explanation:**  
**Sampling bias** occurs when the training data is not representative of the population, leading to skewed model results. In this case, if the model disproportionately flags a specific ethnic group, it is likely due to a bias in the sample data used to train the model.

---

From: https://www.pass4success.com/amazon/exam/aif-c01

### **Question # 1**
A company wants to develop a large language model (LLM) application by using Amazon Bedrock and customer data that is uploaded to Amazon S3. The company's security policy states that each team can access data for only the team's own customers.

**Options:**  
- **A. Create an Amazon Bedrock custom service role for each team that has access to only the team's customer data.**  
- **B. Create a custom service role that has Amazon S3 access. Ask teams to specify the customer name on each Amazon Bedrock request.**  
- **C. Redact personal data in Amazon S3. Update the S3 bucket policy to allow team access to customer data.**  
- **D. Create one Amazon Bedrock role that has full Amazon S3 access. Create IAM roles for each team that have access to only each team's customer folders.**  

**Correct Answer:** **A. Create an Amazon Bedrock custom service role for each team that has access to only the team's customer data.**

**Explanation:**  
Creating a custom service role for each team with specific access to only their customer data aligns with the company’s security policy. This ensures that data access is restricted according to team ownership, maintaining data security and compliance.

---

### **Question # 2**
A company wants to develop a large language model (LLM) application by using Amazon Bedrock and customer data that is uploaded to Amazon S3. The company's security policy states that each team can access data for only the team's own customers.

**Options:**  
- **A. Create an Amazon Bedrock custom service role for each team that has access to only the team's customer data.**  
- **B. Create a custom service role that has Amazon S3 access. Ask teams to specify the customer name on each Amazon Bedrock request.**  
- **C. Redact personal data in Amazon S3. Update the S3 bucket policy to allow team access to customer data.**  
- **D. Create one Amazon Bedrock role that has full Amazon S3 access. Create IAM roles for each team that have access to only each team's customer folders.**  

**Correct Answer:** **A. Create an Amazon Bedrock custom service role for each team that has access to only the team's customer data.**

**Explanation:**  
Using Amazon Bedrock custom service roles that limit each team’s access to their specific customer data aligns with security policies and ensures that data segregation and compliance are maintained.

---

### **Question # 3**
A company has built an image classification model to predict plant diseases from photos of plant leaves. The company wants to evaluate how many images the model classified correctly.

**Options:**  
- **A. R-squared score**  
- **B. Accuracy**  
- **C. Root mean squared error (RMSE)**  
- **D. Learning rate**  

**Correct Answer:** **B. Accuracy**

**Explanation:**  
**Accuracy** is the appropriate metric for evaluating how many images the model classified correctly out of the total number of images. It provides a straightforward measure of the model's overall correctness in a classification task.

---

### **Question # 4**
A company wants to make a chatbot to help customers. The chatbot will help solve technical problems without human intervention. The company chose a foundation model (FM) for the chatbot. The chatbot needs to produce responses that adhere to company tone.

**Options:**  
- **A. Set a low limit on the number of tokens the FM can produce.**  
- **B. Use batch inferencing to process detailed responses.**  
- **C. Experiment and refine the prompt until the FM produces the desired responses.**  
- **D. Define a higher number for the temperature parameter.**  

**Correct Answer:** **C. Experiment and refine the prompt until the FM produces the desired responses.**

**Explanation:**  
To produce responses that match the company tone, prompt engineering is essential. By experimenting and refining the prompt, the company can guide the FM to generate responses that align with their specific requirements for tone and style.

---

### **Question # 5**
A company has installed a security camera. The company uses an ML model to evaluate the security camera footage for potential thefts. The company has discovered that the model disproportionately flags people who are members of a specific ethnic group.

**Options:**  
- **A. Measurement bias**  
- **B. Sampling bias**  
- **C. Observer bias**  
- **D. Confirmation bias**  

**Correct Answer:** **B. Sampling bias**

**Explanation:**  
**Sampling bias** occurs when the training data used is not representative of the broader population, which can lead to a model that unfairly targets specific groups. In this case, if the model disproportionately flags a specific ethnic group, the likely cause is a bias in the training sample data.

---

From: https://vivek-aws.medium.com/aws-certified-ai-practitioner-2c4f8b01baa7

### Domain 1: Fundamentals of AI and ML.

**Question #1**  
What is the primary difference between AI and ML?

**Options:**  
- A) AI is a subset of ML  
- B) ML is a subset of AI  
- C) They are completely unrelated fields  
- D) AI and ML are the same thing  

**Correct Answer:** **B**  
**Explanation:** ML is a subset of AI. Understanding the distinctions between AI, ML, and deep learning is fundamental to the field (Task Statement 1.1).

---

**Question #2**  
Which of the following is NOT a type of machine learning?

**Options:**  
- A) Supervised learning  
- B) Unsupervised learning  
- C) Reinforcement learning  
- D) Diagnostic learning  

**Correct Answer:** **D**  
**Explanation:** Diagnostic learning is not a recognized category of machine learning. Typical types include supervised, unsupervised, and reinforcement learning (Task Statement 1.1).

---

**Question #3**  
What type of data is most suitable for training a computer vision model?

**Options:**  
- A) Tabular data  
- B) Time-series data  
- C) Image data  
- D) Text data  

**Correct Answer:** **C**  
**Explanation:** Computer vision models are designed to work with image data (Task Statement 1.1).

---

**Question #4**  
Which AWS service is best suited for natural language processing tasks?

**Options:**  
- A) Amazon SageMaker  
- B) Amazon Comprehend  
- C) Amazon Polly  
- D) Amazon Transcribe  

**Correct Answer:** **B**  
**Explanation:** Amazon Comprehend is specifically for NLP tasks, supporting analysis of language-based data (Task Statement 1.2).

---

**Question #5**  
What is the primary purpose of exploratory data analysis (EDA) in the ML development lifecycle?

**Options:**  
- A) To train the model  
- B) To deploy the model  
- C) To understand the characteristics of the data  
- D) To monitor the model in production  

**Correct Answer:** **C**  
**Explanation:** EDA helps in understanding the data’s characteristics, a crucial step before model training (Task Statement 1.3).

---

**Question #6**  
Which of the following is NOT a typical stage in an ML pipeline?

**Options:**  
- A) Data collection  
- B) Feature engineering  
- C) Model training  
- D) Customer acquisition  

**Correct Answer:** **D**  
**Explanation:** Customer acquisition is not part of the ML pipeline stages (Task Statement 1.3).

---

**Question #7**  
What does AUC stand for in the context of model performance metrics?

**Options:**  
- A) Average User Cost  
- B) Area Under the Curve  
- C) Automated Universal Calculation  
- D) Augmented Use Case  

**Correct Answer:** **B**  
**Explanation:** AUC, or Area Under the Curve, is a model performance metric related to the ROC curve (Task Statement 1.3).

---

**Question #8**  
Which type of learning is most appropriate when you have a large dataset of labeled examples?

**Options:**  
- A) Unsupervised learning  
- B) Reinforcement learning  
- C) Supervised learning  
- D) Semi-supervised learning  

**Correct Answer:** **C**  
**Explanation:** Supervised learning is ideal for labeled data, as it trains models using input-output pairs (Task Statement 1.1).

---

**Question #9**  
What is the main advantage of using pre-trained models?

**Options:**  
- A) They always perform better than custom models  
- B) They require less computational resources to train  
- C) They are always more accurate  
- D) They can be used immediately without any training data  

**Correct Answer:** **D**  
**Explanation:** Pre-trained models are ready for immediate use without additional training data, making them practical and time-saving (Task Statement 1.3).

---

**Question #10**  
Which AWS service is best suited for automating the process of identifying the best hyperparameters for a model?

**Options:**  
- A) Amazon SageMaker Autopilot  
- B) Amazon Comprehend  
- C) Amazon Polly  
- D) Amazon Transcribe  

**Correct Answer:** **A**  
**Explanation:** Amazon SageMaker Autopilot automates hyperparameter tuning to optimize model performance (Task Statement 1.2 and 1.3).

---

**Question #11**  
What does MLOps stand for?

**Options:**  
- A) Machine Learning Operations  
- B) Multiple Learning Optimizations  
- C) Model Learning Objectives  
- D) Managed Learning Outputs  

**Correct Answer:** **A**  
**Explanation:** MLOps stands for Machine Learning Operations, which involves best practices for operationalizing machine learning models (Task Statement 1.3).

---

**Question #12**  
Which of the following is NOT a typical business metric for evaluating ML models?

**Options:**  
- A) Cost per user  
- B) Development costs  
- C) Customer feedback  
- D) F1 score  

**Correct Answer:** **D**  
**Explanation:** F1 score is a model performance metric, not a business metric. Business metrics generally reflect cost, customer satisfaction, or feedback, rather than technical model performance (Task Statement 1.3).

---

**Question #13**  
What type of learning is most appropriate when you want an agent to learn from its interactions with an environment?

**Options:**  
- A) Supervised learning  
- B) Unsupervised learning  
- C) Reinforcement learning  
- D) Transfer learning  

**Correct Answer:** **C**  
**Explanation:** Reinforcement learning allows an agent to learn by interacting with its environment, often using a reward-based system (Task Statement 1.1).

---

**Question #14**  
Which AWS service is best suited for converting text to speech?

**Options:**  
- A) Amazon Comprehend  
- B) Amazon Translate  
- C) Amazon Transcribe  
- D) Amazon Polly  

**Correct Answer:** **D**  
**Explanation:** Amazon Polly is designed for text-to-speech conversion, turning written text into natural-sounding speech (Task Statement 1.2).

---

**Question #15**  
What is the primary purpose of feature engineering in the ML development lifecycle?

**Options:**  
- A) To collect more data  
- B) To create new features or transform existing ones to improve model performance  
- C) To evaluate the model’s performance  
- D) To deploy the model to production  

**Correct Answer:** **B**  
**Explanation:** Feature engineering involves creating or modifying features to improve the model's ability to make accurate predictions (Task Statement 1.3).

---

**Question #16**  
Which of the following is an example of unsupervised learning?

**Options:**  
- A) Spam detection  
- B) Image classification  
- C) Clustering customer segments  
- D) Predicting house prices  

**Correct Answer:** **C**  
**Explanation:** Clustering, a common unsupervised learning method, groups data based on similarities without labeled outputs (Task Statement 1.1).

---

**Question #17**  
What is the main difference between batch inferencing and real-time inferencing?

**Options:**  
- A) Batch inferencing is always more accurate  
- B) Real-time inferencing can only be done on small datasets  
- C) Batch inferencing processes multiple inputs at once, while real-time inferencing processes individual inputs as they arrive  
- D) Real-time inferencing is always faster than batch inferencing  

**Correct Answer:** **C**  
**Explanation:** Batch inferencing processes multiple data inputs simultaneously, while real-time inferencing handles individual inputs instantly (Task Statement 1.1).

---

**Question #18**  
Which AWS service is best suited for managing the entire machine learning lifecycle?

**Options:**  
- A) Amazon Comprehend  
- B) Amazon SageMaker  
- C) Amazon Polly  
- D) Amazon Translate  

**Correct Answer:** **B**  
**Explanation:** Amazon SageMaker provides tools for the complete ML lifecycle, from data preparation to model deployment and monitoring (Task Statement 1.3).

---

**Question #19**  
What is the primary purpose of model monitoring in production?

**Options:**  
- A) To train new models  
- B) To collect more data  
- C) To detect issues like model drift or data drift  
- D) To perform feature engineering  

**Correct Answer:** **C**  
**Explanation:** Model monitoring helps in identifying issues such as model drift or data drift, ensuring continued model accuracy and reliability (Task Statement 1.3).

---

**Question #20**  
Which of the following is NOT a typical use case for AI/ML?

**Options:**  
- A) Fraud detection  
- B) Recommendation systems  
- C) Manual data entry  
- D) Speech recognition  

**Correct Answer:** **C**  
**Explanation:** Manual data entry is typically a human task, not suited for AI/ML automation, unlike fraud detection, recommendation systems, and speech recognition (Task Statement 1.1).

---

### Domain 2: Fundamentals of Generative AI.

**Question #1**  
What is a token in the context of generative AI?

**Options:**  
- A) A security feature  
- B) A unit of text processed by the model  
- C) A type of neural network  
- D) A model evaluation metric  

**Correct Answer:** **B**  
**Explanation:** In generative AI, a token is a unit of text that the model processes, such as a word or part of a word (Task Statement 2.1).

---

**Question #2**  
Which of the following is NOT a typical use case for generative AI models?

**Options:**  
- A) Image generation  
- B) Summarization  
- C) Data encryption  
- D) Code generation  

**Correct Answer:** **C**  
**Explanation:** Data encryption is not a common application for generative AI models. Generative AI is more commonly used for image generation, summarization, and code generation (Task Statement 2.1).

---

**Question #3**  
What is the primary advantage of generative AI’s adaptability?

**Options:**  
- A) It can only work with structured data  
- B) It can handle a wide range of tasks and domains  
- C) It always produces perfect results  
- D) It eliminates the need for human oversight  

**Correct Answer:** **B**  
**Explanation:** Generative AI’s adaptability allows it to perform well across various tasks and domains (Task Statement 2.2).

---

**Question #4**  
What is a hallucination in the context of generative AI?

**Options:**  
- A) A visual output produced by the model  
- B) A type of model architecture  
- C) An incorrect or fabricated output presented as fact  
- D) A method of model training  

**Correct Answer:** **C**  
**Explanation:** A hallucination in generative AI is when the model produces incorrect or fabricated information as if it were factual (Task Statement 2.2).

---

**Question #5**  
Which AWS service is designed specifically for developing generative AI applications?

**Options:**  
- A) Amazon EC2  
- B) Amazon S3  
- C) Amazon Bedrock  
- D) Amazon RDS  

**Correct Answer:** **C**  
**Explanation:** Amazon Bedrock is an AWS service tailored for generative AI development (Task Statement 2.3).

---

**Question #6**  
What is a foundation model in generative AI?

**Options:**  
- A) A model that can only generate text  
- B) A large, pre-trained model that can be adapted for various tasks  
- C) A model specifically designed for image generation  
- D) A model that requires no training data  

**Correct Answer:** **B**  
**Explanation:** Foundation models are large, pre-trained models adaptable to various tasks (Task Statement 2.1).

---

**Question #7**  
Which of the following is NOT a stage in the foundation model lifecycle?

**Options:**  
- A) Data selection  
- B) Pre-training  
- C) Deployment  
- D) Marketing  

**Correct Answer:** **D**  
**Explanation:** Marketing is not part of the foundation model lifecycle (Task Statement 2.1).

---

**Question #8**  
What is the primary advantage of using AWS generative AI services for building applications?

**Options:**  
- A) They are always free  
- B) They provide a lower barrier to entry  
- C) They guarantee 100% accuracy  
- D) They eliminate the need for any coding  

**Correct Answer:** **B**  
**Explanation:** AWS generative AI services offer a lower barrier to entry, making it easier to develop AI applications (Task Statement 2.3).

---

**Question #9**  
What is prompt engineering in the context of generative AI?

**Options:**  
- A) A method of hardware optimization  
- B) A technique for designing the physical structure of AI models  
- C) The process of crafting effective input prompts to guide model outputs  
- D) A way to reduce energy consumption in AI systems  

**Correct Answer:** **C**  
**Explanation:** Prompt engineering involves creating effective prompts to influence the model’s output (Task Statement 2.1).

---

**Question #10**  
Which of the following is a potential disadvantage of generative AI solutions?

**Options:**  
- A) Adaptability  
- B) Responsiveness  
- C) Inaccuracy  
- D) Simplicity  

**Correct Answer:** **C**  
**Explanation:** Inaccuracy is a known disadvantage in generative AI, as it can sometimes produce incorrect outputs (Task Statement 2.2).

---

**Question #11**  
What is a multi-modal model in generative AI?

**Options:**  
- A) A model that can only process text data  
- B) A model that can work with multiple types of data (e.g., text, images, audio)  
- C) A model that requires multiple GPUs to run  
- D) A model that can only generate images  

**Correct Answer:** **B**  
**Explanation:** A multi-modal model processes multiple data types like text, images, and audio (Task Statement 2.1).

---

**Question #12**  
Which AWS service provides a playground for experimenting with generative AI models?

**Options:**  
- A) Amazon SageMaker  
- B) Amazon Comprehend  
- C) PartyRock  
- D) Amazon Polly  

**Correct Answer:** **C**  
**Explanation:** PartyRock, an Amazon Bedrock Playground, is designed for testing and experimenting with generative AI models (Task Statement 2.3).

---

**Question #13**  
What is a key consideration when selecting an appropriate generative AI model for a business problem?

**Options:**  
- A) The model’s popularity on social media  
- B) The model’s performance requirements  
- C) The model’s development date  
- D) The model’s country of origin  

**Correct Answer:** **B**  
**Explanation:** Performance requirements are crucial in selecting an AI model to ensure it meets business needs (Task Statement 2.2).

---

**Question #14**  
Which of the following is NOT a typical business metric for evaluating generative AI applications?

**Options:**  
- A) Conversion rate  
- B) Average revenue per user  
- C) Customer lifetime value  
- D) Model parameter count  

**Correct Answer:** **D**  
**Explanation:** Model parameter count is a technical metric, not a business metric (Task Statement 2.2).

---

**Question #15**  
What is a key benefit of AWS infrastructure for generative AI applications?

**Options:**  
- A) It eliminates the need for any security measures  
- B) It provides unlimited free computing resources  
- C) It ensures compliance with relevant regulations  
- D) It guarantees that AI models will never make mistakes  

**Correct Answer:** **C**  
**Explanation:** AWS provides infrastructure that supports compliance with industry regulations, which is crucial for many applications (Task Statement 2.3).

---

**Question #16**  
What is chunking in the context of generative AI?

**Options:**  
- A) A method of data compression  
- B) A technique for breaking down large inputs into smaller, manageable pieces  
- C) A type of model architecture  
- D) A way to increase model accuracy  

**Correct Answer:** **B**  
**Explanation:** Chunking involves splitting large inputs into smaller pieces, helping the model process the data more efficiently (Task Statement 2.1).

---

**Question #17**  
Which of the following is a key advantage of generative AI’s simplicity?

**Options:**  
- A) It always produces perfect results  
- B) It requires no human input  
- C) It can be easier to implement and use compared to traditional methods  
- D) It eliminates the need for data preprocessing  

**Correct Answer:** **C**  
**Explanation:** Generative AI’s simplicity can make it easier to implement and use than traditional methods (Task Statement 2.2).

---

**Question #18**  
What is a diffusion model in generative AI?

**Options:**  
- A) A model that only works with textual data  
- B) A type of generative model often used for image generation  
- C) A model that requires no training data  
- D) A model specifically designed for natural language processing  

**Correct Answer:** **B**  
**Explanation:** Diffusion models are used primarily for image generation in generative AI (Task Statement 2.1).

---

**Question #19**  
Which AWS service is designed to provide conversational AI capabilities?

**Options:**  
- A) Amazon Bedrock  
- B) Amazon SageMaker  
- C) Amazon Q  
- D) Amazon S3  

**Correct Answer:** **C**  
**Explanation:** Amazon Q provides conversational AI capabilities as part of AWS services (Task Statement 2.3).

---

**Question #20**  
What is a key consideration in the cost tradeoffs of AWS generative AI services?

**Options:**  
- A) The color scheme of the user interface  
- B) The number of employees in the company  
- C) Token-based pricing  
- D) The physical location of the data center  

**Correct Answer:** **C**  
**Explanation:** Token-based pricing is a primary cost consideration when using AWS generative AI services (Task Statement 2.3).

---

