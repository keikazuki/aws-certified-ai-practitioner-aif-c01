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

### **Question #21**  
What is the primary purpose of embeddings in generative AI?

**Options:**  
- **A) To compress data for storage**  
- **B) To represent data in a high-dimensional space**  
- **C) To encrypt sensitive information**  
- **D) To generate random numbers**  

**Correct Answer:** **B**  
**Explanation:**  
Embeddings are used in generative AI to represent data, such as words or images, in a high-dimensional space. This transformation captures the semantic meaning of the data, allowing models to find relationships or similarities in complex data. 

- **Option A:** Compressing data is not the primary purpose of embeddings, though embeddings can sometimes reduce dimensionality.
- **Option C:** Embeddings do not encrypt data; their role is to map data to a vector space, not for security.
- **Option D:** Embeddings are deterministic representations and are not used for random number generation.

---

### **Question #22**  
Which of the following is NOT a typical use case for generative AI in customer service?

**Options:**  
- **A) Chatbots**  
- **B) Automated email responses**  
- **C) Physical robot assistants**  
- **D) FAQ generation**  

**Correct Answer:** **C**  
**Explanation:**  
Physical robot assistants fall outside the common applications of generative AI in customer service, which typically focuses on tasks like answering questions or generating content.

- **Option A:** Chatbots are widely used for customer interactions.
- **Option B:** Automated email responses can be generated by AI to reply to customer inquiries.
- **Option D:** FAQ generation leverages generative AI for providing relevant answers to frequent questions.

---

### **Question #23**  
What is a key advantage of using AWS generative AI services for building applications in terms of development speed?

**Options:**  
- **A) They automatically write all the code for you**  
- **B) They provide faster time to market**  
- **C) They eliminate the need for testing**  
- **D) They guarantee instant deployment**  

**Correct Answer:** **B**  
**Explanation:**  
AWS generative AI services facilitate a quicker development process, allowing applications to reach the market faster due to managed infrastructure, pre-trained models, and integrated tools.

- **Option A:** While AWS offers helpful tools, it doesn’t automatically generate all the necessary code.
- **Option C:** Testing remains crucial for quality assurance, even with AWS’s streamlined services.
- **Option D:** Instant deployment is not guaranteed, as additional configuration and optimization may be needed.

---

### **Question #24**  
What is nondeterminism in the context of generative AI?

**Options:**  
- **A) A type of model architecture**  
- **B) A method of data preprocessing**  
- **C) The property of producing different outputs for the same input**  
- **D) A technique for improving model accuracy**  

**Correct Answer:** **C**  
**Explanation:**  
Nondeterminism in generative AI refers to the variability in output for identical inputs due to randomness, especially useful for creative or diverse responses.

- **Option A:** Nondeterminism isn’t a model architecture but rather an inherent property of some generative models.
- **Option B:** It is unrelated to data preprocessing, which focuses on preparing input data.
- **Option D:** Nondeterminism can diversify outputs but doesn’t directly improve accuracy.

---

### **Question #25**  
Which AWS service is designed to help developers quickly get started with pre-trained models for generative AI?

**Options:**  
- **A) Amazon EC2**  
- **B) Amazon SageMaker JumpStart**  
- **C) Amazon RDS**  
- **D) Amazon CloudFront**  

**Correct Answer:** **B**  
**Explanation:**  
Amazon SageMaker JumpStart provides access to pre-trained models and templates, allowing developers to start with generative AI applications quickly.

- **Option A:** Amazon EC2 provides general compute services but doesn’t offer pre-trained AI models.
- **Option C:** Amazon RDS is used for database management, not model deployment.
- **Option D:** Amazon CloudFront is a content delivery service and not relevant to AI model development.

---

### Domain 3: Applications of Foundation Models

### **Question #21**  
What is Retrieval Augmented Generation (RAG)?

**Options:**  
- **A) A technique for generating new data**  
- **B) A method of combining retrieved information with model generation**  
- **C) A type of model architecture**  
- **D) A data compression algorithm**  

**Correct Answer:** **B**  
**Explanation:**  
RAG (Retrieval Augmented Generation) is a technique that combines information retrieval with generative models. It retrieves relevant information from a database or document set and uses that information to produce more contextually accurate outputs.

- **Option A:** RAG does not generate new data independently; it combines retrieved information with generated content.
- **Option C:** RAG is a method, not a model architecture.
- **Option D:** It is unrelated to data compression.

---

### **Question #22**  
Which AWS service is suitable for storing embeddings in a vector database?

**Options:**  
- **A) Amazon S3**  
- **B) Amazon RDS**  
- **C) Amazon OpenSearch Service**  
- **D) Amazon EC2**  

**Correct Answer:** **C**  
**Explanation:**  
Amazon OpenSearch Service is ideal for storing and searching embeddings, as it supports vector search, allowing for efficient storage and retrieval of high-dimensional data.

- **Option A:** Amazon S3 is a storage service but lacks vector search capabilities.
- **Option B:** Amazon RDS is primarily for relational databases, not optimized for storing embeddings.
- **Option D:** Amazon EC2 is a compute service, not designed for embedding storage.

---

### **Question #23**  
What is the primary purpose of adjusting the temperature parameter in inference?

**Options:**  
- **A) To control the physical temperature of the server**  
- **B) To adjust the creativity or randomness of the model’s output**  
- **C) To increase the model’s processing speed**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
The temperature parameter controls randomness in model output. A higher temperature results in more varied responses, while a lower temperature produces more predictable outputs.

- **Option A:** It does not control physical temperature.
- **Option C:** Temperature does not impact processing speed.
- **Option D:** It has no direct effect on energy consumption.

---

### **Question #24**  
What is a chain-of-thought prompt?

**Options:**  
- **A) A physical chain used in AI hardware**  
- **B) A prompt that encourages the model to show its reasoning process**  
- **C) A method of linking multiple AI models**  
- **D) A technique for encrypting prompts**  

**Correct Answer:** **B**  
**Explanation:**  
A chain-of-thought prompt guides the model to outline its reasoning steps, improving accuracy in tasks that require logic or calculations.

- **Option A:** This is unrelated to hardware.
- **Option C:** It does not link models but guides the model’s response structure.
- **Option D:** It has no encryption purpose.

---

### **Question #25**  
Which of the following is NOT a typical method for fine-tuning a foundation model?

**Options:**  
- **A) Instruction tuning**  
- **B) Transfer learning**  
- **C) Physical tuning**  
- **D) Continuous pre-training**  

**Correct Answer:** **C**  
**Explanation:**  
Physical tuning is not a recognized method for fine-tuning models. Instruction tuning, transfer learning, and continuous pre-training are common techniques.

- **Option A, B, D:** These are standard methods for model fine-tuning.

---

### **Question #26**  
What is the ROUGE metric used for in evaluating foundation models?

**Options:**  
- **A) Measuring the redness of the model’s output**  
- **B) Evaluating the quality of generated summaries**  
- **C) Calculating the model’s energy efficiency**  
- **D) Determining the model’s processing speed**  

**Correct Answer:** **B**  
**Explanation:**  
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) evaluates the quality of generated summaries by comparing them with reference texts.

- **Option A:** It has nothing to do with color.
- **Option C & D:** ROUGE is not related to efficiency or speed.

---

### **Question #27**  
What is the primary purpose of using Agents for Amazon Bedrock?

**Options:**  
- **A) To hire human agents for AI tasks**  
- **B) To handle multi-step tasks in AI applications**  
- **C) To physically maintain AI hardware**  
- **D) To reduce the cost of AI services**  

**Correct Answer:** **B**  
**Explanation:**  
Agents for Amazon Bedrock manage multi-step processes in AI applications, such as complex workflows or decision-making tasks.

- **Option A:** These are virtual, not human agents.
- **Option C & D:** They are unrelated to hardware maintenance or cost reduction.

---

### **Question #28**  
Which of the following is a key consideration when selecting a pre-trained model?

**Options:**  
- **A) The model’s popularity on social media**  
- **B) The physical size of the server hosting the model**  
- **C) The model’s input/output length capabilities**  
- **D) The color scheme of the model’s documentation**  

**Correct Answer:** **C**  
**Explanation:**  
The input/output length affects how much information the model can process, crucial for determining its suitability for specific tasks.

- **Option A, B, D:** These do not impact the model's effectiveness.

---

### **Question #29**  
What is prompt hijacking in the context of prompt engineering?

**Options:**  
- **A) A method of optimizing prompts**  
- **B) A technique for stealing prompts from competitors**  
- **C) An attack where the model is tricked into ignoring the intended prompt**  
- **D) A way to speed up prompt processing**  

**Correct Answer:** **C**  
**Explanation:**  
Prompt hijacking is a risk where unintended prompts override the original, potentially leading to malicious responses.

- **Option A, B, D:** These do not relate to hijacking or security risks.

---

### **Question #30**  
What is the primary goal of instruction tuning in foundation models?

**Options:**  
- **A) To teach the model to follow specific instructions**  
- **B) To reduce the model’s size**  
- **C) To increase the model’s processing speed**  
- **D) To change the model’s programming language**  

**Correct Answer:** **A**  
**Explanation:**  
Instruction tuning trains the model to follow user-specific commands more accurately.

- **Option B, C, D:** These are unrelated to instruction tuning’s purpose.

---

### Domain 4: Guidelines for Responsible AI

### **Question #1**  
Which of the following is NOT a feature of responsible AI?

**Options:**  
- **A) Fairness**  
- **B) Robustness**  
- **C) Profitability**  
- **D) Inclusivity**  

**Correct Answer:** **C**  
**Explanation:**  
Responsible AI focuses on ethical principles like fairness, robustness, and inclusivity to ensure models are accurate, unbiased, and accessible to all user groups. Profitability is generally a business objective, not a responsible AI feature.

- **Options A, B, and D:** These align with responsible AI's focus on equitable and safe technology.

---

### **Question #2**  
What is the primary purpose of Guardrails for Amazon Bedrock?

**Options:**  
- **A) To physically protect AI hardware**  
- **B) To identify and enforce responsible AI features**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
Guardrails for Amazon Bedrock help ensure responsible AI practices by monitoring models to prevent biases or harmful outputs.

- **Options A, C, and D:** These are unrelated to responsible AI features enforcement.

---

### **Question #3**  
Which of the following is a key consideration in responsible model selection?

**Options:**  
- **A) The model’s popularity**  
- **B) The model’s environmental impact**  
- **C) The model’s country of origin**  
- **D) The model’s color scheme**  

**Correct Answer:** **B**  
**Explanation:**  
Responsible AI includes choosing models that have minimal environmental impact, promoting sustainability.

- **Options A, C, and D:** Popularity, origin, and color are not relevant to responsible AI considerations.

---

### **Question #4**  
What is a potential legal risk of working with generative AI?

**Options:**  
- **A) Physical injury to users**  
- **B) Intellectual property infringement claims**  
- **C) Increased electricity bills**  
- **D) Reduced internet speed**  

**Correct Answer:** **B**  
**Explanation:**  
Using generative AI models can inadvertently produce content that infringes on intellectual property, posing legal risks.

- **Options A, C, and D:** These are not recognized legal risks related to generative AI.

---

### **Question #5**  
Which of the following is NOT a characteristic of datasets important for responsible AI?

**Options:**  
- **A) Inclusivity**  
- **B) Diversity**  
- **C) Size**  
- **D) Balanced representation**  

**Correct Answer:** **C**  
**Explanation:**  
While dataset size can affect model performance, inclusivity, diversity, and balanced representation are essential characteristics to prevent bias and ensure ethical AI.

- **Options A, B, and D:** These align with responsible AI practices for data quality.

---

### **Question #6**  
What is overfitting in the context of AI models?

**Options:**  
- **A) When a model performs too well on the training data but poorly on new data**  
- **B) When a model is too large to fit in memory**  
- **C) When a model generates outputs that are too long**  
- **D) When a model consumes too much energy**  

**Correct Answer:** **A**  
**Explanation:**  
Overfitting occurs when a model is overly tailored to the training data, resulting in poor generalization to new data.

- **Options B, C, and D:** These are not definitions of overfitting.

---

### **Question #7**  
Which AWS service is designed to help detect and monitor bias in machine learning models?

**Options:**  
- **A) Amazon EC2**  
- **B) Amazon S3**  
- **C) Amazon SageMaker Clarify**  
- **D) Amazon RDS**  

**Correct Answer:** **C**  
**Explanation:**  
Amazon SageMaker Clarify provides tools to detect and monitor bias within models, supporting responsible AI practices.

- **Options A, B, and D:** These services are not designed for bias detection.

---

### **Question #8**  
What is the primary difference between transparent and non-transparent AI models?

**Options:**  
- **A) Transparent models are always more accurate**  
- **B) Transparent models allow for understanding of their decision-making process**  
- **C) Transparent models are always smaller in size**  
- **D) Transparent models consume less energy**  

**Correct Answer:** **B**  
**Explanation:**  
Transparent models offer insights into their decision-making process, which is crucial for explainable and accountable AI.

- **Options A, C, and D:** Transparency does not guarantee accuracy, size reduction, or lower energy use.

---

### **Question #9**  
Which tool can be used to document model information for transparency?

**Options:**  
- **A) Amazon SageMaker Model Cards**  
- **B) Amazon EC2**  
- **C) Amazon S3**  
- **D) Amazon RDS**  

**Correct Answer:** **A**  
**Explanation:**  
Amazon SageMaker Model Cards provide documentation about model details, helping ensure transparency in model development and usage.

- **Options B, C, and D:** These services do not support model documentation.

---

### **Question #10**  
What is a potential trade-off between model safety and transparency?

**Options:**  
- **A) Safer models are always less transparent**  
- **B) Transparent models are always less safe**  
- **C) Increased transparency might reveal vulnerabilities**  
- **D) There are no trade-offs between safety and transparency**  

**Correct Answer:** **C**  
**Explanation:**  
Increased transparency can expose model internals, potentially revealing security weaknesses.

- **Options A, B, and D:** Transparency and safety can co-exist; trade-offs depend on specific implementation.

---

### **Question #11**  
What is human-centered design in the context of explainable AI?

**Options:**  
- **A) Designing AI systems that look like humans**  
- **B) Creating AI systems that prioritize human needs and understanding**  
- **C) Using humans instead of AI for all tasks**  
- **D) Designing AI systems that can only be used by humans**  

**Correct Answer:** **B**  
**Explanation:**  
Human-centered design ensures that AI systems are understandable, accessible, and beneficial to humans, prioritizing ease of use and transparency.

- **Options A, C, and D:** These options misinterpret human-centered design’s focus on usability and interpretability.

---

### **Question #12**  
Which of the following is NOT a typical effect of bias in AI systems?

**Options:**  
- **A) Unfair treatment of certain demographic groups**  
- **B) Improved overall accuracy**  
- **C) Potential legal issues**  
- **D) Loss of user trust**  

**Correct Answer:** **B**  
**Explanation:**  
Bias typically reduces model accuracy and fairness, leading to legal issues or loss of trust, rather than improving accuracy.

- **Options A, C, and D:** These are direct consequences of bias in AI systems.

---

### **Question #13**  
What is the primary purpose of subgroup analysis in responsible AI?

**Options:**  
- **A) To divide the development team into subgroups**  
- **B) To analyze the model’s performance across different demographic groups**  
- **C) To reduce the model’s size**  
- **D) To increase the model’s processing speed**  

**Correct Answer:** **B**  
**Explanation:**  
Subgroup analysis checks if the model performs fairly across demographic groups, ensuring equitable treatment.

- **Options A, C, and D:** These do not relate to evaluating demographic fairness in model performance.

---

### **Question #14**  
Which of the following is a key consideration for dataset diversity in responsible AI?

**Options:**  
- **A) Using data from only one source**  
- **B) Ensuring representation of various demographic groups**  
- **C) Using the largest dataset available regardless of content**  
- **D) Using only the most recent data**  

**Correct Answer:** **B**  
**Explanation:**  
Diverse datasets help prevent bias by including representative samples from various demographic groups.

- **Options A, C, and D:** These do not address diversity or responsible data practices.

---

### **Question #15**  
What is veracity in the context of responsible AI?

**Options:**  
- **A) The speed at which the AI system operates**  
- **B) The truthfulness and accuracy of the AI system’s outputs**  
- **C) The size of the AI model**  
- **D) The cost of running the AI system**  

**Correct Answer:** **B**  
**Explanation:**  
Veracity in responsible AI refers to ensuring output accuracy, making it reliable and trustworthy.

- **Options A, C, and D:** These do not relate to the quality or accuracy of AI outputs.

---

### **Question #16**  
Which of the following is NOT a typical method for improving model interpretability?

**Options:**  
- **A) Using simpler models**  
- **B) Providing feature importance rankings**  
- **C) Increasing the model’s size**  
- **D) Generating human-readable explanations**  

**Correct Answer:** **C**  
**Explanation:**  
Simpler models, feature importance rankings, and human-readable explanations enhance interpretability. Increasing model size generally complicates interpretation.

- **Options A, B, and D:** These are widely used techniques to improve interpretability.

---

### **Question #17**  
What is the primary purpose of Amazon Augmented AI (A2I) in responsible AI?

**Options:**  
- **A) To replace human workers with AI**  
- **B) To facilitate human review of AI predictions**  
- **C) To increase the AI model’s size**  
- **D) To reduce energy consumption of AI systems**  

**Correct Answer:** **B**  
**Explanation:**  
Amazon A2I integrates human reviewers into AI processes, ensuring output quality and responsible AI practices.

- **Options A, C, and D:** These are unrelated to A2I’s focus on human oversight in AI predictions.

---

### **Question #18**  
Which of the following is a key consideration when evaluating the fairness of an AI system?

**Options:**  
- **A) The system’s processing speed**  
- **B) The system’s energy consumption**  
- **C) The system’s impact on different demographic groups**  
- **D) The system’s popularity among users**  

**Correct Answer:** **C**  
**Explanation:**  
Evaluating fairness involves analyzing how the model affects different demographic groups to prevent discrimination.

- **Options A, B, and D:** These are not directly related to fairness in responsible AI.

---

### **Question #19**  
What is underfitting in the context of AI models?

**Options:**  
- **A) When a model is too small to fit in memory**  
- **B) When a model performs poorly on both training and new data**  
- **C) When a model generates outputs that are too short**  
- **D) When a model consumes too little energy**  

**Correct Answer:** **B**  
**Explanation:**  
Underfitting occurs when a model is too simple or insufficiently trained, resulting in poor performance on both training and unseen data.

- **Options A, C, and D:** These do not describe underfitting in AI.

---

### **Question #20**  
Which of the following is NOT a typical benefit of using open source models for transparency?

**Options:**  
- **A) Ability to inspect the model’s code**  
- **B) Community-driven improvements**  
- **C) Guaranteed perfect performance**  
- **D) Potential for independent audits**  

**Correct Answer:** **C**  
**Explanation:**  
Open-source models provide transparency but do not guarantee perfect performance. Benefits include code accessibility, community contributions, and external audits.

- **Options A, B, and D:** These are typical transparency benefits of open-source models.

---

### **Question #21**  
What is the primary purpose of analyzing label quality in responsible AI?

**Options:**  
- **A) To improve the visual appearance of labels**  
- **B) To ensure the accuracy and consistency of data labels**  
- **C) To reduce the number of labels used**  
- **D) To increase the model’s processing speed**  

**Correct Answer:** **B**  
**Explanation:**  
Label quality analysis ensures data labels are accurate and consistent, which is essential for reliable model training.

- **Options A, C, and D:** These are not relevant to label quality analysis in responsible AI.

---

### **Question #22**  
Which of the following is a potential consequence of using biased datasets in AI training?

**Options:**  
- **A) Improved model performance for all groups**  
- **B) Unfair or discriminatory outcomes for certain groups**  
- **C) Reduced energy consumption**  
- **D) Faster model training times**  

**Correct Answer:** **B**  
**Explanation:**  
Biased datasets can result in unfair or discriminatory treatment of specific groups, a major concern in responsible AI.

- **Options A, C, and D:** These are not typically consequences of dataset bias.

---

### **Question #23**  
What is the primary goal of responsible practices in model selection?

**Options:**  
- **A) To always choose the largest model available**  
- **B) To select models based solely on performance metrics**  
- **C) To balance performance with ethical considerations and sustainability**  
- **D) To choose the most expensive model**  

**Correct Answer:** **C**  
**Explanation:**  
Responsible model selection considers ethical impacts, sustainability, and performance to make informed decisions.

- **Options A, B, and D:** These approaches do not consider responsible AI practices.

---

### **Question #24**  
Which of the following is NOT a typical characteristic of a curated data source for responsible AI?

**Options:**  
- **A) Verified accuracy**  
- **B) Known provenance**  
- **C) Largest possible size**  
- **D) Ethical collection methods**  

**Correct Answer:** **C**  
**Explanation:**  
While large datasets are helpful, responsible AI values accuracy, provenance, and ethical collection over sheer size.

- **Options A, B, and D:** These are key qualities of responsibly curated data sources.

---

### **Question #25**  
What is the primary purpose of human audits in responsible AI systems?

**Options:**  
- **A) To replace AI systems with human workers**  
- **B) To verify and validate AI system outputs and processes**  
- **C) To increase the AI system’s processing speed**  
- **D) To reduce the AI system’s energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
Human audits validate AI outputs and processes, ensuring they align with ethical and regulatory standards.

- **Options A, C, and D:** These are not objectives of human audits in responsible AI.

---

### Domain 5: Security, Compliance, and Governance for AI Solutions

### **Question #1**  
Which AWS service is primarily used for managing access and permissions for AI systems?

**Options:**  
- **A) Amazon S3**  
- **B) AWS IAM**  
- **C) Amazon EC2**  
- **D) Amazon RDS**  

**Correct Answer:** **B**  
**Explanation:**  
AWS IAM (Identity and Access Management) is the primary AWS service for managing access and permissions, allowing for secure role-based access to AWS resources.

- **Options A, C, and D:** These services are not used for managing access and permissions.

---

### **Question #2**  
What is the primary purpose of Amazon Macie in AI security?

**Options:**  
- **A) To generate AI models**  
- **B) To discover and protect sensitive data**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
Amazon Macie uses machine learning to identify and protect sensitive data, supporting compliance and security practices.

- **Options A, C, and D:** These are not functions of Amazon Macie.

---

### **Question #3**  
What does the AWS shared responsibility model refer to?

**Options:**  
- **A) Sharing AI models between customers**  
- **B) Division of security responsibilities between AWS and the customer**  
- **C) Sharing costs between AWS and the customer**  
- **D) Dividing AI tasks between humans and machines**  

**Correct Answer:** **B**  
**Explanation:**  
The shared responsibility model divides security tasks between AWS (handling infrastructure security) and the customer (responsible for data and application security).

- **Options A, C, and D:** These are unrelated to the AWS shared responsibility model.

---

### **Question #4**  
Which of the following is NOT a typical method for securing AI systems?

**Options:**  
- **A) Encryption**  
- **B) Access control**  
- **C) Public data sharing**  
- **D) Vulnerability management**  

**Correct Answer:** **C**  
**Explanation:**  
Public data sharing is not a security practice and could expose systems to risks. Encryption, access control, and vulnerability management are essential security measures.

- **Options A, B, and D:** These are standard practices for securing AI systems.

---

### **Question #5**  
What is data lineage in the context of AI security?

**Options:**  
- **A) A method of data encryption**  
- **B) Tracking the origin and transformations of data**  
- **C) A type of AI model architecture**  
- **D) A way to increase data processing speed**  

**Correct Answer:** **B**  
**Explanation:**  
Data lineage tracks the origin, modifications, and flow of data, which is essential for auditing and compliance.

- **Options A, C, and D:** These do not define data lineage in AI security.

---

### **Question #6**  
Which AWS service is used for detecting security threats in AI systems?

**Options:**  
- **A) Amazon Macie**  
- **B) Amazon S3**  
- **C) Amazon EC2**  
- **D) Amazon RDS**  

**Correct Answer:** **A**  
**Explanation:**  
Amazon Macie detects sensitive data and possible security risks, providing threat detection for AI and data storage.

- **Options B, C, and D:** These are not specifically designed for threat detection.

---

### **Question #7**  
What is prompt injection in the context of AI security?

**Options:**  
- **A) A method of improving prompt quality**  
- **B) A security vulnerability where malicious input manipulates the AI’s behavior**  
- **C) A technique for speeding up AI processing**  
- **D) A way to reduce AI energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
Prompt injection manipulates AI behavior by introducing malicious input, compromising the integrity of outputs.

- **Options A, C, and D:** These do not relate to prompt injection vulnerabilities.

---

### **Question #8**  
Which of the following is NOT a typical regulatory compliance standard for AI systems?

**Options:**  
- **A) ISO**  
- **B) SOC**  
- **C) HTML**  
- **D) Algorithm accountability laws**  

**Correct Answer:** **C**  
**Explanation:**  
HTML is a markup language, not a compliance standard. ISO, SOC, and algorithm accountability laws are relevant to regulatory compliance in AI.

- **Options A, B, and D:** These are recognized compliance standards.

---

### **Question #9**  
Which AWS service is used for continuous monitoring and assessment of resources?

**Options:**  
- **A) Amazon EC2**  
- **B) AWS Config**  
- **C) Amazon S3**  
- **D) Amazon RDS**  

**Correct Answer:** **B**  
**Explanation:**  
AWS Config continuously monitors AWS resources for configuration compliance and governance.

- **Options A, C, and D:** These services do not offer continuous monitoring for compliance.

---

### **Question #10**  
What is the primary purpose of AWS Artifact in AI governance?

**Options:**  
- **A) To generate AI models**  
- **B) To provide access to AWS compliance reports**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
AWS Artifact offers access to compliance documents and reports, aiding governance and regulatory efforts.

- **Options A, C, and D:** These are not functions of AWS Artifact.

---

### **Question #11**  
Which of the following is NOT typically part of a data governance strategy?

**Options:**  
- **A) Data lifecycle management**  
- **B) Data retention policies**  
- **C) Data public sharing policies**  
- **D) Data monitoring**  

**Correct Answer:** **C**  
**Explanation:**  
Data governance focuses on secure handling, management, and retention, not on sharing data publicly.

- **Options A, B, and D:** These are standard components of a data governance strategy.

---

### **Question #12**  
What is the primary purpose of AWS CloudTrail in AI governance?

**Options:**  
- **A) To generate AI models**  
- **B) To log API calls and account activity**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
AWS CloudTrail logs user activity, supporting governance by tracking API calls and account actions.

- **Options A, C, and D:** These are not functions of CloudTrail.

---

### **Question #13**  
Which of the following is a key consideration in secure data engineering for AI?

**Options:**  
- **A) Maximizing data collection without regard to quality**  
- **B) Implementing privacy-enhancing technologies**  
- **C) Making all data publicly accessible**  
- **D) Using only unencrypted data storage**  

**Correct Answer:** **B**  
**Explanation:**  
Privacy-enhancing technologies safeguard data, ensuring compliance and security in AI.

- **Options A, C, and D:** These approaches do not align with secure data engineering practices.

---

### **Question #14**  
What is the primary purpose of the Generative AI Security Scoping Matrix?

**Options:**  
- **A) To generate AI models**  
- **B) To provide a framework for assessing AI security risks**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
The Generative AI Security Scoping Matrix helps identify and assess potential security risks within generative AI applications.

- **Options A, C, and D:** These do not reflect the purpose of the Security Scoping Matrix.

---

### **Question #15**  
Which AWS service is used for automated security assessments?

**Options:**  
- **A) Amazon EC2**  
- **B) Amazon Inspector**  
- **C) Amazon S3**  
- **D) Amazon RDS**  

**Correct Answer:** **B**  
**Explanation:**  
Amazon Inspector conducts security assessments, identifying vulnerabilities and compliance issues.

- **Options A, C, and D:** These services are not designed for automated security assessments.

---

### **Question #16**  
What is data residency in the context of AI governance?

**Options:**  
- **A) The physical location where data is stored**  
- **B) The duration for which data is kept**  
- **C) The speed at which data is processed**  
- **D) The format in which data is stored**  

**Correct Answer:** **A**  
**Explanation:**  
Data residency refers to the geographical location where data is stored, often for compliance with regional data protection regulations.

- **Options B, C, and D:** These options do not address data residency, which is concerned with physical storage location.

---

### **Question #17**  
Which of the following is NOT a typical consideration in AI application security?

**Options:**  
- **A) Threat detection**  
- **B) Vulnerability management**  
- **C) Maximizing public data sharing**  
- **D) Infrastructure protection**  

**Correct Answer:** **C**  
**Explanation:**  
Public data sharing can expose sensitive information, making it a risk rather than a security practice. AI application security typically focuses on threat detection, vulnerability management, and infrastructure protection.

- **Options A, B, and D:** These are all essential components of AI security.

---

### **Question #18**  
What is the primary purpose of AWS Trusted Advisor in AI governance?

**Options:**  
- **A) To generate AI models**  
- **B) To provide real-time guidance for improving AWS environment**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
AWS Trusted Advisor helps users optimize their AWS setup by offering real-time advice on security, cost efficiency, performance, and fault tolerance.

- **Options A, C, and D:** These are not related to AWS Trusted Advisor's function.

---

### **Question #19**  
Which of the following is a key aspect of data integrity in AI systems?

**Options:**  
- **A) Ensuring data remains unchanged and uncorrupted**  
- **B) Making all data publicly accessible**  
- **C) Using only the largest datasets available**  
- **D) Storing all data in a single location**  

**Correct Answer:** **A**  
**Explanation:**  
Data integrity involves keeping data accurate, consistent, and protected from unauthorized modifications.

- **Options B, C, and D:** These do not align with data integrity practices.

---

### **Question #20**  
What is the primary purpose of encryption at rest in AI security?

**Options:**  
- **A) To protect data while it’s being transmitted**  
- **B) To protect stored data**  
- **C) To increase data processing speed**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
Encryption at rest safeguards stored data by making it unreadable without decryption keys, ensuring data protection when it's not actively in use.

- **Option A:** Protecting data during transmission requires encryption in transit, not at rest.
- **Options C and D:** These are unrelated to encryption.

---

### **Question #21**  
Which of the following is NOT typically part of governance protocols for AI systems?

**Options:**  
- **A) Regular policy reviews**  
- **B) Team training requirements**  
- **C) Maximizing model complexity**  
- **D) Transparency standards**  

**Correct Answer:** **C**  
**Explanation:**  
Governance protocols focus on transparency, policy review, and team training rather than maximizing model complexity.

- **Options A, B, and D:** These are standard elements of AI governance.

---

### **Question #22**  
What is the primary purpose of AWS Audit Manager in AI governance?

**Options:**  
- **A) To generate AI models**  
- **B) To continuously audit AWS usage for compliance**  
- **C) To increase model performance**  
- **D) To reduce energy consumption**  

**Correct Answer:** **B**  
**Explanation:**  
AWS Audit Manager provides continuous auditing for compliance with regulatory standards, supporting governance.

- **Options A, C, and D:** These are unrelated to AWS Audit Manager’s primary function.

---

### **Question #23**  
Which of the following is a key consideration in AI infrastructure protection?

**Options:**  
- **A) Maximizing public access to AI systems**  
- **B) Implementing network security measures**  
- **C) Using only the largest available models**  
- **D) Storing all data in a single location**  

**Correct Answer:** **B**  
**Explanation:**  
Network security is essential for protecting the infrastructure hosting AI systems, preventing unauthorized access and attacks.

- **Options A, C, and D:** These do not contribute to infrastructure security.

---

### **Question #24**  
What is the primary purpose of data cataloging in AI governance?

**Options:**  
- **A) To make all data publicly accessible**  
- **B) To organize and inventory data assets**  
- **C) To increase data processing speed**  
- **D) To reduce data storage costs**  

**Correct Answer:** **B**  
**Explanation:**  
Data cataloging helps manage and organize data, supporting data governance by enabling users to locate and understand data assets.

- **Options A, C, and D:** These do not relate to the purpose of data cataloging.

---

### **Question #25**  
Which of the following is NOT typically a component of a data lifecycle management strategy?

**Options:**  
- **A) Data creation**  
- **B) Data retention**  
- **C) Data deletion**  
- **D) Data public sharing**  

**Correct Answer:** **D**  
**Explanation:**  
Data lifecycle management includes creation, retention, and deletion stages, with a focus on secure handling and compliance rather than public sharing.

- **Options A, B, and C:** These are standard components of data lifecycle management.

---

(Already done)From: https://www.certshero.com/amazon/aif-c01/practice-test

### **Question #1**  
A company wants to classify human genes into 20 categories based on gene characteristics. The company needs an ML algorithm to document how the inner mechanism of the model affects the output.

**Options:**  
- **A) Decision trees**  
- **B) Linear regression**  
- **C) Logistic regression**  
- **D) Neural networks**  

**Correct Answer:** **A. Decision trees**  
**Explanation:**  
Decision trees provide transparency and interpretability, making it possible to understand how each feature affects the output, which is essential for documenting model mechanisms.

- **Option B and C:** While linear and logistic regression offer interpretability, they may not perform well with complex categorization tasks.
- **Option D:** Neural networks are often black-box models, making it challenging to interpret how they reach their outputs.

---

### **Question #2**  
A law firm wants to build an AI application by using large language models (LLMs). The application will read legal documents and extract key points from the documents.

**Options:**  
- **A) Build an automatic named entity recognition system.**  
- **B) Create a recommendation engine.**  
- **C) Develop a summarization chatbot.**  
- **D) Develop a multi-language translation system.**  

**Correct Answer:** **C. Develop a summarization chatbot**  
**Explanation:**  
A summarization chatbot can read and condense information from legal documents, highlighting key points relevant to users.

- **Option A:** Named entity recognition focuses on identifying entities rather than summarizing content.
- **Option B and D:** A recommendation engine or translation system would not fulfill the summarization requirement.

---

### **Question #3**  
A company has built a chatbot that can respond to natural language questions with images. The company wants to ensure that the chatbot does not return inappropriate or unwanted images.

**Options:**  
- **A) Implement moderation APIs.**  
- **B) Retrain the model with a general public dataset.**  
- **C) Perform model validation.**  
- **D) Automate user feedback integration.**  

**Correct Answer:** **A. Implement moderation APIs**  
**Explanation:**  
Moderation APIs can filter content in real-time, ensuring that inappropriate or unwanted images are not shown to users.

- **Option B:** Retraining with a public dataset does not directly address content moderation.
- **Option C and D:** Validation and feedback integration do not guarantee real-time filtering of inappropriate images.

---

### **Question #4**  
A company is training a foundation model (FM). The company wants to increase the accuracy of the model up to a specific acceptance level.

**Options:**  
- **A) Decrease the batch size.**  
- **B) Increase the epochs.**  
- **C) Decrease the epochs.**  
- **D) Increase the temperature parameter.**  

**Correct Answer:** **B. Increase the epochs**  
**Explanation:**  
Increasing the number of epochs allows the model to learn from the data for more cycles, potentially improving accuracy.

- **Option A:** Decreasing batch size affects learning speed and variance but not directly accuracy.
- **Option C:** Decreasing epochs would likely reduce accuracy.
- **Option D:** Temperature adjustment affects randomness, not accuracy.

---

### **Question #5**  
A large retailer receives thousands of customer support inquiries about products every day. The customer support inquiries need to be processed and responded to quickly. The company wants to implement Agents for Amazon Bedrock.

**Options:**  
- **A) Generation of custom foundation models (FMs) to predict customer needs**  
- **B) Automation of repetitive tasks and orchestration of complex workflows**  
- **C) Automatically calling multiple foundation models (FMs) and consolidating the results**  
- **D) Selecting the foundation model (FM) based on predefined criteria and metrics**  

**Correct Answer:** **B. Automation of repetitive tasks and orchestration of complex workflows**  
**Explanation:**  
Amazon Bedrock agents can automate routine customer support tasks and manage multi-step workflows, improving response time and efficiency.

- **Option A:** Bedrock agents use pre-existing models and do not generate custom FMs.
- **Option C and D:** These options relate to model selection and aggregation rather than workflow automation.

---

### **Question #6**  
An accounting firm wants to implement a large language model (LLM) to automate document processing. The firm must proceed responsibly to avoid potential harms.

**Options (Select TWO):**  
- **A) Include fairness metrics for model evaluation.**  
- **B) Adjust the temperature parameter of the model.**  
- **C) Modify the training data to mitigate bias.**  
- **D) Avoid overfitting on the training data.**  
- **E) Apply prompt engineering techniques.**  

**Correct Answers:** **A. Include fairness metrics for model evaluation** and **C. Modify the training data to mitigate bias**  
**Explanation:**  
To responsibly deploy an LLM, it’s essential to incorporate fairness metrics and adjust training data to reduce bias, ensuring the model treats data equitably.

- **Option B:** Temperature adjustment affects output diversity, not fairness.
- **Option D:** Avoiding overfitting is beneficial but not specific to responsible AI.
- **Option E:** Prompt engineering improves performance but doesn’t address fairness or bias.

---

### **Question #7**  
A company is using few-shot prompting on a base model that is hosted on Amazon Bedrock. The model currently uses 10 examples in the prompt. The model is invoked once daily and is performing well. The company wants to lower the monthly cost.

**Options:**  
- **A) Customize the model by using fine-tuning.**  
- **B) Decrease the number of tokens in the prompt.**  
- **C) Increase the number of tokens in the prompt.**  
- **D) Use Provisioned Throughput.**  

**Correct Answer:** **B. Decrease the number of tokens in the prompt**  
**Explanation:**  
Reducing the number of tokens in the prompt can lower inference costs since fewer tokens translate to lower usage fees.

- **Option A:** Fine-tuning incurs additional costs and is unnecessary if performance is already satisfactory.
- **Option C:** Increasing tokens would raise costs.
- **Option D:** Provisioned Throughput is beneficial for high usage but does not lower cost for occasional usage.

---

From: https://www.examtopics.com/exams/amazon/aws-certified-ai-practitioner-aif-c01/view/

Here are answers and explanations for each question in a structured format:

---

### **Question #1**
A company makes forecasts each quarter to decide how to optimize operations to meet expected demand. The company uses ML models to make these forecasts. An AI practitioner is writing a report about the trained ML models to provide transparency and explainability to company stakeholders.  
**What should the AI practitioner include in the report to meet the transparency and explainability requirements?**

**Options:**  
- **A) Code for model training**  
- **B) Partial dependence plots (PDPs)**  
- **C) Sample data for training**  
- **D) Model convergence tables**  

**Correct Answer:** **B. Partial dependence plots (PDPs)**  
**Explanation:**  
PDPs are commonly used to provide insights into the relationships between features and the model's predictions, enhancing transparency and explainability. They help stakeholders understand how changes in specific variables affect predictions.

- **Option A:** Code is helpful but does not provide an interpretive visual explanation.
- **Option C:** Sample data is informative but doesn’t directly explain model behavior.
- **Option D:** Convergence tables show model training progress but do not provide stakeholder-friendly explainability.

---

### **Question #2**
A law firm wants to build an AI application using large language models (LLMs). The application will read legal documents and extract key points from the documents.  
**Which solution meets these requirements?**

**Options:**  
- **A) Build an automatic named entity recognition system.**  
- **B) Create a recommendation engine.**  
- **C) Develop a summarization chatbot.**  
- **D) Develop a multi-language translation system.**  

**Correct Answer:** **C. Develop a summarization chatbot**  
**Explanation:**  
A summarization chatbot is ideal for extracting and condensing key points from legal documents, providing users with concise and relevant information.

- **Option A:** Named entity recognition identifies specific entities but doesn’t summarize.
- **Option B:** A recommendation engine is unrelated to summarization.
- **Option D:** Translation does not fulfill the summarization requirement.

---

### **Question #3**
A company wants to classify human genes into 20 categories based on gene characteristics. The company needs an ML algorithm to document how the inner mechanism of the model affects the output.  
**Which ML algorithm meets these requirements?**

**Options:**  
- **A) Decision trees**  
- **B) Linear regression**  
- **C) Logistic regression**  
- **D) Neural networks**  

**Correct Answer:** **A. Decision trees**  
**Explanation:**  
Decision trees provide a visual and interpretable model structure, making it easier to understand how the model arrives at its classifications.

- **Option B and C:** These algorithms offer some interpretability but may not handle complex multi-category classification tasks effectively.
- **Option D:** Neural networks are typically less interpretable.

---

### **Question #4**
A company has built an image classification model to predict plant diseases from photos of plant leaves. The company wants to evaluate how many images the model classified correctly.  
**Which evaluation metric should the company use to measure the model's performance?**

**Options:**  
- **A) R-squared score**  
- **B) Accuracy**  
- **C) Root mean squared error (RMSE)**  
- **D) Learning rate**  

**Correct Answer:** **B. Accuracy**  
**Explanation:**  
Accuracy measures the proportion of correctly classified images, which is suitable for evaluating performance in image classification tasks.

- **Option A:** R-squared is more relevant for regression models.
- **Option C:** RMSE applies to continuous, not categorical, outputs.
- **Option D:** Learning rate is a training parameter, not a performance metric.

---

### **Question #5**
A company has thousands of customer support interactions per day and wants to analyze these interactions to identify frequently asked questions and develop insights.  
**Which AWS service can the company use to meet this requirement?**

**Options:**  
- **A) Amazon Lex**  
- **B) Amazon Comprehend**  
- **C) Amazon Transcribe**  
- **D) Amazon Translate**  

**Correct Answer:** **B. Amazon Comprehend**  
**Explanation:**  
Amazon Comprehend provides natural language processing capabilities to analyze large volumes of text, enabling the identification of frequently asked questions and insights.

- **Option A:** Amazon Lex is for building conversational interfaces, not for analyzing text.
- **Option C:** Amazon Transcribe converts speech to text but doesn’t analyze content.
- **Option D:** Amazon Translate is for language translation.

---

### **Question #6**
A company has a database of petabytes of unstructured data from internal sources. The company wants to transform this data into a structured format for machine learning (ML) tasks.  
**Which service will meet these requirements?**

**Options:**  
- **A) Amazon Lex**  
- **B) Amazon Rekognition**  
- **C) Amazon Kinesis Data Streams**  
- **D) AWS Glue**  

**Correct Answer:** **D. AWS Glue**  
**Explanation:**  
AWS Glue is designed to extract, transform, and load (ETL) large datasets, enabling data scientists to structure and prepare data for ML tasks.

- **Option A:** Amazon Lex is for conversational AI, not data transformation.
- **Option B:** Amazon Rekognition analyzes images and video, not unstructured data.
- **Option C:** Amazon Kinesis processes streaming data, not transforming unstructured data.

---

### **Question #7**
Which AWS service or feature can help an AI development team quickly deploy and consume a foundation model (FM) within the team's VPC?  

**Options:**  
- **A) Amazon Personalize**  
- **B) Amazon SageMaker JumpStart**  
- **C) PartyRock, an Amazon Bedrock Playground**  
- **D) Amazon SageMaker endpoints**  

**Correct Answer:** **B. Amazon SageMaker JumpStart**  
**Explanation:**  
Amazon SageMaker JumpStart allows for rapid deployment and usage of pre-trained foundation models in a secure environment like a VPC.

- **Option A:** Amazon Personalize is for personalized recommendations, not FMs.
- **Option C:** PartyRock, an Amazon Bedrock Playground, does not directly provide foundation models within a VPC.
- **Option D:** SageMaker endpoints are for deployed models but do not include pre-trained FMs.

---

### **Question #8**
A company wants to use a large language model (LLM) on Amazon Bedrock for sentiment analysis. The company wants to classify the sentiment of text passages as positive or negative.  
**Which prompt engineering strategy meets these requirements?**

**Options:**  
- **A) Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.**  
- **B) Provide a detailed explanation of sentiment analysis and how LLMs work in the prompt.**  
- **C) Provide the new text passage to be classified without any additional context or examples.**  
- **D) Provide the new text passage with a few examples of unrelated tasks, such as text summarization or question answering.**  

**Correct Answer:** **A. Provide examples of text passages with corresponding positive or negative labels in the prompt followed by the new text passage to be classified.**  
**Explanation:**  
Providing examples with labels in the prompt helps the LLM understand the classification task, making it more likely to correctly classify sentiment.

- **Option B:** Explanation of sentiment analysis is unnecessary and may confuse the model.
- **Option C:** Lack of context may reduce accuracy.
- **Option D:** Irrelevant examples could hinder performance.

---

### **Question #9**
A company has installed a security camera. The company uses an ML model to evaluate the security camera footage for potential thefts. The company has discovered that the model disproportionately flags people who are members of a specific ethnic group.  
**Which type of bias is affecting the model output?**

**Options:**  
- **A) Measurement bias**  
- **B) Sampling bias**  
- **C) Observer bias**  
- **D) Confirmation bias**  

**Correct Answer:** **B. Sampling bias**  
**Explanation:**  
Sampling bias occurs when the training data does not adequately represent the population, which can lead the model to disproportionately flag certain groups. This issue may arise if the data used to train the model was unbalanced across ethnic groups.

- **Option A:** Measurement bias occurs if there are inconsistencies in data collection.
- **Option C:** Observer bias is the subjective influence of the person collecting data, not relevant to automated models.
- **Option D:** Confirmation bias is related to interpreting outcomes based on expectations, which is unrelated here.

---

### **Question #10**
A company wants to make a chatbot to help customers. The chatbot will help solve technical problems without human intervention. The company chose a foundation model (FM) for the chatbot. The chatbot needs to produce responses that adhere to company tone.  
**Which solution meets these requirements?**

**Options:**  
- **A) Set a low limit on the number of tokens the FM can produce.**  
- **B) Use batch inferencing to process detailed responses.**  
- **C) Experiment and refine the prompt until the FM produces the desired responses.**  
- **D) Define a higher number for the temperature parameter.**  

**Correct Answer:** **C. Experiment and refine the prompt until the FM produces the desired responses.**  
**Explanation:**  
Refining the prompt helps tailor the FM's responses to match the company's tone by guiding the model's generation process, ensuring consistency with brand voice.

- **Option A:** Limiting tokens affects response length, not tone.
- **Option B:** Batch inferencing is for efficiency, not tone control.
- **Option D:** Increasing temperature adds randomness, which could disrupt tone consistency.

---

### **Question #11**
A student at a university is copying content from generative AI to write essays.  
**Which challenge of responsible generative AI does this scenario represent?**

**Options:**  
- **A) Toxicity**  
- **B) Hallucinations**  
- **C) Plagiarism**  
- **D) Privacy**  

**Correct Answer:** **C. Plagiarism**  
**Explanation:**  
Copying content from generative AI without proper attribution is an issue of plagiarism, a major concern in the responsible use of AI-generated content.

- **Option A:** Toxicity refers to harmful language, not copying content.
- **Option B:** Hallucinations involve AI producing inaccurate or fictitious information, unrelated to copying.
- **Option D:** Privacy concerns are about protecting user data, not plagiarism.

---

### **Question #12**
A company wants to use a pre-trained generative AI model to generate content for its marketing campaigns. The company needs to ensure that the generated content aligns with the company's brand voice and messaging requirements.  
**Which solution meets these requirements?**

**Options:**  
- **A) Optimize the model's architecture and hyperparameters to improve the model's overall performance.**  
- **B) Increase the model's complexity by adding more layers to the model's architecture.**  
- **C) Create effective prompts that provide clear instructions and context to guide the model's generation.**  
- **D) Select a large, diverse dataset to pre-train a new generative model.**  

**Correct Answer:** **C. Create effective prompts that provide clear instructions and context to guide the model's generation.**  
**Explanation:**  
Using prompts that include brand voice guidelines helps ensure that the content generated by the AI aligns with the company's messaging.

- **Option A and B:** Adjusting architecture or complexity may improve performance but not brand alignment.
- **Option D:** Pre-training with a diverse dataset may not focus on the specific brand voice.

---

### **Question #13**
An e-commerce company wants to build a solution to determine customer sentiments based on written customer reviews of products.  
**Which AWS services meet these requirements? (Select TWO)**

**Options:**  
- **A) Amazon Lex**  
- **B) Amazon Comprehend**  
- **C) Amazon Polly**  
- **D) Amazon Bedrock**  
- **E) Amazon Rekognition**  

**Correct Answers:** **B. Amazon Comprehend** and **D. Amazon Bedrock**  
**Explanation:**  
Amazon Comprehend is designed for sentiment analysis of text, while Amazon Bedrock provides access to foundation models, which can also be used for sentiment analysis in a flexible, scalable way.

- **Option A:** Amazon Lex is for building conversational bots, not sentiment analysis.
- **Option C:** Amazon Polly is for text-to-speech conversion.
- **Option E:** Amazon Rekognition is for image analysis, not text.

---

### **Question #14**
A company wants to deploy a conversational chatbot to answer customer questions. The chatbot is based on a fine-tuned Amazon SageMaker JumpStart model. The application must comply with multiple regulatory frameworks.  
**Which capabilities can the company show compliance for? (Select TWO)**

**Options:**  
- **A) Auto scaling inference endpoints**  
- **B) Threat detection**  
- **C) Data protection**  
- **D) Cost optimization**  
- **E) Loosely coupled microservices**  

**Correct Answers:** **B. Threat detection** and **C. Data protection**  
**Explanation:**  
Compliance with regulatory frameworks often requires data protection and threat detection to ensure security and privacy.

- **Option A:** Auto-scaling helps with demand management but does not ensure compliance.
- **Option D:** Cost optimization is not directly related to compliance.
- **Option E:** Loosely coupled microservices improve architecture flexibility, not regulatory compliance.

---

### **Question #15**
A company is building a customer service chatbot. The company wants the chatbot to improve its responses by learning from past interactions and online resources.  
**Which AI learning strategy provides this self-improvement capability?**

**Options:**  
- **A) Supervised learning with a manually curated dataset of good responses and bad responses**  
- **B) Reinforcement learning with rewards for positive customer feedback**  
- **C) Unsupervised learning to find clusters of similar customer inquiries**  
- **D) Supervised learning with a continuously updated FAQ database**  

**Correct Answer:** **B. Reinforcement learning with rewards for positive customer feedback**  
**Explanation:**  
Reinforcement learning allows the chatbot to adjust its responses based on rewards (e.g., positive feedback), promoting self-improvement over time.

- **Option A and D:** Supervised learning relies on pre-labeled data and does not provide dynamic self-improvement.
- **Option C:** Unsupervised learning identifies clusters but doesn’t facilitate response refinement based on interactions.

---

### **Question #16**
How can companies use large language models (LLMs) securely on Amazon Bedrock?

**Options:**  
- **A) Design clear and specific prompts. Configure AWS Identity and Access Management (IAM) roles and policies by using least privilege access.**  
- **B) Enable AWS Audit Manager for automatic model evaluation jobs.**  
- **C) Enable Amazon Bedrock automatic model evaluation jobs.**  
- **D) Use Amazon CloudWatch Logs to make models explainable and to monitor for bias.**  

**Correct Answer:** **A. Design clear and specific prompts. Configure AWS Identity and Access Management (IAM) roles and policies by using least privilege access.**  
**Explanation:**  
Clear prompts and least-privilege IAM policies improve security by restricting model usage and access, aligning with best practices for secure AI deployment.

- **Option B and C:** AWS Audit Manager and Amazon Bedrock evaluation jobs aid compliance, not direct security.
- **Option D:** Amazon CloudWatch logs are for monitoring but do not directly secure LLM usage.

---

### **Question #17**
A company has built a solution using generative AI to translate training manuals from English into other languages. The company wants to evaluate the accuracy of the solution by examining the text generated for the manuals.  
**Which model evaluation strategy meets these requirements?**

**Options:**  
- **A) Bilingual Evaluation Understudy (BLEU)**  
- **B) Root mean squared error (RMSE)**  
- **C) Recall-Oriented Understudy for Gisting Evaluation (ROUGE)**  
- **D) F1 score**  

**Correct Answer:** **A. Bilingual Evaluation Understudy (BLEU)**  
**Explanation:**  
BLEU is a widely used metric for evaluating the quality of translations by comparing generated text to reference translations.

- **Option B:** RMSE is for evaluating continuous errors in regression.
- **Option C:** ROUGE is used for summarization evaluation.
- **Option D:** F1 score is typically used for classification tasks.

---

### **Question #18**
A company is using an Amazon Bedrock base model to summarize documents for an internal use case. The company trained a custom model to improve the summarization quality.  
**Which action must the company take to use the custom model through Amazon Bedrock?**

**Options:**  
- **A) Purchase Provisioned Throughput for the custom model.**  
- **B) Deploy the custom model in an Amazon SageMaker endpoint for real-time inference.**  
- **C) Register the model with the Amazon SageMaker Model Registry.**  
- **D) Grant access to the custom model in Amazon Bedrock.**  

**Correct Answer:** **B. Deploy the custom model in an Amazon SageMaker endpoint for real-time inference**  
**Explanation:**  
Amazon Bedrock typically provides access to pre-trained foundation models; to use a custom model, it needs to be deployed on Amazon SageMaker, which can then serve real-time inference requests.

- **Option A:** Provisioned Throughput is not directly related to model deployment.
- **Option C:** Registering the model alone does not enable inference; deployment is required.
- **Option D:** Granting access alone is insufficient without deployment.

---

### **Question #19**
An AI practitioner wants to use a foundation model (FM) to design a search application. The search application must handle queries that have text and images.  
**Which type of FM should the AI practitioner use to power the search application?**

**Options:**  
- **A) Multi-modal embedding model**  
- **B) Text embedding model**  
- **C) Multi-modal generation model**  
- **D) Image generation model**  

**Correct Answer:** **A. Multi-modal embedding model**  
**Explanation:**  
A multi-modal embedding model can handle both text and image inputs by encoding them into a shared embedding space, which is ideal for search applications requiring cross-modal queries.

- **Option B:** A text embedding model only handles text.
- **Option C:** Multi-modal generation is for generating content rather than embedding.
- **Option D:** Image generation models create images, not handle multi-modal search.

---

### **Question #20**
Which option is a benefit of ongoing pre-training when fine-tuning a foundation model (FM)?

**Options:**  
- **A) Helps decrease the model's complexity**  
- **B) Improves model performance over time**  
- **C) Decreases the training time requirement**  
- **D) Optimizes model inference time**  

**Correct Answer:** **B. Improves model performance over time**  
**Explanation:**  
Ongoing pre-training allows the model to continually learn from new data, improving its performance and accuracy over time.

- **Option A:** Pre-training typically does not reduce model complexity.
- **Option C:** Ongoing pre-training usually increases training requirements.
- **Option D:** Inference time optimization is not directly affected by pre-training.

---

### **Question #21**
Which metric measures the runtime efficiency of operating AI models?

**Options:**  
- **A) Customer satisfaction score (CSAT)**  
- **B) Training time for each epoch**  
- **C) Average response time**  
- **D) Number of training instances**  

**Correct Answer:** **C. Average response time**  
**Explanation:**  
Average response time measures how quickly an AI model can generate results in real-time applications, directly reflecting runtime efficiency.

- **Option A:** CSAT is related to user satisfaction, not model runtime efficiency.
- **Option B:** Training time per epoch measures training efficiency, not runtime.
- **Option D:** Number of training instances pertains to data quantity, not runtime performance.

---

### **Question #22**
A company has terabytes of data in a database that the company can use for business analysis. The company wants to build an AI-based application that can build a SQL query from input text that employees provide. The employees have minimal experience with technology.  
**Which solution meets these requirements?**

**Options:**  
- **A) Generative pre-trained transformers (GPT)**  
- **B) Residual neural network**  
- **C) Support vector machine**  
- **D) WaveNet**  

**Correct Answer:** **A. Generative pre-trained transformers (GPT)**  
**Explanation:**  
GPT-based models are capable of translating natural language inputs into structured queries like SQL, making them suitable for non-technical users to generate SQL queries.

- **Option B:** Residual neural networks are mainly used for image recognition tasks.
- **Option C:** Support vector machines are not designed for natural language processing.
- **Option D:** WaveNet is primarily used for generating audio, not text-based applications like SQL generation.

---

### **Question #23**
Which strategy evaluates the accuracy of a foundation model (FM) that is used in image classification tasks?

**Options:**  
- **A) Calculate the total cost of resources used by the model.**  
- **B) Measure the model's accuracy against a predefined benchmark dataset.**  
- **C) Count the number of layers in the neural network.**  
- **D) Assess the color accuracy of images processed by the model.**  

**Correct Answer:** **B. Measure the model's accuracy against a predefined benchmark dataset**  
**Explanation:**  
Evaluating accuracy using a benchmark dataset is the standard method for assessing image classification performance.

- **Option A:** Cost of resources does not reflect accuracy.
- **Option C:** The number of layers is unrelated to performance evaluation.
- **Option D:** Color accuracy is not a general measure of classification accuracy.

---

### **Question #24**
A social media company wants to use a large language model (LLM) for content moderation. The company wants to evaluate the LLM outputs for bias and potential discrimination against specific groups or individuals.  
**Which data source should the company use to evaluate the LLM outputs with the LEAST administrative effort?**

**Options:**  
- **A) User-generated content**  
- **B) Moderation logs**  
- **C) Content moderation guidelines**  
- **D) Benchmark datasets**  

**Correct Answer:** **D. Benchmark datasets**  
**Explanation:**  
Benchmark datasets designed for evaluating bias provide a reliable and low-effort approach for assessing potential discrimination in model outputs.

- **Option A:** User-generated content would require extensive manual evaluation.
- **Option B:** Moderation logs may lack standardization, requiring more effort.
- **Option C:** Content moderation guidelines are not data sources and would not facilitate direct evaluation.

---
