# End2End-MediBot (LLM Powered Healthcare Assistant)
A production-ready Medibot that answers medical queries in simple language, leveraging modern DevOps practices including CI/CD, Dockerization, AWS deployment, and Git-based version control.

- LLM-powered medical query answering
- Retrieval-Augmented Generation (RAG) using Pinecone
- Fast inference using Groq LLM
- API integration with Tavily for web search
- End-to-end deployment using Docker and AWS
- CI/CD pipeline with GitHub Actions

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/vairavellingam/End2End-MediBot.git

git clone https://github.com/vairavellingam/End2End-MediBot.git
cd End2End-MediBot
```


### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```


```bash
conda activate medibot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone,groq & tavily api key credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TAVILY_API_KEY= "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```



### Techstack Used:

- Python
- LangChain
- LangGraph
- Flask
- Pinecone (Vector DB)
- Groq (LLM Inference)
- Docker
- AWS (EC2, ECR)
- GitHub Actions (CI/CD)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 820763418986.dkr.ecr.ap-south-1.amazonaws.com/medibot
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - GROQ_API_KEY
   - TAVILY_API_KEY
