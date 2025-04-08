# usage
1. make sure you have Shell commands plugin in Obsidian
2. make sure you install "pyscript" foldr into you obsidian vault root directory

convert json styled mark down file into structured obsidian notes:
```
python py_scripts/obsidian_study_guide_generator/json_to_obsidian.py -i "{{file_name}}" 
```

add current folder's notes name to knowledge_index (in case you already have learning notes and want to prevent generate redundant learning notes)
```
python py_scripts/obsidian_study_guide_generator/UpdateKnowledgeIndex.py "{{folder_path:relative}}"
```

# compare the Deep Research result
check the result in deepResearchResults folder on five different platforms on same initial research prompt 

| provider   | research time (min) | quality                                                 | view source | cited source | require account |
| ---------- | ------------------- | ------------------------------------------------------- | ----------- | ------------ | --------------- |
| gemini     | 3                   | Cover most relevant topics | 146         | 86           | yes             |
| openai     | 12                  | Cite most relevant sources  | 31          | ?            | yes             |
| jina       | 5                   | descent writing                                         | 73          | 4            | no              |
| perplexity | 2                   | give more specific concept than grok                    | 19          | 19           | yes             |
| grok       | 2                   | give vague concept name like machine learning basics    | ?           | 13           | yes             |





# prompt
## prompt for generate study guide on complex topics
```
topic_name = "janus flow"
Develop a comprehensive roadmap to deeply understand {topic_name}, first try to find high quality papers that relevant to the concepts, extract concepts from papers.

This roadmap should first explain What are the potential benefits, such as career advancement, monetization opportunities, and research prospects, of acquiring knowledge in {topic_name}, then it should list all concepts involved in {topic_name}, organized from foundational to advanced levels.  

To ensure depth and precision, concept names must be specific—avoid broad terms like, probability, machine learning, AI, or text-to-image.  
Instead, focus on concrete techniques and components such as gradient descent, proximal policy optimization (PPO), multi-head attention, or multi-layer perceptron (MLP).  

For each concept, provide the following details:  

1. Concept Name: A precise and well-defined term.  
2. Prerequisite Concepts: Names of concepts that should be understood beforehand.  
3. Learning Resources: working website Links to materials (articles, tutorials, videos).  
4. High-Level Explanation: A brief but insightful overview of the concept, how the concept relevant to the target topicwhat is the application of this concept.  
5. Estimated Learning Time: An approximate duration required to grasp the concept (normally take days to fully understand).  
6. Category: The domain to which the concept belongs (e.g., optimization algorithm, transformer architecture, reinforcement learning, dataset processing).
```

## prompt for convert study guide into json format
```
return the result in json format 
{
"main_topic":{
	"concept_name": str,
	"category": str
	"explanation_motivation":"high level explanation and real-world application & opportunities",
	"approximate_study_time": str,
	"prerequisites": List[str],
	"learning_resources": List[str],
	"study_questions": List[str]
},
"prerequisites": {
	"xxx"{
		"concept_name": str,
		"category": str
		"explanation_motivation":"how this prerequisite involved in main_topic",
		"approximate_study_time": str,
		"prerequisites": List[str],
		"learning_resources": List[str],
		"study_questions": List[str]
	},
	...
}

}
```

## prompt for generate simple study guide: 
```
concept = "gradient descent"
I want to learn {concept}
help me generate a md file in this format
---
start-date: "2025-03-30"
deadline: "2025-04-02"
---

## Category
Neural Networks

## Time Needed
3 days

## Prerequisites
- [[Neural_Networks|Neural Networks]]
- [[Backpropagation|Backpropagation]]
- [[Loss_Functions_eg_Mean_Squared_Error|Loss Functions (e.g., Mean Squared Error)]]

## Explanation
Autoencoders are a type of neural network used to learn efficient representations of data, typically for the purpose of dimensionality reduction or feature learning. They consist of an encoder that compresses the input into a lower-dimensional representation and a decoder that reconstructs the original input from this representation. Understanding Autoencoders is crucial for grasping how Vector Quantization (VQ) Tokenizers work, as VQ-VAEs (Vector Quantized Variational Autoencoders) leverage the principles of autoencoders to learn discrete latent representations, which are essential for effective tokenization in generative models.

## Learning Resources
- https://www.tensorflow.org/tutorials/generative/autoencoder
- https://www.coursera.org/learn/deep-learning-specialization


## Practice Questions
- [ ] What are the main components of an autoencoder?
- [ ] How does the training process of an autoencoder work?
- [ ] What are the applications of autoencoders in machine learning?

## Notes

---
#concept #neural-networks
```

