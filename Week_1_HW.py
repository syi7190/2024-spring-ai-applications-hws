from transformers import pipeline

#classifier = pipeline('sentiment-analysis')
#generator = pipeline('text-generation', model='distilgpt2')
#classifier2 = pipeline('zero-shot-classification')

#classifier_res = classifier("I've been waiting for a HuggingFace course my whole life.")

# generator_res = generator(
#     "In this course, we will teach you how to",
#     max_length=40,
#     num_return_sequences=3,
# )

# classifier2_res = classifier2(
#     "This is a course about Python list comprehension",
#     candidate_labels=["education", "politics", "business"],
# )

#print(classifier_res)
#print(generator_res)
#print(classifier2_res)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ In the ever-evolving field of Natural Language Processing (NLP), the quest for more intelligent, context-aware systems is ongoing. This is where Retrieval Augmented Generation (RAG) comes into the picture, addressing some of the limitations of traditional generative models. So, what drives the increasing adoption of RAG?
Firstly, RAG provides a solution for generating text that isn't just fluent but also factually accurate and information-rich. By combining retrieval models with generative models, RAG ensures that the text it produces is both well-informed and well-written. Retrieval models bring the "what"—the factual content—while generative models contribute the "how"—the art of composing these facts into coherent and meaningful language.
Secondly, the dual nature of RAG offers an inherent advantage in tasks requiring external knowledge or contextual understanding. For instance, in question-answering systems, traditional generative models might struggle to offer precise answers. In contrast, RAG can pull in real-time information through its retrieval component, making its responses more accurate and detailed.
Lastly, scenarios demanding multi-step reasoning or synthesis of information from various sources are where RAG truly shines. Think of legal research, scientific literature reviews, or even complex customer service queries. RAG's capability to search, select, and synthesize information makes it unparalleled in handling such intricate tasks.
In summary, RAG's hybrid architecture delivers superior text generation capabilities, making it an ideal choice for applications requiring depth, context, and factual accuracy.
While Retrieval Augmented Generation (RAG) offers a myriad of advantages, it is not without its share of challenges and limitations. One of the most evident drawbacks is the model complexity. Given that RAG combines both retrieval and generative components, the overall architecture becomes more intricate, requiring more computational power and making debugging more complex.
Another difficulty is in data preparation: making available clean, non-redundant text and then developing and testing an approach to chunk that text into pieces that will be useful to the generative model is not a simple activity. After all of that work, you then have to find an embedding model that performs well across a potentially large and diverse amount of information!
Engaging a Large Language Model (LLM) often requires prompt engineering - while RAG is able to better inform the generative model with high-quality retrieved information, that information often needs to be correctly framed for the LLM to generate high-quality responses.
Lastly, there's the performance trade-off. The dual nature of RAG—retrieving and then generating text—can increase latency in real-time applications. Decisions must be made about how to balance the depth of retrieval against the speed of response, especially in time-sensitive situations."""
print(summarizer(ARTICLE, max_length=150, min_length=30, do_sample=False))