# Newsies

The Interactive News Explorer

!["Lewis Hine: Luigi, 6 years old, newsboy-beggar, Sacramento, California, 1915" by trialsanderrors is licensed under CC BY 2.0.](./docs/3952519005_4d3030935e_c.jpg)

This application provides a framework for conversational interaction with an AI-enhanced agent that makes finding the news most interesting to the user a hands-free experience.

It is anticipated that this will provide an API service for projects like Jarvus (<https://www.github.com/matthewapeters/gpt4all_agent>) which provides speech-to-text and text-to-speech services locally.

- [Newsies](#newsies)
  - [Research Project](#research-project)
  - [Application Flow](#application-flow)
  - [A Note on Run Tasks](#a-note-on-run-tasks)
  - [Command-Line](#command-line)
  - [Application Goals and Strategies](#application-goals-and-strategies)
    - [Goal](#goal)
    - [Strategies](#strategies)
      - [RAG](#rag)
      - [PEFT LoRA Adapter Model Fine Tuning](#peft-lora-adapter-model-fine-tuning)
  - [The LoRA Adapter Fine-Tuning Strategy](#the-lora-adapter-fine-tuning-strategy)
    - [Pipelines](#pipelines)
    - [Get Articles Pipeline](#get-articles-pipeline)
    - [Analyze Pipeline](#analyze-pipeline)
    - [LLM Training Pipeline](#llm-training-pipeline)
    - [Daily-Run Pipeline](#daily-run-pipeline)
  - [Article Clustering](#article-clustering)
  - [Pydep Graph of Newsies Modules](#pydep-graph-of-newsies-modules)
    - [Technologies Used in Newsies](#technologies-used-in-newsies)

## Research Project

This is a personal research project using open-source and freely-available technologies.  It has been made available
for public viewing.  It may become open-source in the future.

 NOTE: Models are (c) their respective owners and may not be used commercially without appropriate licensing

## Application Flow

![application sequence](./docs/Newsies_API_Sequence.png)
[I use sequencediagram.org to generate UML Sequence Diagrams](<https://sequencediagram.org/>)

NOTE: I am using FastAPI as its APIs are largely self-documenting.
![FastAPI API Documentation](./docs/FastAPI_docs.png)

## A Note on Run Tasks

Newsies uses a semaphore to ensure that only one task pipeline can run at a time.  Requesting multiple tasks while another task is running effectively queues the tasks.  While this solution is not fault-tolerant, the goal is to make it so - in the event of a cash, tasks should be resumable on service restart.  I am looking at using either Redis or a pickled state object towards this end.

![screenshot of queued and running tasks](./docs/tasks_api.png)

## Command-Line

Services can be started and stopped from the command line, and pipelines can be initiated from the CLI as well.

All commands start with invoking `./scripts/newsies <command>`.

```bash
$ ./scripts/newsies 
+----------------------------------------+
|                                        |
|             .......                    |
|            .~~~~~~~~..                 |
|           .~.~~~~~.......              |
|           .~~~~... ..   .              |
|           .~~~...~:~~~.                |
|           .~. .~+:+:~~.                |
|              ..:::+~~.                 |
|           ~:::~::~:~~~~~~~..           |
|         .=o=+==+~:+++++=====+:.        |
|        .o==++=+~~=====+===+=+=+.       |
|       ~o===+++~.+====++==++++++++~     |
|      ~o===+:~~.:o===++::+~::=++++=:    |
|      =o==+~~~~+===+:+++~::..::+==+:    |
|     .=oo=+:::::::~~~::+.~~  .+==++~    |
|      :====+:=o+.......  . .~==++:.     |
|      :o===++o=+~ ..... .~:+=+:~+.      |
|     .=oo::=+=o+~ .... .:+++:.          |
|     +ooo==+++++~ .... ..::~            |
|    :oo==oo++:::~  ....                 |
|    =oo==o=++::++  ......               |
|    .=ooooo=+:+:=. ... ...              |
|      .~::+++++++  .... ...             |
|        ~........ .....  .              |
|        ........ ......                 |
|        ...... .  ...  . .              |
|         ~... .. ..........             |
|         ~.....  ..........             |
|         ........  ..... .              |
|           ..... . .~.....              |
|            .~...~   .....              |
|             ~~...    ....              |
|             ~~~..    .~...             |
|            .~:~~.    .:..~             |
|          .~:~~~..    .::~~.            |
|        .~:::~...     .::~:~.           |
|       .::~~~..       .::::~~.          |
|                       ....             |
+----------------------------------------+

Newsies Usage

Usage: newsies [routine]

CLI Commands
 get-news   download latest news from apnews.com
 analyze    schedule model training based on story page-ranking
 train  train model based on story page-ranking
 cli    initiate interactive agent session using most recently trained model
 up     tart backend services (ChromaDB, Redis)
 down   stop backend services
 chroma-stats   show backend service status
 chroma-logs    show backend service logs
--------------
API Server
 serve  start the FastAPI server
 kill   terminate the FastAPI server
 api    launch FastAPI session (shows interactive API docs)$ ./scripts/newsies 

```

## Application Goals and Strategies

### Goal

The goal of Newsies is to provide an AI agent that can converse about daily news.  Towards this end, I have taken a couple of approaches, with varying degress of success.

### Strategies

#### RAG

In this approach, the application attempts to understand what the user is asking about.  It tries to retrieve relevant documents and feeds them as context to an LLM, which generates a response using the retrieved documents.  This is extremely limited due to a couple of concerns:  

- LLMs can be quite limitted in the extend of context they can consume  
- Determining the user's actual intent is nearly 90% of the challenge.  By the time you can craft a query for the Prompt Engineering, the LLM is really just trying to summarize the retrieved content.

#### PEFT LoRA Adapter Model Fine Tuning

This is the current approach.  With this approach, there is little to no prompt engineering - the model has a layer fine-tuned on specific data, and can be prompted against that data.  There are still challenges.  

- Time: a recent test to fine-tune a LoRA Adapter with one day's worth of stories took 23 hours to complete.  It will be difficult and costly to come up with a solution that is cost-effective to maintain this level of fine-tuning daily
- Training and Testing Data: Training data must be generated for each story.  How much training data, and how the results are structured require some forethought.
  - I am currently relying on leveraging the Flan-T5 model to generate n questions about each story and n questions about each Named Entity that is found within the story.  Prompt engineering must be used to ensure that the questions cannot be answered without reading the article.  The article and its various sections and headlines (linked text to the article) must be provided as context to Flan-T5.
  - Training / Testing data must be tokenized to be consumed by the model in training.  This is also a time consuming exercise.
  - Training / Testing data needs to be saved and retrievable, as it is expensive to regenerate.  I split the training data and test data 80:20, tokenize each, and store the tokenized and untokenized data as .parquet files.  The folder structure is tied to the date and hour of generation.
- Training should be addative and strategically selective.  In order to make the model useful, break the daily training effort into strategic batches.
  - Use a vector database (ChromaDB) to index the embeddings of stories, along with their tags and NERs
  - Employ a KNN clustering strategy to identify clusters of data.  Clusters shold not just be based on today's data, although the training data will.
  - Use the clusters to identify which stories' training data to use.  Schedule a training session for each cluster.  Training sessions are asynchronous.
- The application, at run-time, will use the most recently fine-tuned LoRA Adapter with the model, even if training is ongoing.  The bet is that most questions will be related to central stories.  Over time, less central stories will make it intod the model from the corpus.

## The LoRA Adapter Fine-Tuning Strategy

### Pipelines

There are several tasks in shepherding the information from articles into the LLM.  A design for organizing data in a systematically retrievable way must be implemented.  Large processes must be broken into atomic units of work that can be "committed" when complete -- meaning the system can detect that the unit of work is complete and does not need to be re-processed.  An interface for initiating processes and tracking their progress, success, and failures must be designed and implemented.  If necessary, steps need to be reversed and undone so features and corrections can be added to atomic tasks and pipelines and the data can be reprocessed.  This is the approach I have chosen to follow.

Pipelines define a sequence of atomic tasks that will run against a common collection of data representing a fairly well-defined and stable segment of the corpus.  I have chosen the Publish Date as the basis for clustering data.  Each pipeline processes a distinct family of tasks against a collection of the corpus data.  Under normal execution, each publish date is processed once through the pipeline, and the daily
routine will focus on handling the most recent collection of data.  Eventual feature updates will provide for chosing an arbitrary publish date and re-run pipelines -- the assumption for this is that articles
may be made available or found for a publish date after the date was initially processed.

The data collections include:

1) Individual Articles
2) Datasets or DataFrames of information extracted from Articles

The families of work are:

1) Retrieving and on-boarding articles.  This includes extracting NERs, creating article embeddings, and indexing articles into the high-dimensional database (ChromaDB)
2) Analyzing articles.  This includes clustering articles based on semantic similarity.
3) Fine-tuning the LLM with data extracted from the articles.  This is includes converting articles into data frames and data sets.

I have found the cleanest way to organize the code for pipelines is as follows:

1) Define a class that represents the collection as an object.
2) Define a module for managing the sequential application of tasks
3) Represent each task as a Visitor class.  The collection object is then visited by these in turn.  The visitor may iterate over constituent parts serially or it may parallelize the work.

### Get Articles Pipeline

As new articles are found and retrieved, they are registered with other articles sharing the same publish date (all dates are UTC).  This registration is maintained in Redis.  When an atomic task
for a Publish Date is complete, the publish date and its member articles are added dictionary that is pickled to disk specific for the task.  When a pipeline is run, it compares registered articles (including protentially newly discovered articles) against the completed articles.  Completed articles are skipped.  New articles are processed and the completed article is added to the pickled dictionary.  The pipeline
itself is not atomic, the task per unit of work is atomic.  When a task is complete for an article, the task is registered within the article object which is also pickled, providing a history of when the task was completed for the article, and allowing tasks to be interrupted mid-stream and resumed without re-processing each task for the article that has been completed.  The commit of the task is at the time the article
is pickled.

### Analyze Pipeline

### LLM Training Pipeline

### Daily-Run Pipeline

## Article Clustering

![articles clusterd by KNN](./docs/semanitc_clustering.png)

`One day's worth of articles clustered by KNN`

the `/dashboard/` endpoint shows how the lastest collection of articles cluster when KNN is applied across all of the articles collected to-date.  Initially, the goal is to break the heavy-lift of fine-tuning across clustered batches, but training based on date published is more direct and manageable.  It also provides for better control or training interruption and resumption, as publish dates are less changing than clusters.  There is one cluster created for stories that are quite dissimilar to others (represented by multiple colors in the cluster ring and a lack of connecting edges).  This cluster starts quite small but eventually grows larger than any onther cluster.  I am not sure what the basis for this phenonmenon is, as there is not a direct way of identifing _what_ clusters articles together. Articles cluster based on semantic embedding proximity - but the subject of their proximity cannot -- yet -- be retrieved to explain what they share in common.

## Pydep Graph of Newsies Modules

![pydep class dependencies graph for newsies](./docs/pydep_newsies_main.png)

### Technologies Used in Newsies

<table width="100%">
  <tr>
    <td>
      <img src="./docs/chromadb-227103090-6624bf7d-9524-4e05-9d2c-c28d5d451481.png" scale="20%"/>
    </td>
    <td>
      <img src="./docs/Cytoscape_Black.png" scale="33%"/>
    </td>
    <td>
      <img src="./docs/redis-Logotype.svg" scale="300%" />
    </td>
  </tr>

  <tr>
    <td>
      <img src="./docs/plotly_dash.png" scale="25%" />
    </td>
    <td>
      <img src="./docs/docker.png" scale="200%" />
    </td>
    <td>
      <img src="./docs/python_logo.png" scale="100%" />
    </td>
  </tr>

  <tr>
    <td>
      <img src="./docs/beautiful-soup.png" scale="99%" />
    </td>
    <td>
      <img src="./docs/fastapi-logo-teal.png" scale="100%" />
    </td>
    <td>
      <img src="./docs/huggingface_logo-noborder.svg" scale="200%" />
    </td>
  </tr>

</table>
