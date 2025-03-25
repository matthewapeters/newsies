# Newsies

The Interactive News Explorer

!["Lewis Hine: Luigi, 6 years old, newsboy-beggar, Sacramento, California, 1915" by trialsanderrors is licensed under CC BY 2.0.](./docs/3952519005_4d3030935e_c.jpg)

This application provides a framework for conversational interaction with an AI-enhanced agent that makes finding the news most interesting to the user a hands-free experience.

It is anticipated that this will provide an API service for projects like Jarvus (<https://www.github.com/matthewapeters/gpt4all_agent>) which provides speech-to-text and text-to-speech services locally.

## Research Project

This is a personal research project using open-source and freely-available technologies.  It has been made available
for public viewing.  It may become open-source in the future.

 NOTE: Models are (c) their respective owners and may not be used commercially without appropriate licensing

## Application Flow

![application sequence](./docs/Newsies_API_Sequence.png)
[I use sequencediagram.org to generate UML Sequence Diagrams](<https://sequencediagram.org/>)

NOTE: I am using FastAPI as its APIs are largely self-documenting.

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

### Technologies Used in Newsies

![python logo](./docs/python_logo.png) Python  
![docker logo](./docs/docker.png) Docker  
![fastAPI](./docs/fastapi-logo-teal.png) FastAPI  
![Redis](./docs/redis-Logotype.svg) Redis  
![beautifulsoup logo](./docs/tap-beautifulsoup.png) Beautifulsoup4  
![huggingface logo](./docs/huggingface_logo-noborder.svg) HuggingFace  
![ChromaDB](./docs/chromadb-227103090-6624bf7d-9524-4e05-9d2c-c28d5d451481.png) ChromaDB  
