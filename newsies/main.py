"""
newsies.main
"""

import sys
import uuid
import os
import pwd


# pylint: disable=import-outside-toplevel


def usage():
    """usage"""
    print(
        """

                Newsies
          Interactive News Explorer

    SUGGESION:
        Use the ./scripts/newsies script - it has greater functionality

    USAGE:

        newsies [routine] [options]

        ROUTINES:
          * get-news - daily routine to gather and analyze news
          * cli <archive date> - interactive command-line agent
            - archive date in the form YYYY-MM-DD
            - defaults to today
          """
    )


def logo():
    """logo"""
    print(
        """
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
"""
    )


if __name__ == "__main__":
    logo()

    if len(sys.argv) < 2:
        usage()
    else:
        TASK_ID = str(uuid.uuid4())
        USER_ID = pwd.getpwuid(os.getuid())[0]
        match sys.argv[1]:
            case "get-news":
                from .pipelines import get_articles_pipeline
                from newsies.pipelines.task_status import TASK_STATUS

                TASK_STATUS[TASK_ID] = {
                    "session_id": "N/A",
                    "status": "queued",
                    "task": "get-articles",
                    "username": USER_ID,
                }
                get_articles_pipeline(task_id=TASK_ID)

            case "analyze":
                from .pipelines import analyze_pipeline
                from newsies.pipelines.task_status import TASK_STATUS

                TASK_STATUS[TASK_ID] = {
                    "session_id": "N/A",
                    "status": "queued",
                    "task": "analyze",
                    "username": USER_ID,
                }
                analyze_pipeline(task_id=TASK_ID)
            case "cli":
                from newsies.cli import cli_read_news

                cli_read_news()
            case _:
                usage()
