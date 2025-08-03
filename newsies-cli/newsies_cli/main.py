"""
newsies.main
"""

import sys
import uuid
import os
import pwd


# pylint: disable=import-outside-toplevel,global-variable-undefined,global-statement


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


TASK_ID = None
USER_ID = None


def main():
    """Main CLI entry point"""
    logo()

    global TASK_ID
    global USER_ID

    if len(sys.argv) < 2:
        usage()
    else:
        TASK_ID = str(uuid.uuid4())
        USER_ID = pwd.getpwuid(os.getuid())[0]
        match sys.argv[1]:
            case "get-news":
                from newsies_scraper.pipelines.get_articles import get_articles_pipeline
                # Use Redis-based distributed task status for Kubernetes
                from newsies_common.redis_task_status import TASK_STATUS

                TASK_STATUS[TASK_ID] = {
                    "session_id": "N/A",
                    "status": "queued",
                    "task": "get-articles",
                    "username": USER_ID,
                }
                get_articles_pipeline(task_id=TASK_ID)

            case "analyze":
                from newsies_analyzer.pipelines.analyze import analyze_pipeline
                # Use Redis-based distributed task status for Kubernetes
                from newsies_common.redis_task_status import TASK_STATUS

                TASK_STATUS[TASK_ID] = {
                    "session_id": "N/A",
                    "status": "queued",
                    "task": "analyze",
                    "username": USER_ID,
                }
                analyze_pipeline(task_id=TASK_ID)
            case "cli":
                from newsies_cli.cli.main import cli_read_news

                cli_read_news()
            case _:
                usage()


if __name__ == "__main__":
    main()
