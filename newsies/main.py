"""
newsies.main
"""

import sys
import uuid

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
        match sys.argv[1]:
            case "get-news":
                from .pipelines import get_news_pipeline

                get_news_pipeline(task_id=TASK_ID)
            case "analyze":
                from .pipelines import analyze_pipeline

                analyze_pipeline(task_id=TASK_ID)
            case "cli":
                from newsies.cli import cli_read_news

                date_archive: str = None
                if len(sys.argv) > 3:
                    date_archive = sys.argv[2]
                cli_read_news(date_archive)
            case _:
                usage()
