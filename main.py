"""Repository-root command-line entry point for AdaptMolMAC."""

from AdaptMolMAC import cli


if __name__ == "__main__":
    cli.DEBUG = True
    cli.run()