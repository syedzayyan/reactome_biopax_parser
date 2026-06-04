import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that download files from the Reactome / UniProt APIs.",
    )
