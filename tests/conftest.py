import pytest


@pytest.fixture(scope="session")
def spark():
    from pipelines.spark import get_spark

    session = get_spark("tests")
    yield session
    session.stop()
