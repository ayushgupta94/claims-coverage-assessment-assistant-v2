from mongomock_motor import AsyncMongoMockClient

from app.api.security import extract_access_code, is_valid_access_code
from app.config import Settings
from app.db.repositories.usage_repository import UsageRepository


def make_settings(codes: str) -> Settings:
    return Settings(_env_file=None, openai_api_key="unused", access_codes=codes)


def test_valid_codes_are_accepted():
    settings = make_settings("alpha-123, beta-456")
    assert is_valid_access_code("alpha-123", settings)
    assert is_valid_access_code("beta-456", settings)


def test_wrong_or_missing_codes_are_rejected():
    settings = make_settings("alpha-123")
    assert not is_valid_access_code("alpha-12", settings)
    assert not is_valid_access_code(None, settings)
    assert not is_valid_access_code("", settings)


def test_fails_closed_when_no_codes_are_configured():
    assert not is_valid_access_code("anything", make_settings(""))


def test_code_is_read_from_header_or_bearer_token():
    assert extract_access_code({"x-access-code": "abc"}) == "abc"
    assert extract_access_code({"authorization": "Bearer abc"}) == "abc"
    assert extract_access_code({"authorization": "Basic abc"}) is None


async def test_daily_run_cap_is_enforced():
    usage = UsageRepository(AsyncMongoMockClient()["test_usage"])
    results = [await usage.try_consume_run(3) for _ in range(5)]
    assert results == [True, True, True, False, False]
    assert await usage.try_consume_run(0) is False
