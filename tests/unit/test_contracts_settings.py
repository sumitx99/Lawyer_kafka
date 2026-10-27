from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.secrets_control import Settings
from pyconsumers.helpers.storage_helper import Storage


def credentials():
    return dict(postgres_password='test-db-secret', valkey_password='test-cache-secret',
                storage_access_key='test-access', storage_secret_key='test-storage-secret')


def test_settings_redact_secrets_and_escape_database_password():
    values = credentials()
    values['postgres_password'] = 'p@ss:/word#%'
    settings = Settings(_env_file=None, **values)
    assert settings.database_url.password == values['postgres_password']
    assert settings.database_url.drivername == 'postgresql+asyncpg'
    assert values['postgres_password'] not in repr(settings)
    assert values['postgres_password'] not in str(settings.database_url)


def test_secret_files_override_dotenv_but_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('\n'.join(f'{key.upper()}={value}' for key, value in credentials().items()))
    secrets_dir = tmp_path / 'secrets'
    secrets_dir.mkdir()
    (secrets_dir / 'postgres_password').write_text('mounted-secret')
    for key in credentials():
        monkeypatch.delenv(key.upper(), raising=False)
    settings = Settings(_env_file=env_file, _secrets_dir=secrets_dir)
    assert settings.postgres_password.get_secret_value() == 'mounted-secret'
    monkeypatch.setenv('POSTGRES_PASSWORD', 'runtime-secret')
    settings = Settings(_env_file=env_file, _secrets_dir=secrets_dir)
    assert settings.postgres_password.get_secret_value() == 'runtime-secret'


@pytest.mark.parametrize('override', [{'embedding_dimension': 768}, {'postgres_password': 'change-me'},
                                     {'valkey_password': ''}, {'storage_provider': 'minio'}])
def test_reject_invalid_settings(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **(credentials() | override))


def test_event_roundtrip_and_required_identity():
    event = RagEvent(event_type='doc.uploaded', project_id=uuid4(), doc_id=uuid4())
    assert RagEvent.model_validate_json(event.model_dump_json()) == event
    for change in [{'doc_id': None}, {'schema_version': 2}, {'query_id': uuid4()},
                   {'occurred_at': datetime(2026, 1, 1)}]:
        with pytest.raises(ValidationError):
            RagEvent.model_validate(event.model_dump() | change)
    with pytest.raises(ValidationError):
        RagEvent(event_type='query.requested', project_id=uuid4())


@pytest.mark.parametrize('key', ['', '/absolute.pdf', 'raw/../document.pdf'])
def test_storage_rejects_invalid_object_keys(key):
    with pytest.raises(ValueError):
        Storage._key(key)
