"""Exercise the running local stack. Uses Python's standard library and Docker CLI."""

import argparse
import datetime as dt
import hashlib
import hmac
import json
from pathlib import Path
import socket
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[1]


def compose(*args, input=None, timeout=60):
    result = subprocess.run(
        ['docker', 'compose', *args], cwd=ROOT, input=input,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode:
        # Never echo interpolated Compose configuration or environment credentials.
        raise RuntimeError(f'Docker Compose operation failed: {args[0]} (exit {result.returncode})')
    return result.stdout.strip()


def s3_request(method, endpoint, path, access_key, secret, region, payload=b''):
    """Sign a path-style S3 request using AWS Signature V4, targeting local SeaweedFS."""
    parsed = urlsplit(endpoint)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('Smoke test storage endpoint must be local HTTP.')
    now = dt.datetime.now(dt.timezone.utc)
    timestamp, day = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y%m%d')
    digest = hashlib.sha256(payload).hexdigest()
    uri = quote(path, safe='/~')
    headers = {
        'host': parsed.netloc,
        'x-amz-content-sha256': digest,
        'x-amz-date': timestamp,
    }
    signed = ';'.join(sorted(headers))
    canonical_headers = ''.join(f'{key}:{headers[key]}\n' for key in sorted(headers))
    canonical = '\n'.join([method, uri, '', canonical_headers, signed, digest])
    scope = f'{day}/{region}/s3/aws4_request'
    to_sign = '\n'.join(['AWS4-HMAC-SHA256', timestamp, scope,
                         hashlib.sha256(canonical.encode()).hexdigest()])
    key = ('AWS4' + secret).encode()
    for item in [day, region, 's3', 'aws4_request']:
        key = hmac.new(key, item.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    headers['Authorization'] = (
        f'AWS4-HMAC-SHA256 Credential={access_key}/{scope}, '
        f'SignedHeaders={signed}, Signature={signature}'
    )
    request = Request(endpoint.rstrip('/') + uri, data=payload if method == 'PUT' else None,
                      headers=headers, method=method)
    with urlopen(request, timeout=20) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-model', action='store_true',
                        help='Also require the configured model and generate a short response.')
    args = parser.parse_args()
    subprocess.run(['docker', 'info'], capture_output=True, check=True, timeout=15)
    config = json.loads(compose('config', '--format', 'json'))
    services = config['services']
    # Verify every published endpoint is reachable from host-run Python services.
    for name in ['postgres', 'valkey', 'kafka', 'seaweedfs', 'ollama']:
        for port in services[name].get('ports', []):
            with socket.create_connection(('127.0.0.1', int(port['published'])), timeout=5):
                pass
    print('PASS: host ports reachable')

    pg = services['postgres']['environment']
    sql = '''BEGIN;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
CREATE TEMP TABLE infra_vector_test (embedding vector(1024));
INSERT INTO infra_vector_test VALUES (array_fill(0.1::real, ARRAY[1024])::vector);
SELECT vector_dims(embedding) FROM infra_vector_test;
SELECT (embedding <=> embedding) < 0.00001 FROM infra_vector_test;
ROLLBACK;'''
    result = compose('exec', '-T', 'postgres', 'psql', '-X', '-v', 'ON_ERROR_STOP=1',
                     '-U', pg['POSTGRES_USER'], '-d', pg['POSTGRES_DB'], '-At', '-c', sql)
    if '1024' not in result.splitlines() or 't' not in result.splitlines():
        raise RuntimeError('Vector dimension or cosine-distance check failed.')
    print('PASS: PostgreSQL transaction and pgvector 1024-dimension round-trip')

    token = uuid.uuid4().hex
    cache_key = f'infra:smoke:{token}'
    # Credentials stay inside the container environment, out of host command arguments.
    cache = compose('exec', '-T', 'valkey', 'sh', '-ec',
                    'export VALKEYCLI_AUTH="$VALKEY_PASSWORD"; '
                    f'valkey-cli SET {cache_key} {token} EX 30 >/dev/null; '
                    f'valkey-cli GET {cache_key}; valkey-cli DEL {cache_key} >/dev/null')
    if cache != token:
        raise RuntimeError('Valkey round-trip failed.')
    print('PASS: authenticated Valkey write/read/delete')

    topics = compose('exec', '-T', 'kafka', '/opt/kafka/bin/kafka-topics.sh',
                     '--bootstrap-server', 'kafka:19092', '--list').splitlines()
    expected = {'doc.uploaded', 'doc.parsed', 'query.requested'}
    expected |= {f'{name}.dlq' for name in tuple(expected)}
    if not expected.issubset(topics):
        raise RuntimeError('Missing topics; run docker compose up kafka-init.')
    topic = f'infra.smoke.{token}'
    compose('exec', '-T', 'kafka', '/opt/kafka/bin/kafka-topics.sh',
            '--bootstrap-server', 'kafka:19092', '--create', '--topic', topic,
            '--partitions', '1', '--replication-factor', '1',
            '--config', 'retention.ms=60000')
    try:
        compose('exec', '-T', 'kafka', '/opt/kafka/bin/kafka-console-producer.sh',
                '--bootstrap-server', 'kafka:19092', '--topic', topic,
                input=token + '\n')
        received = compose('exec', '-T', 'kafka', '/opt/kafka/bin/kafka-console-consumer.sh',
                           '--bootstrap-server', 'kafka:19092', '--topic', topic,
                           '--from-beginning', '--max-messages', '1', '--timeout-ms', '15000')
        if received != token:
            raise RuntimeError('Kafka message round-trip failed.')
    finally:
        compose('exec', '-T', 'kafka', '/opt/kafka/bin/kafka-topics.sh',
                '--bootstrap-server', 'kafka:19092', '--delete', '--topic', topic)
    print('PASS: required topics, DLQs and Kafka produce/consume')

    storage = services['seaweedfs']['environment']
    storage_port = services['seaweedfs']['ports'][0]['published']
    endpoint = f'http://127.0.0.1:{storage_port}'
    path = f'/{storage["S3_BUCKET"]}/infra-smoke/{token}.txt'
    kwargs = dict(endpoint=endpoint, path=path, access_key=storage['AWS_ACCESS_KEY_ID'],
                  secret=storage['AWS_SECRET_ACCESS_KEY'], region='us-east-1')
    data = f'lexaior-infra-{token}'.encode()
    s3_request('PUT', payload=data, **kwargs)
    try:
        if s3_request('GET', **kwargs) != data:
            raise RuntimeError('Storage content mismatch.')
    finally:
        s3_request('DELETE', **kwargs)
    print('PASS: authenticated SeaweedFS upload/download/delete')

    ollama_port = services['ollama']['ports'][0]['published']
    base = f'http://127.0.0.1:{ollama_port}'
    with urlopen(base + '/api/tags', timeout=10) as response:
        json.load(response)
    with urlopen(base + '/v1/models', timeout=10) as response:
        models = json.load(response)['data']
    print('PASS: Ollama native and OpenAI-compatible APIs')
    if args.check_model:
        # Compose does not consume LLM_MODEL; read the generated local dotenv value.
        model = next(line.split('=', 1)[1] for line in (ROOT / '.env').read_text().splitlines()
                     if line.startswith('LLM_MODEL='))
        if model not in {item['id'] for item in models}:
            raise RuntimeError('Configured model is not installed; pull it first.')
        request = Request(base + '/v1/chat/completions', method='POST',
                          headers={'Content-Type': 'application/json'},
                          data=json.dumps({'model': model, 'messages': [
                              {'role': 'user', 'content': 'Reply with OK. /no_think'}],
                              'max_tokens': 64, 'stream': False}).encode())
        with urlopen(request, timeout=300) as response:
            answer = json.load(response)['choices'][0]['message']['content']
        if not answer or not answer.strip():
            raise RuntimeError('Model returned no answer text.')
        print('PASS: local model generated an answer')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, StopIteration,
            subprocess.SubprocessError, HTTPError) as error:
        # Avoid exposing subprocess stderr or credentials in diagnostics.
        print(f'FAIL: infrastructure smoke check ({type(error).__name__}). '
              'Check docker info, docker compose ps and service logs.', file=sys.stderr)
        sys.exit(1)
