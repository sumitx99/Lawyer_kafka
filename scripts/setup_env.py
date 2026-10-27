"""Create a local .env with random credentials without overwriting existing settings."""

from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / '.env'
    if destination.exists():
        print('.env already exists; left unchanged.')
        return
    content = (root / '.env.example').read_text()
    lines = []
    for line in content.splitlines():
        key, separator, value = line.partition('=')
        if separator and value == 'change-me':
            line = f'{key}={secrets.token_hex(24)}'
        lines.append(line)
    # Exclusive creation prevents replacing credentials on repeated/concurrent runs.
    with destination.open('x') as handle:
        destination.chmod(0o600)
        handle.write('\n'.join(lines) + '\n')
    print('Created local .env with random credentials (not printed).')


if __name__ == '__main__':
    main()
